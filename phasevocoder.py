import numpy as np
import scipy.signal

MATCH_PEAKS = True
PEAK_MAX_DIST = 16
PEAK_THRESH = 0.01


class PhaseVocoder:
    def __init__(self, samplerate, blocksize):
        self.samplerate = samplerate
        self.blocksize = blocksize

        self.fft_size = (blocksize // 2) + 1

        self.last_phase = np.zeros(self.fft_size)
        self.last_phase_out = np.zeros(self.fft_size)

        self.window = np.hanning(blocksize)
        self.freq = np.fft.rfftfreq(blocksize, 1 / samplerate)

    def analyze(self, block, advance):
        in_block = block * self.window  # np.fft.fftshift()
        fft = np.fft.rfft(in_block)

        magnitude = np.abs(fft)
        phase = np.angle(fft)

        dt = advance / self.samplerate

        min_f = self.est_freqs_div(phase, dt)

        self.last_phase = phase

        return magnitude, phase, min_f

    def est_freqs_div(self, phase, dt):
        # TODO: This runs into problems at first bin, investigate

        freq_base = (phase - self.last_phase) / (2 * np.pi * dt)
        n = np.maximum(np.round((self.freq - freq_base) * dt), 0)
        min_f = freq_base + (n / dt)

        return min_f

    def constrain_phase(self, phase):
        return ((phase + np.pi) % (np.pi * 2)) - np.pi

    def synthesize(self, magnitude, frequency, advance):
        dt = advance / self.samplerate

        out_phase = self.last_phase_out + 2 * np.pi * frequency * dt

        out_phase = self.constrain_phase(out_phase)

        self.last_phase_out = out_phase

        fft = magnitude * np.exp(1j * out_phase)

        out_block = np.fft.irfft(fft) * self.window

        return out_block


class PeakPhaseVocoder(PhaseVocoder):
    def __init__(self, samplerate, blocksize):
        super().__init__(samplerate, blocksize)

        self.last_peaks = []

    def compare(self, a, b):
        return np.greater(a, b + PEAK_THRESH)

    def analyze(self, block, advance):
        magnitude, phase, freq = super().analyze(block, advance)

        # peak_pos, prop = scipy.signal.find_peaks(magnitude, width=9)
        peak_pos = scipy.signal.argrelextrema(magnitude, self.compare, order=4)[0]
        # print(peak_pos)

        peak_start = []
        peak_end = []

        for i, peak in enumerate(peak_pos):
            start = 0
            if i != 0:
                start = np.argmin(magnitude[peak_pos[i - 1] : peak])
                # start = (peak_pos[i-1] + peak) // 2
                peak_end.append(start)
            peak_start.append(start)
        peak_end.append(magnitude.size)

        # plt.plot(magnitude)
        # plt.scatter(peak_pos, magnitude[peak_pos])
        # plt.show()

        return magnitude, phase, freq, list(zip(peak_pos, peak_start, peak_end))

    def synthesize(self, magnitude, phase, frequency, peaks, in_adv, out_adv):
        dt = out_adv / self.samplerate

        out_phase = np.zeros(phase.size)

        alpha = out_adv / in_adv
        # TODO: Also try beta = 1
        beta = alpha
        # print("last",self.last_peaks)
        for peak, peak_start, peak_end in peaks:
            old_peak = peak

            if MATCH_PEAKS:
                min_dist = None
                for last_peak, lp_start, lp_end in self.last_peaks:
                    dist = abs(peak - last_peak)
                    # print(dist)
                    if (min_dist == None or dist < min_dist) and dist <= PEAK_MAX_DIST:
                        min_dist = dist
                        old_peak = last_peak

            peak_phase = (
                self.last_phase_out[old_peak] + 2 * np.pi * frequency[peak] * dt
            )
            # force-lock partial phases to the peak phase
            out_phase[peak_start:peak] = peak_phase + beta * (
                phase[peak_start:peak] - phase[peak]
            )
            out_phase[peak + 1 : peak_end] = peak_phase + beta * (
                phase[peak + 1 : peak_end] - phase[peak]
            )
            out_phase[peak] = peak_phase

        out_phase = self.constrain_phase(out_phase)

        self.last_phase_out = out_phase
        self.last_peaks = peaks

        # self.window_out(magnitude, out_phase)

        fft = magnitude * np.exp(1j * out_phase)

        out_block = np.fft.irfft(fft) * self.window

        return out_block

