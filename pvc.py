import argparse

import numpy as np
import scipy.signal
import soundfile

# import matplotlib.pyplot as plt

# Default parameters
D_BLOCK_SIZE = 4096
D_N_BLOCKS = 4
D_LENGTH_MULT = 1
D_PITCH_MULT = 1
D_F_PITCH_MULT = 1
D_F_FILTER_SIZE = 8
D_F_CORR = False


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
        # min_f2 = self.est_freqs_bruteforce(phase, dt)

        # err = np.abs(min_f2[1:] - min_f[1:])
        # err_q = np.max(err)

        # print(err)
        # if err_q > 0.001:
        #    err_i = np.argmax(err)
        #    print("err")
        #    #print("err",err_i, err_q, min_f[err_i], min_f2[err_i], self.freq[err_i])
        #    #print("diff",np.max(np.abs(min_f - self.freq)),np.max(np.abs(min_f2 - self.freq)))

        self.last_phase = phase

        return magnitude, phase, min_f

    def est_freqs_div(self, phase, dt):
        # TODO: This runs into problems at first bin, investigate

        # fn = (phase - self.last_phase + 2 * np.pi * n)/(2 * np.pi * dt)
        # solving for n:
        # fn - (phase - last_phase) / (2 * np.pi * dt) = (2 * np.pi * n) / (2 * np.pi * dt)
        # fn - (phase - last_phase) / (2 * np.pi * dt) = n / dt
        # (fn - (phase - last_phase) / (2 * np.pi * dt)) * dt = n

        freq_base = (phase - self.last_phase) / (2 * np.pi * dt)
        # n = np.round((self.freq - freq_base) / ((2 * np.pi) / (2 * np.pi * dt)))

        n = np.maximum(np.round((self.freq - freq_base) * dt), 0)
        # print(n)
        # print(freq_base, n, (n < 0).any())

        # if (n < 0).any():
        #    print("negative n", np.min(n), np.argmin(n))

        min_f = freq_base + (n / dt)

        # if (min_f < 0).any():
        #    i = np.argmin(min_f)
        #    print("negative", np.min(min_f), i, n[i], freq_base[i], self.freq[i] - freq_base[i], 1/dt)

        return min_f

    def est_freqs_bruteforce(self, phase, dt):
        min_diff = None
        min_f2 = np.zeros(self.freq.size)
        n = 0
        while True:
            fn = (phase - self.last_phase + 2 * np.pi * n) / (2 * np.pi * dt)
            diff = np.abs(fn - self.freq)

            if min_diff is None:
                min_diff = diff
                min_f2 = fn
            else:
                success = diff <= min_diff
                min_diff[success] = diff[success]
                min_f2[success] = fn[success]

            if (fn > self.freq).all():
                break

            n += 1
        return min_f2

    """
    def window_out(self, magnitude, phase):
        # TODO: figure out why this isn't working
        extrema = scipy.signal.argrelextrema(magnitude, np.greater)
        for peak in extrema[0]:
            n = 1
            while (peak - n >= 0):
                i = peak - n
                value = magnitude[i]
                if n > 1 and value > last_value:
                    break
                last_value = value
                phase[i] = (phase[peak] + np.pi * (i % 2))
                n += 1
            n = 1
            while (peak + n < phase.size):
                i = peak + n
                value = magnitude[i]
                if n > 1 and value > last_value:
                    break
                last_value = value
                phase[i] = (phase[peak] + np.pi * (i % 2))
                n += 1
    """

    def constrain_phase(self, phase):
        return ((phase + np.pi) % (np.pi * 2)) - np.pi

    def synthesize(self, magnitude, frequency, advance):
        dt = advance / self.samplerate

        out_phase = self.last_phase_out + 2 * np.pi * frequency * dt
        self.last_phase_out = out_phase

        # self.window_out(magnitude, out_phase)
        out_phase = self.constrain_phase(out_phase)

        fft = magnitude * np.exp(1j * out_phase)

        out_block = np.fft.irfft(fft) * self.window

        """#plt.subplot(311)
        ax1 = plt.subplot(311)
        #plt.plot(phase)
        #ax2 = plt.subplot(212,sharex=ax1,sharey=ax1)
        plt.plot(out_phase)
        ax3 = plt.subplot(312,sharex=ax1)
        plt.plot(magnitude)
        plt.subplot(313)
        plt.plot(out_block)
        plt.show()"""

        return out_block


class PitchShifter(PhaseVocoder):
    def __init__(
        self, samplerate, blocksize, pitch_mult, f_pitch_mult, f_corr, f_filter_size, linear,
    ):
        super().__init__(samplerate, blocksize)
        self.indices = np.arange(self.fft_size)
        self.pitch_mult = pitch_mult
        self.f_pitch_mult = f_pitch_mult
        self.f_corr = f_corr
        self.f_filter_size = f_filter_size
        
        self.linear = linear

    def process(self, block, in_shift, out_shift):
        magnitude, phase, frequency = self.analyze(block, in_shift)

        contour = None
        if self.f_corr and (self.pitch_mult != 1 or self.f_pitch_mult != 1):
            contour = np.maximum(
                scipy.ndimage.maximum_filter1d(magnitude, self.f_filter_size), 0.001
            )

            # divide the formants out of the signal, leaving the smaller-scale peaks
            magnitude = magnitude / contour

            # plt.plot(contour)
            # plt.plot(magnitude)
            # plt.show()

            if self.f_pitch_mult != 1:
                # todo: try doing this using discrete method?
                contour = np.interp(
                    self.indices / self.f_pitch_mult, self.indices, contour, 0, 0
                )

        if self.pitch_mult != 1:
            if self.linear:
                # stretch or squeeze in the frequency domain to perform pitch shifting
                # this works fine for integer multiples but causes phase artifacts otherwise
                magnitude = np.interp(
                    self.indices / self.pitch_mult, self.indices, magnitude, 0, 0
                )
                frequency = (
                    np.interp(self.indices / self.pitch_mult, self.indices, frequency, 0, 0)
                    * self.pitch_mult
                )
                # phase = np.interp(indices/pitch_mult,indices,fft_phase,period=np.pi*2)
            else:
                # https://stackoverflow.com/questions/4364823/how-do-i-obtain-the-frequencies-of-each-value-in-an-fft
                # Frequency: F = i * Fs / N
                # i = F * N / Fs
                
                # discrete pitch scaling seems to reduce phase artifacts in some cases
                # however, it still seems to be an issue when using formant shift
                new_freq = frequency * self.pitch_mult
                target_bins = np.round(new_freq * self.blocksize / self.samplerate).astype(int)
                
                valid = (target_bins < self.fft_size)
                
                new_mag = np.zeros(magnitude.size)
                new_freq_scaled = np.zeros(frequency.size)
                
                # TODO: try using a for loop for better behavior with pitch mult < 1
                new_mag[target_bins[valid]] = magnitude[valid]
                new_freq_scaled[target_bins[valid]] = new_freq[valid] #* magnitude[valid]
                #new_freq_scaled[new_mag > 0] /= new_mag[new_mag > 0]
                
                magnitude = new_mag
                frequency = new_freq_scaled

        if self.f_corr and (self.pitch_mult != 1 or self.f_pitch_mult != 1):
            # re-apply the formants
            magnitude = magnitude * contour

        out_block = self.synthesize(magnitude, frequency, out_shift)
        return out_block


class FileProcessor:
    def __init__(
        self,
        filename,
        block_size=D_BLOCK_SIZE,
        n_blocks=D_N_BLOCKS,
        length_mult=D_LENGTH_MULT,
        pitch_mult=D_PITCH_MULT,
        f_pitch_mult=D_F_PITCH_MULT,
        f_filter_size=D_F_FILTER_SIZE,
        f_corr=D_F_CORR,
        linear=False,
    ):
        self.block_size = block_size
        self.n_blocks = n_blocks

        # print(block_size, n_blocks, length_mult, pitch_mult, f_pitch_mult)

        self.length_mult = length_mult
        self.pitch_mult = pitch_mult
        self.f_pitch_mult = f_pitch_mult
        # Always enable formant correction if a formant pitch scale is given
        self.f_corr = f_corr or (f_pitch_mult != 1)
        self.f_filter_size = f_filter_size

        self.in_shift = self.block_size // self.n_blocks
        self.out_shift = int(self.in_shift * self.length_mult)

        self.in_file = soundfile.SoundFile(filename)
        self.rate = self.in_file.samplerate

        self.total_blocks = np.ceil(self.in_file.frames / self.in_shift)
        self.out_length = int(self.total_blocks * self.out_shift + self.block_size)

        self.out_data = np.zeros((self.out_length, self.in_file.channels))

        self.pvc = [
            PitchShifter(
                self.rate,
                self.block_size,
                self.pitch_mult,
                self.f_pitch_mult,
                self.f_corr,
                self.f_filter_size,
                linear,
            )
            for i in range(self.in_file.channels)
        ]

    def run(self):
        t = 0
        for block in self.in_file.blocks(
            blocksize=self.block_size,
            overlap=(self.block_size - self.in_shift),
            always_2d=True,
        ):
            if block.shape[0] != self.block_size:
                block = np.pad(block, ((0, self.block_size - block.shape[0]), (0, 0)))
            for channel in range(self.in_file.channels):
                out_block = self.process_block(block[:, channel], channel)
                self.out_data[t : t + self.block_size, channel] += out_block

            t += self.out_shift

        self.in_file.close()

        self.out_data = self.out_data / np.max(np.abs(self.out_data))

    def process_block(self, block, channel):
        out_block = self.pvc[channel].process(block, self.in_shift, self.out_shift)
        return out_block

    def write(self, filename):
        soundfile.write(filename, self.out_data, self.rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("in_file", type=str, help="the input audio file to process")
    parser.add_argument("out_file", type=str, help="the output audio file")
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        default=D_BLOCK_SIZE,
        help="the block size to use when processing",
    )
    parser.add_argument(
        "-n",
        "--n_blocks",
        type=int,
        default=D_N_BLOCKS,
        help="the number of overlapped windows per block",
    )
    parser.add_argument(
        "-l",
        "--length_mult",
        type=float,
        default=D_LENGTH_MULT,
        help="the factor to scale the length of the input file by",
    )
    parser.add_argument(
        "-p",
        "--pitch_mult",
        type=float,
        default=D_PITCH_MULT,
        help="the factor to scale the pitch of the input file by",
    )
    parser.add_argument(
        "-F",
        "--formant_corr",
        action="store_true",
        help="perform formant correction on the pitch shifted audio",
    )
    parser.add_argument(
        "-f",
        "--formant_mult",
        type=float,
        default=D_F_PITCH_MULT,
        help="the factor to scale the pitch of the formants of the input file by (implies -F when != 1)",
    )
    parser.add_argument(
        "-g",
        "--formant_filter",
        type=int,
        default=D_F_FILTER_SIZE,
        help="the size of the filter window to use when identifying formants",
    )
    parser.add_argument(
        "-L",
        "--linear",
        action="store_true",
        help="use linear interpolation on frequency spectrum",
    )

    args = parser.parse_args()

    processor = FileProcessor(
        args.in_file,
        args.block_size,
        args.n_blocks,
        args.length_mult,
        args.pitch_mult,
        args.formant_mult,
        args.formant_filter,
        args.formant_corr,
        args.linear,
    )
    processor.run()
    processor.write(args.out_file)
