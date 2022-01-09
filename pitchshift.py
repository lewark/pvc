import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt

import phasevocoder
import formant


class PitchShifter(phasevocoder.PhaseVocoder):
    """
    Pitch-shifts the input signal by warping the frequency spectrum
    """

    def __init__(
        self,
        samplerate,
        blocksize,
        pitch_mult,
        f_pitch_mult,
        f_corr,
        f_filter_size,
        linear,
    ):
        super().__init__(samplerate, blocksize)
        self.indices = np.arange(self.fft_size)
        self.pitch_mult = pitch_mult
        self.f_pitch_mult = f_pitch_mult
        self.f_corr = f_corr
        self.f_filter_size = f_filter_size
        
        self.formant_corr = formant.FormantModifier(self, f_filter_size, f_pitch_mult, (1,1.2)) #formant.FormantCorr(self, f_filter_size, f_pitch_mult)

        self.linear = linear

    def process(self, block, in_shift, out_shift):
        magnitude, phase, frequency = self.analyze(block, in_shift)

        if self.f_corr and (self.pitch_mult != 1 or self.f_pitch_mult != 1):
            magnitude = self.formant_corr.remove_formants(magnitude)

        if self.pitch_mult != 1:
            if self.linear:
                # stretch or squeeze in the frequency domain to perform pitch shifting
                # this works fine for integer multiples but causes phase artifacts otherwise
                magnitude = np.interp(
                    self.indices / self.pitch_mult, self.indices, magnitude, 0, 0
                )
                frequency = (
                    np.interp(
                        self.indices / self.pitch_mult, self.indices, frequency, 0, 0
                    )
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
                target_bins = np.round(
                    new_freq * self.blocksize / self.samplerate
                ).astype(int)

                valid = target_bins < self.fft_size

                new_mag = np.zeros(magnitude.size)
                new_freq_scaled = np.zeros(frequency.size)

                # TODO: try using a for loop for better behavior with pitch mult < 1
                new_mag[target_bins[valid]] = magnitude[valid]
                new_freq_scaled[target_bins[valid]] = new_freq[
                    valid
                ]  # * magnitude[valid]
                # new_freq_scaled[new_mag > 0] /= new_mag[new_mag > 0]

                magnitude = new_mag
                frequency = new_freq_scaled

        if self.f_corr and (self.pitch_mult != 1 or self.f_pitch_mult != 1):
            # re-apply the formants
            magnitude = self.formant_corr.apply_formants(magnitude)

        out_block = self.synthesize(magnitude, frequency, out_shift)
        return out_block


class TimeDomainPitchShifter(phasevocoder.PhaseVocoder):
    """
    Combines phase-vocoder time stretching with
    time-domain interpolation to perform pitch shifting
    """

    def __init__(
        self,
        samplerate,
        blocksize,
        pitch_mult,
        f_pitch_mult,
        f_corr,
        f_filter_size,
    ):
        super().__init__(samplerate, blocksize)
        self.indices = np.arange(self.fft_size)
        self.pitch_mult = pitch_mult
        self.f_pitch_mult = f_pitch_mult
        self.f_corr = f_corr
        self.f_filter_size = f_filter_size

    def process(self, block, in_shift, out_shift):
        magnitude, phase, frequency = self.analyze(block, in_shift)

        # print(np.max(np.abs(frequency - self.freq)),np.mean(np.abs(frequency - self.freq)))

        new_length = int(round(self.blocksize / self.pitch_mult))

        if self.f_corr and (self.pitch_mult != 1 or self.f_pitch_mult != 1):
            contour = np.maximum(
                scipy.ndimage.maximum_filter1d(magnitude, self.f_filter_size), 0.001
            )
            
            

            # remove the formant resonances from the signal,
            # leaving the smaller-scale peaks
            magnitude = magnitude / contour

            # compensate for overall pitch shift
            pitch_mult = self.f_pitch_mult / (self.blocksize / new_length)
            contour = np.interp(self.indices / pitch_mult, self.indices, contour, 0, 0)

            # re-apply the formants
            magnitude = magnitude * contour

        out_block = self.synthesize(
            magnitude, frequency, out_shift * (self.blocksize / new_length)
        )

        # Resample the output block to pitch-shift it
        out_block = np.interp(
            np.arange(new_length) * self.pitch_mult,
            np.arange(self.blocksize),
            out_block,
            0,
            0,
        )

        return out_block


class PeakPitchShifter(phasevocoder.PeakPhaseVocoder):
    """
    Pitch shifts the input signal with frequencies phase-locked to peaks
    """

    def __init__(
        self,
        samplerate,
        blocksize,
        pitch_mult,
        f_pitch_mult,
        f_corr,
        f_filter_size,
    ):
        super().__init__(samplerate, blocksize)
        self.indices = np.arange(self.fft_size)
        self.pitch_mult = pitch_mult
        self.f_pitch_mult = f_pitch_mult
        self.f_corr = f_corr
        self.f_filter_size = f_filter_size

    def process(self, block, in_shift, out_shift):
        magnitude, phase, frequency, peaks = self.analyze(block, in_shift)

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
            # print(self.f_pitch_mult)
            if self.f_pitch_mult != 1:
                # todo: try doing this using discrete method?
                contour = np.interp(
                    self.indices / self.f_pitch_mult, self.indices, contour, 0, 0
                )

        if self.pitch_mult != 1:
            # https://stackoverflow.com/questions/4364823/how-do-i-obtain-the-frequencies-of-each-value-in-an-fft
            # Frequency: F = i * Fs / N
            # i = F * N / Fs

            # discrete pitch scaling seems to reduce phase artifacts in some cases
            new_freq = frequency * self.pitch_mult
            target_bins = np.round(new_freq * self.blocksize / self.samplerate).astype(
                int
            )

            valid = target_bins < self.fft_size

            new_mag = np.zeros(magnitude.size)
            new_phase = np.zeros(phase.size)
            new_freq_scaled = np.zeros(frequency.size)

            # Remap the PVC bins to apply pitch shift
            # TODO: try using a for loop for better behavior with pitch mult < 1
            new_mag[target_bins[valid]] = magnitude[valid]
            new_phase[target_bins[valid]] = phase[valid]
            new_freq_scaled[target_bins[valid]] = new_freq[valid]  # * magnitude[valid]
            # new_freq_scaled[new_mag > 0] /= new_mag[new_mag > 0]

            new_peaks = []
            # Remap peak bin indexes
            # TODO: deal with bin overlap when shifting downward
            for peak in peaks:
                peak_pos = peak[0]
                peak_start = peak[1]
                peak_end = peak[2]

                if peak_pos < target_bins.size:
                    peak_pos = target_bins[peak_pos]
                if peak_start < target_bins.size:
                    peak_start = target_bins[peak_start]
                if peak_end < target_bins.size:
                    peak_end = target_bins[peak_end]

                # print(peak_pos, peak_start, peak_end)

                if peak_pos >= magnitude.size or peak_start >= magnitude.size:
                    if len(new_peaks) > 0:
                        new_peaks[-1][2] = magnitude.size
                    continue

                if peak_end > magnitude.size:
                    peak_end = magnitude.size

                new_peaks.append([peak_pos, peak_start, peak_end])

            peaks = new_peaks
            magnitude = new_mag
            phase = new_phase
            frequency = new_freq_scaled

        if self.f_corr and (self.pitch_mult != 1 or self.f_pitch_mult != 1):
            # re-apply the formants
            magnitude = magnitude * contour

        out_block = self.synthesize(
            magnitude, phase, frequency, peaks, in_shift, out_shift
        )
        return out_block
