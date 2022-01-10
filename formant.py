import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt

class FormantCorr:
    """
    Removes formants before the pitch shifter performs
    pitch shifting, and re-applies them afterward.
    """
    def __init__(self, pitch_shifter, filter_size, pitch_mult):
        self.filter_size = filter_size
        self.contour = None
        self.pitch_mult = pitch_mult
        self.pitch_shifter = pitch_shifter
    
    def remove_formants(self, magnitude):
        # get a rough contour of the higher-energy parts of the frequency spectrum
        contour = np.maximum(
            scipy.ndimage.maximum_filter1d(magnitude, self.filter_size), 0.001
        )

        # divide these formants out of the signal, leaving the smaller-scale peaks
        magnitude = magnitude / contour
        
        self.contour = self.process_formants(contour, magnitude)
        
        return magnitude
    
    def process_formants(self, contour, magnitude):
        # apply formant shifting if requested
        if self.pitch_mult != 1:
            #indices = np.arange(contour.size)
            contour = np.interp(
                self.pitch_shifter.indices / self.pitch_mult, self.pitch_shifter.indices, contour, 0, 0
            )
        return contour
    
    def apply_formants(self, data):
        return data * self.contour

class FormantModifier(FormantCorr):
    """
    Allows shifting of individual formants
    for more complex vocal effects.
    """
    def __init__(self, pitch_shifter, filter_size, pitch_mult, formant_pitch):
        super().__init__(pitch_shifter, filter_size, pitch_mult)
        self.formant_pitch = formant_pitch
        
    def process_formants(self, contour, magnitude):        
        blurred = scipy.ndimage.gaussian_filter1d(contour, self.filter_size/2, mode="constant", cval=0)
        #peaks, prop = scipy.signal.find_peaks(blurred, prominence=0.01)
        peaks, prop = scipy.signal.find_peaks(blurred, height=0.1)
        #peaks2 = scipy.signal.argrelextrema(blurred, np.greater)
        
        #peaks = []
        #for peak in peaks2[0]:
        #    if blurred[peak] >= 0.1:
        #        peaks.append(peak)
        
        #plt.plot(self.freq,magnitude)
        #plt.plot(self.freq,contour)
        #plt.plot(self.freq,blurred)
        #plt.legend(["magnitude","contour","blurred"])
        #plt.scatter(self.freq[peaks], blurred[peaks])
        #plt.show()

        # Shift the formants
        
        end = 0
        n_peaks = len(peaks)
        new_contour = np.zeros(contour.size)

        for i, peak in enumerate(peaks):
            start = 0
            if i == 0:
                start = 0
            else:
                start = end
                
            if i < n_peaks - 1:
                next_peak = peaks[i + 1]
                end = np.argmin(blurred[peak:next_peak]) + peak - 1
            else:
                end = blurred.size

            # TODO: doesn't work right with pitch_mult != 1
            if i < len(self.formant_pitch) and (self.formant_pitch[i] != 1 or self.pitch_mult != 1):
                layer = np.zeros(contour.size)
                layer[start:end] = contour[start:end]
                layer = np.interp(
                    self.pitch_shifter.indices / (self.formant_pitch[i] * self.pitch_mult), self.pitch_shifter.indices, layer, 0, 0
                )
                new_contour += layer
            else:
                new_contour[start:end] += contour[start:end]
        
        return new_contour