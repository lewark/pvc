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
        self.contour = np.maximum(
            scipy.ndimage.maximum_filter1d(magnitude, self.filter_size), 0.001
        )

        # divide these formants out of the signal, leaving the smaller-scale peaks
        magnitude = magnitude / self.contour
        
        return magnitude
    
    def process_formants(self):
        # apply formant shifting if requested
        if self.pitch_mult != 1:
            #indices = np.arange(contour.size)
            self.contour = np.interp(
                self.pitch_shifter.indices / self.pitch_mult, self.pitch_shifter.indices, contour, 0, 0
            )
    
    def apply_formants(self, data):
        self.process_formants()
        return data * self.contour

class FormantModifier(FormantCorr):
    """
    Allows shifting of individual formants
    for more complex vocal effects.
    """
    def __init__(self, filter_size, pitch_mult, formant_pitch):
        super().__init__(filter_size, pitch_mult):
        self.formant_pitch = formant_pitch
        
    def process_formants(self):        
        blurred = scipy.ndimage.gaussian_filter1d(self.contour, self.filter_size/2, mode="constant", cval=0)
        #peaks, prop = scipy.signal.find_peaks(blurred, prominence=0.01)
        peaks, prop = scipy.signal.find_peaks(blurred, height=0.1)
        #peaks2 = scipy.signal.argrelextrema(blurred, np.greater)
        
        #peaks = []
        #for peak in peaks2[0]:
        #    if blurred[peak] >= 0.1:
        #        peaks.append(peak)
        
        plt.plot(self.freq,magnitude)
        plt.plot(self.freq,contour)
        plt.plot(self.freq,blurred)
        plt.legend(["magnitude","contour","blurred"])
        plt.scatter(self.freq[peaks], blurred[peaks])
        plt.show()

        # TODO: Shift the formants

        super().process_formants()