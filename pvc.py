import numpy as np
import scipy.signal
import soundfile

import matplotlib.pyplot as plt

class PhaseVocoder:
    def __init__(self, samplerate, blocksize):
        self.samplerate = samplerate
        self.blocksize = blocksize
        
        self.fft_size = (blocksize//2)+1
        
        #self.last_fft = None
        #self.last_mag = np.zeros(self.fft_size)
        self.last_phase = np.zeros(self.fft_size)
        self.last_phase_out = np.zeros(self.fft_size)
        
        self.window = np.hanning(BLOCKSIZE) #np.sin(np.arange(BLOCKSIZE)*np.pi/BLOCKSIZE)
        self.freq = np.fft.rfftfreq(blocksize,1/samplerate)
        
    def analyze(self, block, advance):
        in_block = block * self.window #np.fft.fftshift()
        fft = np.fft.rfft(in_block)
        
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        #if np.max(np.abs(block)) > 0.25:
        #    plt.plot(phase)
        #    plt.show()
        
        dt = advance / self.samplerate
        
        min_diff = None
        min_f = np.zeros(self.freq.size)
        n = 0
        while True:
            fn = (phase - self.last_phase + 2 * np.pi * n)/(2 * np.pi * dt)
            diff = np.abs(fn - self.freq)
            
            if min_diff is None:
                min_diff = diff
                last_diff = diff
                
            success = diff < min_diff
            min_diff[success] = diff[success]
            min_f[success] = fn[success]
            
            if (fn > self.freq).all():
                break
            
            n += 1
        
        #self.last_fft = fft
        #self.last_mag = magnitude
        self.last_phase = phase

        return magnitude, phase, min_f
    
    def window_out(self, magnitude, phase):
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
    
    def constrain_phase(self, phase):
        return ((phase + np.pi) % (np.pi * 2)) - np.pi
    
    def synthesize(self, magnitude, frequency, advance):
        dt = advance / self.samplerate
        
        out_phase = self.last_phase_out + 2 * np.pi * frequency * dt
        self.last_phase_out = out_phase
    
        #self.window_out(magnitude, out_phase)
        out_phase = self.constrain_phase(out_phase)
    
        fft = magnitude * np.exp(1j*out_phase)
        
        out_block = np.fft.irfft(fft) #* self.window
        
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

if __name__ == "__main__":
    INFILE="audio/test.flac"
    OUTFILE="audio/out2.flac"
    BLOCKSIZE = 4096
    
    #n_windows = 4

    overlap = BLOCKSIZE // 2

    out_blocks = []
    last_block = np.zeros(BLOCKSIZE) #for n in range(n_windows)]

    file = soundfile.SoundFile(INFILE)
    rate = file.samplerate

    pitch_mult = 2
    last_block = np.zeros(BLOCKSIZE)

    pvc = PhaseVocoder(rate, BLOCKSIZE)

    indices = np.arange(pvc.fft_size)
    
    for block in file.blocks(blocksize=BLOCKSIZE, overlap=overlap):
        if block.shape[0] == BLOCKSIZE:
            magnitude, phase, frequency = pvc.analyze(block, overlap)
            
            #magnitude = np.interp(indices/pitch_mult,indices,magnitude,0,0)
            #frequency = np.interp(indices/pitch_mult,indices,frequency,0,0)*pitch_mult
            
            #phase = np.interp(indices/pitch_mult,indices,fft_phase,period=np.pi*2)
            
            out_block = pvc.synthesize(magnitude, frequency, overlap)
            
            joined_block = last_block[overlap:] + out_block[:overlap]
            last_block = out_block
            
            out_blocks.append(joined_block)
        
    soundfile.write(OUTFILE, np.concatenate(out_blocks), rate)
