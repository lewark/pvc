import numpy as np
import soundfile

import matplotlib.pyplot as plt

class PhaseVocoder:
    def __init__(self, samplerate, blocksize):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.overlap = blocksize // 2
        
        self.last_fft = None
        self.last_mag = np.zeros(self.overlap+1)
        self.last_phase = np.zeros(self.overlap+1)
        self.last_phase_out = np.zeros(self.overlap+1)
        
        self.window = np.hanning(BLOCKSIZE) #np.sin(np.arange(BLOCKSIZE)*np.pi/BLOCKSIZE)
        self.freq = np.fft.rfftfreq(blocksize,1/samplerate)
        
    def analyze(self, block):
        in_block = block * self.window
        fft = np.fft.rfft(in_block)
        
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        
        if np.max(np.abs(block)) > 0.25:
            plt.plot(phase)
            plt.show()
        
        dt = self.blocksize / self.samplerate
        
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
        #self.last_phase = phase

        return magnitude, phase, min_f
    
    def synthesize(self, magnitude, frequency):
        dt = self.blocksize / self.samplerate
        
        phase = (self.last_phase_out + 2 * np.pi * frequency * dt) % (np.pi * 2)
        self.last_phase_out = phase
    
        # TODO: nearby phase adjustment
        fft = magnitude * np.exp(1j*phase)
        
        out_block = np.fft.irfft(fft)
        
        plt.subplot(311)
        plt.plot(phase)
        plt.subplot(312)
        plt.plot(magnitude)
        plt.subplot(313)
        plt.plot(out_block)
        plt.show()
        
        return out_block

if __name__ == "__main__":
    INFILE="audio/test.flac"
    OUTFILE="audio/out2.flac"
    BLOCKSIZE = 4096

    out_blocks = []
    last_block = np.zeros(BLOCKSIZE)

    file = soundfile.SoundFile(INFILE)
    rate = file.samplerate

    pitch_mult = 2
    last_block = np.zeros(BLOCKSIZE)

    pvc = PhaseVocoder(rate, BLOCKSIZE)

    indices = np.arange(pvc.overlap+1)
    
    for block in file.blocks(blocksize=BLOCKSIZE, overlap=pvc.overlap):
        if block.shape[0] == BLOCKSIZE:
            magnitude, phase, frequency = pvc.analyze(block)
            
            magnitude = np.interp(indices/pitch_mult,indices,magnitude,0,0)
            frequency = np.interp(indices/pitch_mult,indices,frequency,0,0)*pitch_mult
            #phase = np.interp(indices/pitch_mult,indices,fft_phase,period=np.pi*2)
            
            out_block = pvc.synthesize(magnitude, frequency)
            
            joined_block = last_block[pvc.overlap:] + out_block[:pvc.overlap]
            last_block = out_block
            
            out_blocks.append(joined_block)
        
    soundfile.write(OUTFILE, np.concatenate(out_blocks), rate)
