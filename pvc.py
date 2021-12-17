import sys

import numpy as np
#import scipy.signal
import soundfile
#import matplotlib.pyplot as plt

class PhaseVocoder:
    def __init__(self, samplerate, blocksize):
        self.samplerate = samplerate
        self.blocksize = blocksize
        
        self.fft_size = (blocksize//2)+1
        
        self.last_phase = np.zeros(self.fft_size)
        self.last_phase_out = np.zeros(self.fft_size)
        
        self.window = np.hanning(blocksize)
        self.freq = np.fft.rfftfreq(blocksize,1/samplerate)
        
    def analyze(self, block, advance):
        in_block = block * self.window #np.fft.fftshift()
        fft = np.fft.rfft(in_block)
        
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        dt = advance / self.samplerate
        
        # TODO: optimize this?
        min_diff = None
        min_f = np.zeros(self.freq.size)
        n = 0
        while True:
            fn = (phase - self.last_phase + 2 * np.pi * n)/(2 * np.pi * dt)
            diff = np.abs(fn - self.freq)
            
            if min_diff is None:
                min_diff = diff
                
            success = diff < min_diff
            min_diff[success] = diff[success]
            min_f[success] = fn[success]
            
            if (fn > self.freq).all():
                break
            
            n += 1
        
        self.last_phase = phase

        return magnitude, phase, min_f
    
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
    
        #self.window_out(magnitude, out_phase)
        out_phase = self.constrain_phase(out_phase)
    
        fft = magnitude * np.exp(1j*out_phase)
        
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

if __name__ == "__main__":
    block_size = 4096
    n_blocks = 4

    if len(sys.argv) < 5:
        print("Usage: {} <in_filename> <out_filename> <length_mult> <pitch_mult> [block_size={}] [n_blocks={}]".format(
            sys.argv[0], block_size, n_blocks
        ))
        sys.exit()
    
    in_filename = sys.argv[1]
    out_filename = sys.argv[2]
    length_mult = float(sys.argv[3])
    pitch_mult = float(sys.argv[4])
    
    if len(sys.argv) >= 6:
        block_size = int(sys.argv[5])
    if len(sys.argv) >= 7:
        n_blocks = int(sys.argv[6])

    in_shift = block_size // n_blocks
    out_shift = int(in_shift * length_mult)

    in_file = soundfile.SoundFile(in_filename)
    rate = in_file.samplerate
    
    n_blocks = np.ceil(in_file.frames / in_shift)
    out_length = int(n_blocks * out_shift + block_size)
    
    #print("from", in_file.frames, "to", out_length)
    
    out_data = np.zeros((out_length,in_file.channels))

    pvc = [PhaseVocoder(rate, block_size) for i in range(in_file.channels)]

    indices = np.arange(pvc[0].fft_size)
    
    t = 0
    for block in in_file.blocks(blocksize=block_size, overlap=(block_size-in_shift), always_2d=True):
        if block.shape[0] != block_size:
            block = np.pad(block, ((0,block_size-block.shape[0]),(0,0)))
        for channel in range(in_file.channels):
            magnitude, phase, frequency = pvc[channel].analyze(block[:,channel], in_shift)
            
            if pitch_mult != 1:
                # stretch or squeeze in the frequency domain to perform pitch shifting
                magnitude = np.interp(indices/pitch_mult,indices,magnitude,0,0)
                frequency = np.interp(indices/pitch_mult,indices,frequency,0,0)*pitch_mult
                #phase = np.interp(indices/pitch_mult,indices,fft_phase,period=np.pi*2)
            
            out_block = pvc[channel].synthesize(magnitude, frequency, out_shift)
            out_data[t:t+block_size,channel] += out_block    
            
        t += out_shift
    
    out_data = out_data / np.max(np.abs(out_data))
    
    soundfile.write(out_filename, out_data, rate)
    in_file.close()
