import sys

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import soundfile

# Test of pitch-shift quality without PVC

if __name__ == "__main__":
    block_size = 4096
    n_blocks = 4
    
    FILT_SIZE = 8

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

    indices = np.arange(block_size // 2 + 1)
    
    window = np.hanning(block_size)
    
    t = 0
    for block in in_file.blocks(blocksize=block_size, overlap=(block_size-in_shift), always_2d=True):
        if block.shape[0] != block_size:
            block = np.pad(block, ((0,block_size-block.shape[0]),(0,0)))
        for channel in range(in_file.channels):
            fft = np.fft.rfft(block[:,channel] * window)
        
            fft_mag = np.abs(fft)
            fft_phase = np.angle(fft)
            
            if pitch_mult != 1:
                contour = np.maximum(scipy.ndimage.maximum_filter1d(fft_mag,FILT_SIZE),0.001)
                
                #plt.plot(contour)
                #plt.plot(magnitude)
                #plt.show()
                
                fft_mag = fft_mag / contour
            
                # stretch or squeeze in the frequency domain to perform pitch shifting
                fft_mag = np.interp(indices/pitch_mult,indices,fft_mag,0,0)
                fft_phase = np.interp(indices/pitch_mult,indices,fft_phase,period=np.pi*2)
                #phase = np.interp(indices/pitch_mult,indices,fft_phase,period=np.pi*2)
                
                fft_mag = fft_mag * contour
                
            fft = fft_mag * np.exp(1j*fft_phase)
            
            out_block = np.fft.irfft(fft) * window

            out_data[t:t+block_size,channel] += out_block    
            
        t += out_shift
    
    out_data = out_data / np.max(np.abs(out_data))
    
    soundfile.write(out_filename, out_data, rate)
    in_file.close()
