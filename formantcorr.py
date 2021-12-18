import sys

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import soundfile

import pvc

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

    t_pvc = [pvc.PhaseVocoder(rate, block_size) for i in range(in_file.channels)]

    indices = np.arange(t_pvc[0].fft_size)
    
    t = 0
    for block in in_file.blocks(blocksize=block_size, overlap=(block_size-in_shift), always_2d=True):
        if block.shape[0] != block_size:
            block = np.pad(block, ((0,block_size-block.shape[0]),(0,0)))
        for channel in range(in_file.channels):
            magnitude, phase, frequency = t_pvc[channel].analyze(block[:,channel], in_shift)
            
            if pitch_mult != 1:
                contour = np.maximum(scipy.ndimage.maximum_filter1d(magnitude,FILT_SIZE),0.001)
                
                #plt.plot(contour)
                #plt.plot(magnitude)
                #plt.show()
                
                magnitude = magnitude / contour
            
                # stretch or squeeze in the frequency domain to perform pitch shifting
                magnitude = np.interp(indices/pitch_mult,indices,magnitude,0,0)
                frequency = np.interp(indices/pitch_mult,indices,frequency,0,0)*pitch_mult
                #phase = np.interp(indices/pitch_mult,indices,fft_phase,period=np.pi*2)
                
                magnitude = magnitude * contour
            
            out_block = t_pvc[channel].synthesize(magnitude, frequency, out_shift)
            out_data[t:t+block_size,channel] += out_block    
            
        t += out_shift
    
    out_data = out_data / np.max(np.abs(out_data))
    
    soundfile.write(out_filename, out_data, rate)
    in_file.close()
