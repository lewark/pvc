import numpy as np
import soundfile

import matplotlib.pyplot as plt

INFILE="test.flac"
OUTFILE="out.flac"
BLOCKSIZE = 4096

overlap = BLOCKSIZE//2

out_blocks = []
last_block = np.zeros(BLOCKSIZE)
window = np.sin(np.arange(BLOCKSIZE)*np.pi/BLOCKSIZE) #np.hanning(BLOCKSIZE)

file = soundfile.SoundFile(INFILE)
rate = file.samplerate

pitch_mult = 2

indices = np.arange(overlap)

for block in file.blocks(blocksize=BLOCKSIZE, overlap=overlap):
    if block.shape[0] == BLOCKSIZE:
        in_block = block * window

        fft = np.fft.fft(in_block)
        
        #print(np.max(np.abs(fft[:overlap]-fft[overlap:][::-1])))
        
        fft_pos = fft[:overlap]
        #fft_pos[:3] = 0
        #fft_pos = np.interp(indices/pitch_mult,indices,fft_pos,0,0)
        
        fft_mag = np.abs(fft_pos)
        fft_phase = np.angle(fft_pos)
        fft_mag = np.interp(indices/pitch_mult,indices,fft_mag,0,0)
        fft_phase = np.interp(indices/pitch_mult,indices,fft_phase,period=np.pi*2)
        
        fft_pos = fft_mag * np.exp(1j*fft_phase)
        
        
        #fft_neg = fft[overlap:]
        #fft_neg = np.interp((overlap-indices)/pitch_mult,overlap-indices,fft_neg,0,0)
        fft_neg = fft_pos[::-1]

        fft_out = np.concatenate((fft_pos, fft_neg))
        
        out_block = np.fft.ifft(fft_out).real * window
        
        #plt.plot(out_block)
        #plt.show()

        joined_block = last_block[overlap:] + out_block[:overlap]
        last_block = out_block
        
        out_blocks.append(joined_block)
    
soundfile.write(OUTFILE, np.concatenate(out_blocks), rate)