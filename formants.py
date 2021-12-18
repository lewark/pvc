import sys

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import soundfile

import pvc

# TODO: This doesn't work correctly, and grabs small peaks rather than the larger-scale formants

if __name__ == "__main__":
    block_size = 4096
    n_blocks = 4
    vol_thresh = 0.01
    peak_border = 4

    #if len(sys.argv) < 5:
    #    print("Usage: {} <in_filename> <out_filename> <length_mult> <pitch_mult> [block_size={}] [n_blocks={}]".format(
    #        sys.argv[0], block_size, n_blocks
    #    ))
    #    sys.exit()
    
    in_filename = "audio/test.flac" #sys.argv[1]
    out_filename = "audio/test_formants.flac" #sys.argv[2]
    pitch_formant = [1,1,1,1,1] #float(sys.argv[4])
    pitch_shift = 1.5
    
    if len(sys.argv) >= 6:
        block_size = int(sys.argv[5])
    if len(sys.argv) >= 7:
        n_blocks = int(sys.argv[6])

    in_shift = block_size // n_blocks
    out_shift = in_shift

    in_file = soundfile.SoundFile(in_filename)
    rate = in_file.samplerate
    
    n_blocks = int(np.ceil(in_file.frames / in_shift))
    out_length = int(n_blocks * out_shift + block_size)
    
    #print("from", in_file.frames, "to", out_length)
    
    out_data = np.zeros(out_length)

    t_pvc = pvc.PhaseVocoder(rate, block_size)

    indices = np.arange(t_pvc.fft_size)
    
    print(t_pvc.fft_size, n_blocks)
    
    spectrogram = np.zeros((t_pvc.fft_size,n_blocks))
    out_spectrogram = np.zeros((t_pvc.fft_size,n_blocks))
    p_x = []
    p_y = []
    
    index = 0
    t = 0
    for block in in_file.blocks(blocksize=block_size, overlap=(block_size-in_shift)):
        if block.shape[0] != block_size:
            block = np.pad(block, ((0,block_size-block.shape[0]),(0,0)))
        magnitude, phase, frequency = t_pvc.analyze(block, in_shift)
        
        spectrogram[:,index] = np.log(magnitude)
        
        if np.max(np.abs(block)) >= vol_thresh:
            peaks, properties = scipy.signal.find_peaks(magnitude, prominence=8) #, height=np.mean(magnitude))
            
            print(peaks)
            
            #plt.plot(t_pvc.freq, magnitude)
            #plt.scatter(t_pvc.freq[peaks], magnitude[peaks])
            #plt.show()
            
            target_peaks = len(pitch_formant)
            n_peaks = len(peaks)
            
            p_x.extend((index,) * len(peaks))
            p_y.extend(peaks)
            
            #if n_peaks > target_peaks:

            #peaks2 = peaks[:target_peaks]
            
            last_peak = 0
            end = 0
            first_start = 0
            
            layers = []
            
            for i, peak in enumerate(peaks):
                if i >= target_peaks:
                    break
                
                start = 0
                if i == 0:
                    start = max(peak - peak_border,0)
                    first_start = start
                else:
                    start = np.argmin(magnitude[last_peak:peak])+last_peak
                if i < n_peaks - 1:
                    next_peak = peaks[i+1]
                    end = np.argmin(magnitude[peak:next_peak])+peak-1
                else:
                    end = min(peak + peak_border,magnitude.size-1)
                    
                layers.append((start,end,pitch_formant[i]))
                last_peak = peak
            
            layers.append((0,first_start,pitch_shift))
            layers.append((end,magnitude.size-1,pitch_shift))
            
            mask = np.zeros(magnitude.size)
            mag_sum = np.zeros(magnitude.size)
            freq_sum = np.zeros(magnitude.size)
            
            for layer in layers:
                start = layer[0]
                end = layer[1]
                pitch = layer[2]
                if end > start:
                    peak_mag = np.zeros(magnitude.size)
                    peak_freq = np.zeros(magnitude.size)
                    peak_mask = np.zeros(magnitude.size)
                    peak_mask[start:end] = 1
                    peak_mag[start:end] = magnitude[start:end]
                    peak_freq[start:end] = frequency[start:end]
                        
                    peak_mag = np.interp(indices/pitch,indices,peak_mag,0,0)
                    peak_freq = np.interp(indices/pitch,indices,peak_freq,0,0)*pitch
                    peak_mask = np.interp(indices/pitch,indices,peak_mask,0,0)
                    
                    mag_sum += peak_mag
                    freq_sum += peak_freq
                    mask += peak_mask
            
            to_modify = (mask > 0)
            freq_sum[to_modify] /= mask[to_modify]
            
            magnitude = mag_sum
            frequency = freq_sum
        
        out_spectrogram[:,index] = np.log(magnitude)
        
        out_block = t_pvc.synthesize(magnitude, frequency, out_shift)
        out_data[t:t+block_size] += out_block    
            
        t += out_shift
        index += 1
    
    in_file.close()
    
    out_data = out_data / np.max(np.abs(out_data))
    soundfile.write(out_filename, out_data, rate)
    
    plt.subplot(211)
    ax1 = plt.imshow(spectrogram, origin="lower") #, extent=[0,n_blocks*in_shift,0,t_pvc.freq[-1]]) #, aspect=0.01)
    plt.scatter(p_x, p_y)
    plt.ylim((0,100))
    plt.subplot(212)
    ax2 = plt.imshow(out_spectrogram, origin="lower")
    plt.ylim((0,100))
    #plt.yticks(t_pvc.freq)
    #plt.xticks(np.arange(n_blocks)*in_shift/rate)
    plt.show()
    
