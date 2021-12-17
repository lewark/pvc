# pvc

This is a simple phase vocoder implementation written in Python. It can be used to time-stretch audio files. This program also implements independent pitch-shifting of audio in the frequency domain.

Dependencies: numpy, soundfile

## Usage
```
python3 pvc.py <in_filename> <out_filename> <length_mult> <pitch_mult> [block_size=4096] [n_blocks=4]
```

## References

Bernsee, Stephan M. "Pitch Shifting Using The Fourier Transform." *Stephan Bernsee's Blog,* 21 Sep. 1999, [blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/](http://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/).

De Götzen, Amalia, et al. "Traditional (?) Implementations of a Phase-Vocoder: The Tricks of the Trade." *Proceedings of the COST G-6 Conference on Digital Audio Effects,* 7-9 Dec. 2000.

Dudas, Richard, and Cort Lippe. "The Phase Vocoder - Part I." *Cycling '74,* 2 Nov. 2006, [cycling74.com/tutorials/the-phase-vocoder-%E2%80%93-part-i](https://cycling74.com/tutorials/the-phase-vocoder-%E2%80%93-part-i).

Ellis, Dan. "A Phase Vocoder in Matlab." *Dan Ellis's Home Page,* 2002, [www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/](https://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/).

Sethares, William A. "A Phase Vocoder in Matlab." *sethares homepage,* [sethares.engr.wisc.edu/vocoders/phasevocoder.html](https://sethares.engr.wisc.edu/vocoders/phasevocoder.html).
