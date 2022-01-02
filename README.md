# pvc

This is a phase vocoder implementation written in Python. It can be used to time-stretch audio files. This program also implements independent pitch-shifting of audio in the frequency domain, with optional formant correction.

Dependencies: numpy, scipy, SoundFile

To use the JACK application you must also install tkinter and JACK-Client.

## Usage

Run the following command to view the available parameters:

```
python3 pvc.py --help
```

### JACK application

A basic JACK client is also implemented that can perform pitch shifting of audio in real time. It can be started using the following command:

```
python3 pvc_jack.py
```

You'll then need to connect the inputs and outputs up using QJackCtl (or an equivalent program) as with any other JACK application. To use the program under PipeWire, add ```pw-jack``` at the beginning of the above command (on Debian, you may need to install the pipewire-audio-client-libraries package).

## Limitations

Currently, this phase vocoder can create significant phase artifacts when pitch-shifting audio. A few alternate algorithms are provided that may be able to reduce this problem but so far I have not found a complete solution.

## References

Bernsee, Stephan M. "On the Importance of Formants in Pitch Shifting." *Stephan Bernsee's Blog,* 6 Feb. 2000, [blogs.zynaptiq.com/bernsee/formants-pitch-shifting/](http://blogs.zynaptiq.com/bernsee/formants-pitch-shifting/).

Bernsee, Stephan M. "Pitch Shifting Using The Fourier Transform." *Stephan Bernsee's Blog,* 21 Sep. 1999, [blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/](http://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/).

De Götzen, Amalia, et al. "Traditional (?) Implementations of a Phase-Vocoder: The Tricks of the Trade." *Proceedings of the COST G-6 Conference on Digital Audio Effects,* 7-9 Dec. 2000.

Dudas, Richard, and Cort Lippe. "The Phase Vocoder - Part I." *Cycling '74,* 2 Nov. 2006, [cycling74.com/tutorials/the-phase-vocoder-%E2%80%93-part-i](https://cycling74.com/tutorials/the-phase-vocoder-%E2%80%93-part-i).

Ellis, Dan. "A Phase Vocoder in Matlab." *Dan Ellis's Home Page,* 2002, [www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/](https://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/).

Laroche, Jean, and Mark Dolson. "Phase-Vocoder: About this phasiness business." *ResearchGate,* Nov. 1997, [www.researchgate.net/publication/3714372_Phase-vocoder_about_this_phasiness_business](https://www.researchgate.net/publication/3714372_Phase-vocoder_about_this_phasiness_business).

Sethares, William A. "A Phase Vocoder in Matlab." *sethares homepage,* [sethares.engr.wisc.edu/vocoders/phasevocoder.html](https://sethares.engr.wisc.edu/vocoders/phasevocoder.html).
