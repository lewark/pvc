import argparse

import numpy as np
import soundfile

# import matplotlib.pyplot as plt

import phasevocoder
import pitchshift

# Default parameters
D_BLOCK_SIZE = 2048  # 4096
D_N_BLOCKS = 4
D_LENGTH_MULT = 1
D_PITCH_MULT = 1
D_F_PITCH_MULT = 1
D_F_FILTER_SIZE = 8 #8
D_F_CORR = False


class FileProcessor:
    def __init__(
        self,
        filename,
        block_size=D_BLOCK_SIZE,
        n_blocks=D_N_BLOCKS,
        length_mult=D_LENGTH_MULT,
        pitch_mult=D_PITCH_MULT,
        f_pitch_mult=D_F_PITCH_MULT,
        f_filter_size=D_F_FILTER_SIZE,
        f_corr=D_F_CORR,
    ):
        self.block_size = block_size
        self.n_blocks = n_blocks

        # print(block_size, n_blocks, length_mult, pitch_mult, f_pitch_mult)

        self.length_mult = length_mult
        self.pitch_mult = pitch_mult
        self.f_pitch_mult = f_pitch_mult
        # Always enable formant correction if a formant pitch scale is given
        self.f_corr = f_corr or (f_pitch_mult != 1)
        self.f_filter_size = f_filter_size

        self.in_shift = self.block_size // self.n_blocks
        self.out_shift = int(self.in_shift * self.length_mult)

        self.in_file = soundfile.SoundFile(filename)
        self.rate = self.in_file.samplerate

        self.total_blocks = np.ceil(self.in_file.frames / self.in_shift)
        self.out_length = int(self.total_blocks * self.out_shift + self.block_size)

        self.out_data = np.zeros((self.out_length, self.in_file.channels))

        self.pvc = [
            pitchshift.PitchShifter(
                self.rate,
                self.block_size,
                self.pitch_mult,
                self.f_pitch_mult,
                self.f_corr,
                self.f_filter_size,
                True,
            )
            for i in range(self.in_file.channels)
        ]

    def run(self):
        t = 0
        for block in self.in_file.blocks(
            blocksize=self.block_size,
            overlap=(self.block_size - self.in_shift),
            always_2d=True,
        ):
            if block.shape[0] != self.block_size:
                block = np.pad(block, ((0, self.block_size - block.shape[0]), (0, 0)))
            for channel in range(self.in_file.channels):
                out_block = self.process_block(block[:, channel], channel)
                self.out_data[t : t + out_block.size, channel] += out_block

            t += self.out_shift

        self.in_file.close()

        self.out_data = self.out_data / np.max(np.abs(self.out_data))

    def process_block(self, block, channel):
        out_block = self.pvc[channel].process(block, self.in_shift, self.out_shift)
        return out_block

    def write(self, filename):
        soundfile.write(filename, self.out_data, self.rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("in_file", type=str, help="the input audio file to process")
    parser.add_argument("out_file", type=str, help="the output audio file")
    parser.add_argument(
        "-b",
        "--block_size",
        type=int,
        default=D_BLOCK_SIZE,
        help="the block size to use when processing",
    )
    parser.add_argument(
        "-n",
        "--n_blocks",
        type=int,
        default=D_N_BLOCKS,
        help="the number of overlapped windows per block",
    )
    parser.add_argument(
        "-l",
        "--length_mult",
        type=float,
        default=D_LENGTH_MULT,
        help="the factor to scale the length of the input file by",
    )
    parser.add_argument(
        "-p",
        "--pitch_mult",
        type=float,
        default=D_PITCH_MULT,
        help="the factor to scale the pitch of the input file by",
    )
    parser.add_argument(
        "-F",
        "--formant_corr",
        action="store_true",
        help="perform formant correction on the pitch shifted audio",
    )
    parser.add_argument(
        "-f",
        "--formant_mult",
        type=float,
        default=D_F_PITCH_MULT,
        help="the factor to scale the pitch of the formants of the input file by (implies -F when != 1)",
    )
    parser.add_argument(
        "-g",
        "--formant_filter",
        type=int,
        default=D_F_FILTER_SIZE,
        help="the size of the filter window to use when identifying formants",
    )

    args = parser.parse_args()

    processor = FileProcessor(
        args.in_file,
        args.block_size,
        args.n_blocks,
        args.length_mult,
        args.pitch_mult,
        args.formant_mult,
        args.formant_filter,
        args.formant_corr,
    )
    processor.run()
    processor.write(args.out_file)
