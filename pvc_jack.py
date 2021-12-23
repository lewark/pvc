import threading

import numpy as np
import jack

import pvc


class PVCJack:
    def __init__(self):
        self.client = jack.Client("PVCJack")
        self.client.inports.register("input")
        self.client.outports.register("output")
        self.event = threading.Event()
        self.window_size = 4096
        self.n_windows = 4

        self.buffer = []
        self.buffer_readpos = 0
        self.buffer_samples = 0

        initial_delay = self.window_size * 2
        self.outbuffer = [
            np.zeros(self.client.blocksize)
            for i in range(int(np.ceil(initial_delay / self.client.blocksize)))
        ]
        self.outbuffer_writepos = initial_delay

        self.pvc = pvc.PitchShifter(self.client.samplerate, self.window_size, 1, 1, True, 8)
        
        self.client.set_process_callback(self.process)
        self.client.set_shutdown_callback(self.shutdown)

    def process(self, frames):
        """Abandon hope, all ye who enter here"""
        #print("process")
        assert frames == self.client.blocksize
        data = self.client.inports[0].get_array()
        self.buffer.append(np.array(data))
        self.buffer_samples += frames

        while self.buffer_samples - self.buffer_readpos >= self.window_size:
            #print("perform pvc")
            block = np.zeros(self.window_size)
            i = 0
            read = 0
            delay = self.buffer_readpos
            #print('begin read')
            while read < self.window_size:
                piece = self.buffer[i]
                if i == 0:
                    piece = piece[self.buffer_readpos :]
                n = min(piece.size, self.window_size - i)
                block[i : i + n] = piece[:n]
                read += n
                i += 1
                #print(i, n, piece.size, self.window_size)
            #print('finish reading')

            advance = self.window_size // self.n_windows

            out_block = self.pvc.process(block, advance, advance)

            i = 0
            written = 0
            delay = self.outbuffer_writepos
            while written < self.window_size:
                while len(self.outbuffer) < i + 1:
                    self.outbuffer.append(np.zeros(frames))

                piece = self.outbuffer[i]
                if delay < piece.size:
                    to_write = min(piece.size - delay, out_block.size - written)
                    piece[delay : delay + to_write] += out_block[written : written + to_write]
                    delay = 0
                    written += to_write
                else:
                    delay -= piece.size
                #print('writing')

                i += 1
                
            self.buffer_readpos += advance
            self.outbuffer_writepos += advance
        #print("finish")

        while len(self.buffer) > 0 and self.buffer_readpos > self.buffer[0].size:
            self.buffer_readpos -= self.buffer[0].size
            self.buffer_samples -= self.buffer[0].size
            del self.buffer[0]

        # TODO: This breaks if the server buffer size changes
        self.client.outports[0].get_array()[:] = self.outbuffer.pop(0)
        self.outbuffer_writepos -= frames

    def shutdown(self, status, reason):
        print("JACK shutdown!")
        print("status:", status)
        print("reason:", reason)
        self.event.set()

    def run(self):
        with self.client:
            try:
                self.event.wait()
            except KeyboardInterrupt:
                print("Interrupted by user")

if __name__ == "__main__":
    pvc_jack = PVCJack()
    pvc_jack.run()
