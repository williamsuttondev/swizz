from PyQt5.QtCore import QThread, pyqtSignal
import pyaudio
import numpy as np

class AudioStreamWorker(QThread):
    update_decibels = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=44100,
                        input=True,
                        frames_per_buffer=512)
        while self.running:
            data = stream.read(512, exception_on_overflow=False)
            data_int = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(data_int)))
            decibels = 20 * np.log10(rms) if rms > 0 else -np.inf
            self.update_decibels.emit(decibels)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self.running = False
