from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QPainter, QColor, QPen
import numpy as np
from audio_stream_worker import AudioStreamWorker


class AudioVisualiser(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.muted = False
        self.audio_data = np.zeros(512)  # Initialize with zeros
        self.init_ui()
        self.audio_thread = AudioStreamWorker()
        self.audio_thread.update_audio_data.connect(self.update_audio_data)
        self.audio_thread.start()

    def init_ui(self):
        self.setFixedSize(600, 200)
        self.setStyleSheet("background-color: black;")

    def update_audio_data(self, data):
        if not self.muted:
            self.audio_data = data
        else:
            self.audio_data = np.zeros_like(data)  # Silence when muted
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.muted:
            painter.setPen(QColor(255, 0, 0))
            painter.drawLine(self.rect().topLeft(), self.rect().bottomRight())
        else:
            width = self.rect().width()
            height = self.rect().height()
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)

            # Apply FFT to audio data
            fft_result = np.fft.rfft(self.audio_data)
            fft_magnitude = np.abs(fft_result)

            # Normalize the result and map it to the height of the widget
            fft_magnitude = fft_magnitude / np.max(fft_magnitude)  # Normalize
            fft_magnitude = np.nan_to_num(fft_magnitude)  # Replace NaNs with 0

            # Draw the frequency bars
            num_bars = len(fft_magnitude)
            bar_width = width / num_bars
            for i in range(num_bars):
                bar_height = fft_magnitude[i] * height
                painter.drawLine(int(i * bar_width), height, int(i * bar_width), height - int(bar_height))

    def toggle_mute(self, event):
        self.muted = not self.muted
        self.repaint()

    def closeEvent(self, event):
        self.audio_thread.stop()
        self.audio_thread.wait()
        event.accept()