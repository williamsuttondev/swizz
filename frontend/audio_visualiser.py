from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor
import pyaudio
import struct
import numpy as np


class AudioVisualiser(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.muted = False
        self.decibels = 0
        self.init_audio_stream()
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(600, 100)

    def init_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=44100,
                                  input=True,
                                  frames_per_buffer=1024)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.muted:
            painter.setPen(QColor(255, 0, 0))
            painter.drawLine(self.rect().topLeft(), self.rect().bottomRight())
        else:
            painter.setPen(Qt.green)
            height = self.rect().height()
            width = self.rect().width()
            line_height = int(self.decibels / 100 * height)
            painter.drawLine(0, height - line_height, width, height - line_height)

    def update(self):
        if not self.muted:
            data = self.stream.read(1024, exception_on_overflow=False)
            data_int = struct.unpack(str(2 * 1024) + 'B', data)
            self.decibels = max(data_int) - 128
        self.repaint()

    def toggle_mute(self):
        self.muted = not self.muted
        self.repaint()
