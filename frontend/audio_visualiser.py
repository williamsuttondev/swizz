from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer, pyqtSignal, Qt
from PyQt5.QtGui import QPainter, QColor
import numpy as np
from audio_stream_worker import AudioStreamWorker

class AudioVisualiser(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.muted = False
        self.decibels = 0
        self.init_ui()
        self.audio_thread = AudioStreamWorker()
        self.audio_thread.update_decibels.connect(self.update_decibels)
        self.audio_thread.start()

    def init_ui(self):
        self.setFixedSize(600, 100)
        self.setStyleSheet("background-color: black;")

    def update_decibels(self, decibels):
        if not self.muted:
            self.decibels = decibels
        self.repaint()

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

            # Ensure decibels are non-negative for visualization purposes
            adjusted_decibels = max(self.decibels, 0)
            line_height = int((adjusted_decibels / 100) * height)
            painter.drawLine(0, height - line_height, width, height - line_height)

    def toggle_mute(self, event):
        self.muted = not self.muted
        self.repaint()

    def closeEvent(self, event):
        self.audio_thread.stop()
        self.audio_thread.wait()
        event.accept()
