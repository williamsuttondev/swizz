import sys
import mss
import numpy as np
import pyaudio
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QApplication, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from monitor_selector import MonitorSelector

class AudioVisualiser(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.muted = False
        self.decibels = 0
        self.init_audio_stream()
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(600, 100)
        self.setStyleSheet("background-color: black;")

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

            # Ensure decibels are non-negative for visualization purposes
            adjusted_decibels = max(self.decibels, 0)
            line_height = int((adjusted_decibels / 100) * height)
            painter.drawLine(0, height - line_height, width, height - line_height)

    def update(self):
        if not self.muted:
            data = self.stream.read(1024, exception_on_overflow=False)
            data_int = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(data_int)))
            self.decibels = 20 * np.log10(rms) if rms > 0 else -np.inf
        self.repaint()

    def toggle_mute(self):
        self.muted = not self.muted
        self.repaint()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.selected_monitor = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_preview)

    def init_ui(self):
        self.setWindowTitle("swizz")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Initial State
        self.monitor_selector = MonitorSelector(self)
        self.layout.addWidget(self.monitor_selector)

        self.monitor_preview = QLabel("Please select a monitor preview", self)
        self.monitor_preview.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.monitor_preview)

        self.start_button_layout = QHBoxLayout()

        # Start Swizz Button (Initially hidden)
        self.start_swizz_button = QPushButton("Start Swizz", self)
        self.start_swizz_button.clicked.connect(self.start_swizz)
        self.start_button_layout.addWidget(self.start_swizz_button)
        self.start_swizz_button.hide()

        # Start Swizz Without Video Button (Visible initially)
        self.start_swizz_without_video_button = QPushButton("Start Swizz without Video", self)
        self.start_swizz_without_video_button.clicked.connect(self.start_swizz_without_video)
        self.start_button_layout.addWidget(self.start_swizz_without_video_button)

        self.layout.addLayout(self.start_button_layout)

        # Connect the monitor selection
        self.monitor_selector.currentIndexChanged.connect(self.on_monitor_selected)

    def on_monitor_selected(self):
        self.selected_monitor = self.monitor_selector.current_monitor()
        self.start_swizz_button.show()
        self.update_live_preview()

    def start_swizz(self):
        self.timer.start(30)  # Start live preview updates

        # Clear layout and set new state
        self.clear_layout(self.layout)

        # Add the live monitor preview
        self.monitor_preview = QLabel(self)
        self.monitor_preview.setFixedSize(600, 400)
        self.layout.addWidget(self.monitor_preview)

        # Add the audio visualizer
        self.audio_visualiser = AudioVisualiser(self)
        self.audio_visualiser.setFixedSize(600, 100)
        self.audio_visualiser.mousePressEvent = self.audio_visualiser.toggle_mute
        self.layout.addWidget(self.audio_visualiser)

    def start_swizz_without_video(self):
        # Stop live preview updates
        self.timer.stop()

        # Clear layout and set new state
        self.clear_layout(self.layout)

        # Add the audio visualizer only
        self.audio_visualiser = AudioVisualiser(self)
        self.audio_visualiser.setFixedSize(600, 100)
        self.audio_visualiser.mousePressEvent = self.audio_visualiser.toggle_mute
        self.layout.addWidget(self.audio_visualiser)

    def update_live_preview(self):
        if self.selected_monitor:
            with mss.mss() as sct:
                sct_img = sct.grab(self.selected_monitor)
                img = np.array(sct_img)
                img = img[..., :3]  # Discard alpha channel if present
                img = np.ascontiguousarray(img)
                h, w, ch = img.shape
                bytes_per_line = ch * w
                qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                qt_pixmap = QPixmap.fromImage(qt_img).scaled(self.monitor_preview.size(), Qt.KeepAspectRatio)
                self.monitor_preview.setPixmap(qt_pixmap)

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
