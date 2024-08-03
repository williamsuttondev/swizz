import mss
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PyQt5.QtCore import Qt
from monitor_selector import MonitorSelector
from audio_visualiser import AudioVisualiser

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Select Monitor")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Monitor Selector
        self.monitor_selector = MonitorSelector(self)
        self.layout.addWidget(self.monitor_selector)

        # Audio Visualiser
        self.audio_visualiser = AudioVisualiser(self)
        self.layout.addWidget(self.audio_visualiser)

        # Mute Button
        self.mute_button = QPushButton("Mute", self)
        self.mute_button.clicked.connect(self.audio_visualiser.toggle_mute)
        self.layout.addWidget(self.mute_button)

        # Monitor Preview Placeholder
        self.monitor_preview = QLabel("Monitor Preview", self)
        self.monitor_preview.setFixedSize(400, 225)
        self.monitor_preview.setStyleSheet("background-colour: black; colour: white;")
        self.layout.addWidget(self.monitor_preview)

        self.monitor_selector.currentIndexChanged.connect(self.update_monitor_preview)

    def update_monitor_preview(self):
        monitor = self.monitor_selector.current_monitor()
        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = img[..., :3]
            img = np.ascontiguousarray(img)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            qt_pixmap = QPixmap.fromImage(qt_img).scaled(self.monitor_preview.size(), Qt.KeepAspectRatio)
            self.monitor_preview.setPixmap(qt_pixmap)