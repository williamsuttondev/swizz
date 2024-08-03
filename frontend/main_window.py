import mss
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt, QBuffer, QIODevice
from monitor_selector import MonitorSelector
from audio_visualiser import AudioVisualiser
from output_window import OutputWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.selected_monitor = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_preview)
        self.output_window = None
        self.audio_visualiser = None  # Initialize as None

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

        # Add extra spacing between the audio visualizer and the buttons
        self.spacing_label = QLabel(self)
        self.spacing_label.setFixedHeight(30)  # Adjust this value for desired spacing
        self.layout.addWidget(self.spacing_label)

        # Open the output window
        self.output_window = OutputWindow()
        self.output_window.show()

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

        # Add extra spacing between the audio visualizer and the buttons
        self.spacing_label = QLabel(self)
        self.spacing_label.setFixedHeight(30)  # Adjust this value for desired spacing
        self.layout.addWidget(self.spacing_label)

        # Open the output window
        self.output_window = OutputWindow()
        self.output_window.show()

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

                # Convert QImage to JPEG
                buffer = QBuffer()
                buffer.open(QIODevice.WriteOnly)
                qt_img.save(buffer, "JPEG")
                jpeg_data = buffer.data()
                # Call backend function with JPEG data
                #self.send_to_backend(jpeg_data)
                
                
    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
