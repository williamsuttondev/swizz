import sys
import mss
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QLabel, QApplication
from PyQt5.QtCore import QTimer, Qt
from monitor_selector import MonitorSelector

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("swizz")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Monitor Selector
        self.monitor_selector = MonitorSelector(self)
        self.layout.addWidget(self.monitor_selector)

        # Monitor Preview Placeholder
        self.monitor_preview = QLabel("Please select a monitor preview", self)
        self.monitor_preview.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.monitor_preview)

        self.monitor_selector.currentIndexChanged.connect(self.start_live_preview)
        self.selected_monitor = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_preview)

    def start_live_preview(self):
        # Set the selected monitor
        self.selected_monitor = self.monitor_selector.current_monitor()

        # Start the timer to update the live preview
        if not self.timer.isActive():
            self.timer.start(30)  # Refresh rate set to approximately 30 FPS

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())