from PyQt5.QtWidgets import QComboBox
import mss


def get_monitors():
    with mss.mss() as sct:
        return sct.monitors


class MonitorSelector(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setEditable(False)
        self.monitors = get_monitors()
        self.addItems([f"Monitor {i + 1}" for i in range(len(self.monitors))])

    def current_monitor(self):
        index = self.currentIndex()
        return self.monitors[index]
