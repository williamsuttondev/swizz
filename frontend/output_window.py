from PyQt5.QtWidgets import QMainWindow, QTextEdit
from PyQt5.QtCore import QTimer

class OutputWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_output)
        self.timer.start(2000)  # Update every 2 seconds

    def init_ui(self):
        self.setWindowTitle("Output")
        self.setGeometry(900, 100, 400, 300)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.setCentralWidget(self.text_edit)

    def update_output(self):
        # This should be replaced with the actual function or method that gets the output
        output_text = get_voice_prompt_output()
        self.text_edit.append(output_text)

def get_voice_prompt_output():
    # Placeholder function to simulate getting output from a voice prompt
    # Replace this with the actual implementation
    return "Simulated output from voice prompt..."