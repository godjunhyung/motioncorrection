from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()

        self.setGeometry(200, 200, 500, 400)
        self.setWindowTitle("Styled GUI with PyQt")

        # 버튼 추가
        self.button = QtWidgets.QPushButton(self)
        self.button.setText("Click Me")
        self.button.setGeometry(150, 150, 200, 50)

        # QSS 스타일 적용
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())

window()
