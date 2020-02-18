from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *
from functools import partial
#设置不同的窗口风格 ['windowsvista', 'Windows', 'Fusion']
class WindowsStyleDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('设置窗口的样式')

        self.resize(500,300)
        #保持窗口保持在前
        self.setWindowFlags(
            QtCore.Qt.WindowMaximizeButtonHint | QtCore.Qt.WindowStaysOnTopHint)
        # self.setWindowFlags(QtCore.Qt.WindowMaximizeButtonHint|QtCore.Qt.WindowStaysOnTopHint|QtCore.Qt.FramelessWindowHint)
        self.setObjectName('mian_window')
        self.setStyleSheet("#main_window{border-image:url(img_python.jpg);}")

    def handleStyleChanged(self,style):
        QApplication.setStyle(style)
        # self.styleComBox.setCurrentIndex(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = WindowsStyleDemo()
    main.show()
    sys.exit(app.exec_())