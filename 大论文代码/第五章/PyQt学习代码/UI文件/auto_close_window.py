from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # main = ShowTime()
    label=QLabel('<font color=green size=140><f>窗口将在5秒候关闭！</f></font>')
    label.setWindowFlags(QtCore.Qt.SplashScreen|QtCore.Qt.FramelessWindowHint)
    label.show()
    QTimer.singleShot(5000,app.quit)
    # main.show()
    sys.exit(app.exec_())