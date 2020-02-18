from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *
from functools import partial
from PyQt5.QtCore import Qt
#设置不同的窗口风格 ['windowsvista', 'Windows', 'Fusion']
class WindowsStyleDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('用代码控制窗口最大化和最小化')

        self.resize(500,300)
        #保持窗口保持在前
        layout=QVBoxLayout()
        max_button1=QPushButton()
        max_button1.setText('窗口最大化1')
        # self.setWindowFlags(Qt.WindowMaximizeButtonHint)
        max_button2=QPushButton()
        max_button2.setText('窗口最大化2')
        max_button2.clicked.connect(self.showMaximized)
        max_button1.clicked.connect(self.setMaxmin)
        layout.addWidget(max_button2)
        layout.addWidget(max_button1)
        self.setLayout(layout)
        # self.setWindowFlags(QtCore.Qt.WindowMaximizeButtonHint|QtCore.Qt.WindowStaysOnTopHint|QtCore.Qt.FramelessWindowHint)

    def setMaxmin(self):
        #获得桌面
        desktop=QApplication.desktop()
        # 获得桌面可用尺寸
        rect=desktop.availableGeometry()
        self.setGeometry(rect)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = WindowsStyleDemo()
    main.show()
    sys.exit(app.exec_())