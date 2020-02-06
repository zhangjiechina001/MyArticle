import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class QColorDialogDemo(QWidget):
    def __init__(self):
        super(QColorDialogDemo, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('color dialog 例子')
        layout=QVBoxLayout()
        self.colorButton=QPushButton('选择颜色')
        self.colorButton.clicked.connect(self.get_color)
        layout.addWidget(self.colorButton)
        self.color_bg_Button=QPushButton('选择背景颜色')
        self.color_bg_Button.clicked.connect(self.get_bg_color)
        layout.addWidget(self.color_bg_Button)
        self.color_label=QLabel('hello，测试颜色例子')
        layout.addWidget(self.color_label)
        self.setLayout(layout)

    def get_color(self):
        color=QColorDialog.getColor()
        # if result==True:
        p=QPalette()
        p.setColor(QPalette.WindowText,color)
        self.color_label.setPalette(p)

    def get_bg_color(self):
        color=QColorDialog.getColor()
        # if result==True:
        p=QPalette()
        p.setColor(QPalette.Window,color)
        self.color_label.setAutoFillBackground(True)
        self.color_label.setPalette(p)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = QColorDialogDemo()
    form.show()
    sys.exit(app.exec_())