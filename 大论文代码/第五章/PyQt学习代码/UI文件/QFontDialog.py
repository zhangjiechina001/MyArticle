import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class QFontDialogDemo(QWidget):
    def __init__(self):
        super(QFontDialogDemo, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('font dialog 例子')
        layout=QVBoxLayout()
        self.fontButton=QPushButton('选择字体')
        self.fontButton.clicked.connect(self.get_font)
        layout.addWidget(self.fontButton)
        self.font_label=QLabel('hello，测试字体例子')
        layout.addWidget(self.font_label)
        self.setLayout(layout)

    def get_font(self):
        font,result=QFontDialog.getFont()
        if result==True:
            self.font_label.setFont(font)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = QFontDialogDemo()
    form.show()
    sys.exit(app.exec_())