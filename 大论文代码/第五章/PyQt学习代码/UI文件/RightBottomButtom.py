from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *

class Right_bottom_button(QWidget):
    def __init__(self):
        super(Right_bottom_button,self).__init__()
        self.setWindowTitle('让按钮永远在右下角')
        self.resize(400,300)

        ok_button=QPushButton('确定')
        cancle_button = QPushButton('取消')

        hbox=QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(ok_button)
        hbox.addWidget(cancle_button)

        vbox=QVBoxLayout()
        btn1=QPushButton('按钮1')
        btn2 = QPushButton('按钮2')
        btn3 = QPushButton('按钮3')
        #为0的先排列
        vbox.addStretch(0)
        vbox.addWidget(btn1)
        vbox.addWidget(btn2)
        vbox.addWidget(btn3)
        #为1的后考虑
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setLayout(vbox)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Right_bottom_button()
    main.show()
    sys.exit(app.exec_())