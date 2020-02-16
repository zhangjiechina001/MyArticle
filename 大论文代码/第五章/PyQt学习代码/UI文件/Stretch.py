from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *


# 伸缩量

class Hboxlayout_example(QWidget):

    def __init__(self):
        super(Hboxlayout_example, self).__init__()
        # self.setWindowTitle('水平盒布局')
        layout=QHBoxLayout()
        layout.addStretch(0)
        layout.addWidget(QPushButton('按钮1'))
        layout.addStretch(0)
        layout.addWidget(QPushButton('按钮2'))
        layout.addStretch(0)
        layout.addWidget(QPushButton('按钮3'))
        layout.addStretch(0)
        layout.addWidget(QPushButton('按钮4'))
        layout.addWidget(QPushButton('按钮5'))
        layout.addWidget(QPushButton('按钮6'))
        #设置控件距离
        layout.setSpacing(20)
        #设置控件对齐方式
        # layout.setAlignment()
        self.setLayout(layout)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Hboxlayout_example()
    main.show()
    sys.exit(app.exec_())