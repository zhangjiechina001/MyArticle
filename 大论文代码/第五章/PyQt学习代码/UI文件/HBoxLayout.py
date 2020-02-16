from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *


# 同时使用Python和Web开发程序，混合开发

class Hboxlayout_example(QWidget):

    def __init__(self):
        super(Hboxlayout_example, self).__init__()
        # self.setWindowTitle('水平盒布局')
        layout=QHBoxLayout()
        layout.addWidget(QPushButton('按钮1'))
        layout.addWidget(QPushButton('按钮1'),1,QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        layout.addWidget(QPushButton('按钮1'),1,QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        layout.addWidget(QPushButton('按钮1'),1,QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        layout.addWidget(QPushButton('按钮1'),1,QtCore.Qt.AlignLeft|QtCore.Qt.AlignBottom)
        layout.addWidget(QPushButton('按钮1'),1,QtCore.Qt.AlignRight|QtCore.Qt.AlignTop)
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