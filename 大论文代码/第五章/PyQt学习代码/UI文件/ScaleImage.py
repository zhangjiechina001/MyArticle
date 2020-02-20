'''
缩放图片
QImage.scaled
'''

from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl, QPoint
from PyQt5.QtGui import QColor, QBrush, QPixmap, QPainter, QImage
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *
from functools import partial
from PyQt5.QtCore import Qt

class ScaleImage(QWidget):
    def __init__(self):
        super(ScaleImage,self).__init__()
        self.setWindowTitle('图片大小缩放的例子')
        fileName='./img_python.jpg'
        self.img=QImage(fileName)
        self.label1=QLabel()
        self.label1.setMinimumWidth(100)
        self.label1.setMinimumHeight(100)
        result=self.img.scaled(self.label1.width(),self.label1.height(),Qt.IgnoreAspectRatio,Qt.SmoothTransformation)
        self.label1.setPixmap(QPixmap.fromImage(result))
        hbox=QVBoxLayout()
        hbox.addWidget(self.label1)
        self.setLayout(hbox)
        self.label1.setScaledContents(True)
        self.label1.resize(self.size())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = ScaleImage()
    main.show()
    sys.exit(app.exec_())