'''
实现绘图应用
1.如何绘图
在方法中通过调用painEvent
2.在那里绘图
3.如何通过移动鼠标绘图
鼠标拥有三个事件
（1）按下
（2）移动
（3）抬起
'''
from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl, QPoint
from PyQt5.QtGui import QColor, QBrush, QPixmap, QPainter
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *
from functools import partial
from PyQt5.QtCore import Qt

class Drawing(QWidget):
    def __init__(self):
        super(Drawing,self).__init__()
        self.setWindowTitle('绘图应用')
        self.pix=QPixmap()
        self.last_point=QPoint()
        self.end_point = QPoint()
        self.initUI()

    def initUI(self):
        self.resize(600,600)
        self.pix=QPixmap(600,600)
        self.pix.fill(Qt.white)
#按下时记录鼠标位置，移动时进行绘图
    def paintEvent(self, QPaintEvent):
        #往pix上画线
        pp=QPainter(self.pix)
        pp.drawLine(self.last_point,self.end_point)
        #让前一个坐标值等于后一个坐标值
        #这样就能实现画出连续的线
        self.last_point=self.end_point
        painter=QPainter(self)
        painter.drawPixmap(0,0,self.pix)

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button()==Qt.LeftButton:
            self.last_point=QMouseEvent.pos()

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton:
            self.end_point=QMouseEvent.pos()
            self.update()

    def mouseReleaseEvent(self, QMouseEvent):
        #鼠标左键释放
        if QMouseEvent.button()==Qt.LeftButton:
            self.end_point=QMouseEvent.pos()
            #进行重新绘制
            self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Drawing()
    main.show()
    sys.exit(app.exec_())