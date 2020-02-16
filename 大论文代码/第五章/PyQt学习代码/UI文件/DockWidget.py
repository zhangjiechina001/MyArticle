from PyQt5 import QtCore
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
class DockDemo(QMainWindow):
    def __init__(self):
        super(DockDemo,self).__init__()
        self.setWindowTitle('停靠控件（QDockWidget)')
        layout=QHBoxLayout()
        #先创建一个承载items的Dock对象
        self.items=QDockWidget('Dockable',self)
        self.listWidget=QListWidget()
        for i in range(10):
            self.listWidget.addItem('item%d'%i)
        #讲listWidget添加到Dock中去
        self.items.setWidget(self.listWidget)
        #设置控件悬浮
        # self.items.setFloating(True)
        self.setCentralWidget(QLineEdit('武汉加油'))
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea,self.items)


if __name__=='__main__':
    app = QApplication(sys.argv)
    main = DockDemo()
    main.show()
    sys.exit(app.exec_())