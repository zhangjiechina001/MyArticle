from PyQt5 import QtCore
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
class MultiWindows(QMainWindow):
    count=0
    def __init__(self):
        super(MultiWindows,self).__init__()
        self.setWindowTitle('容纳多文档的窗口')
        self.mdi=QMdiArea()
        self.setCentralWidget(self.mdi)
        bar=self.menuBar()
        #容纳多文档的窗口
        file=bar.addMenu('file')
        file.addAction('New')
        file.addAction('cascade')
        file.addAction('tiled')
        file.triggered.connect(self.window_action)

    def window_action(self,event):
        print(event.text())
        if (event.text()=='New'):
            MultiWindows.count=MultiWindows.count+1
            sub=QMdiSubWindow()
            sub.setWidget(QTextEdit())
            sub.setWindowTitle('子窗口'+str(MultiWindows.count))
            #添加子窗口
            self.mdi.addSubWindow(sub)
            sub.show()
        #窗口重叠
        elif event.text()=='cascade':
            self.mdi.cascadeSubWindows()
        #平铺
        elif event.text()=='tiled':
            self.mdi.tileSubWindows()

if __name__=='__main__':
    app = QApplication(sys.argv)
    main = MultiWindows()
    main.show()
    sys.exit(app.exec_())