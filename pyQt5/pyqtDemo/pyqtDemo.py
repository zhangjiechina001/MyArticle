# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\Administrator\PycharmProjects\MyPython\pyQt5\pyqtDemo\pyqtDemo'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow,QApplication
from qt01 import Ui_MainWindow

class LayoutDemo(QMainWindow,Ui_MainWindow):
    def __init__(self,parent=None):
        super(LayoutDemo,self).__init__(parent)
        self.setupUi(self)
        self.on_pushButton_click.connect(close)
    def on_pushButton_click(self,event):
        print('hello world!')
    def close(self):
        self.close()

if __name__=='__main__':
    import sys

    app=QApplication(sys.argv)
    ui=LayoutDemo()
    ui.show()
    sys.exit(app.exec_())