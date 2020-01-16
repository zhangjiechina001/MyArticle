import sys
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow
from PyQt5 import QtCore, QtGui
from 大论文代码.第五章.PyQt学习代码.UI文件.demo_1 import Ui_MainWindow
class mywindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = mywindow()
    ui.show()
    sys.exit(app.exec_())