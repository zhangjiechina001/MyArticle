import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import cv2
from zhifangtu import Ui_Form

class MainWindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)
        # self.widget_toolbar = NavigationToolbar(self.frame_4)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())