import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QSizePolicy, QFileDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import cv2
from matplotlib.figure import Figure

from zhifangtu import Ui_Form

#首先定义一个继承自FigureCanvas的类
class Mydemo(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=256):
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        #创建一个2*2布局的表格
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes_1 = self.fig.add_subplot(3, 3, 1)
        self.axes_2 = self.fig.add_subplot(3, 3, 2)
        self.axes_3 = self.fig.add_subplot(3, 3, 3)
        self.axes_4 = self.fig.add_subplot(3, 3, 4)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.fig.tight_layout()

import cv2
class MainWindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)
        self.vlayout=QtWidgets.QVBoxLayout(self)
        #创建图表的实例
        self.cavas=Mydemo(width=5, height=4, dpi=100)
        self.widget_toolbar=NavigationToolbar(self.cavas, self.widget)
        self.v_Layout_dis.addWidget(self.widget_toolbar)
        self.v_Layout_dis.addWidget(self.cavas)
        self.btn_openfile_ywzf.clicked.connect(self.btn_open_file_ywzf_clicked)
        self.statusTip()

    def open_image_file(self):
        '''打开一个图像文件'''
        fileName, filetype = QFileDialog.getOpenFileName(self,
                                                         "open file", '.',
                                                         "jpg Files (*.jpg);;png Files (*.png);;All Files (*)")
        return fileName


    def btn_open_file_ywzf_clicked(self):
        import numpy as np
        file_name=self.open_image_file()
        image=cv2.imread(filename=file_name,flags=cv2.IMREAD_GRAYSCALE)
        #图片显示
        self.cavas.axes_1.imshow(image,cmap='gray')
        self.cavas.axes_1.set_title('原图')
        self.cavas.axes_2.hist(image.ravel(),256,[0,255],color='r')
        self.cavas.axes_2.set_title('直方图')
        self.cavas.axes_3.hist(image.ravel(),256,[0,255],color='g')
        #在界面上显示
        self.cavas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())