# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DataDisplayUI.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QGridLayout, QApplication
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.LineDisplayGB = QtWidgets.QGroupBox(self.centralwidget)
        self.LineDisplayGB.setObjectName("LineDisplayGB")
        self.gridLayout.addWidget(self.LineDisplayGB, 0, 0, 1, 1)
        self.BarDisplayGB = QtWidgets.QGroupBox(self.centralwidget)
        self.BarDisplayGB.setObjectName("BarDisplayGB")
        self.gridLayout.addWidget(self.BarDisplayGB, 0, 1, 1, 1)
        self.ImageDisplayGB = QtWidgets.QGroupBox(self.centralwidget)
        self.ImageDisplayGB.setObjectName("ImageDisplayGB")
        self.gridLayout.addWidget(self.ImageDisplayGB, 1, 0, 1, 1)
        self.SurfaceDisplayGB = QtWidgets.QGroupBox(self.centralwidget)
        self.SurfaceDisplayGB.setObjectName("SurfaceDisplayGB")
        self.gridLayout.addWidget(self.SurfaceDisplayGB, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.LineDisplayGB.setTitle(_translate("MainWindow", "Line Display"))
        self.BarDisplayGB.setTitle(_translate("MainWindow", "Bar Display"))
        self.ImageDisplayGB.setTitle(_translate("MainWindow", "Image Display"))
        self.SurfaceDisplayGB.setTitle(_translate("MainWindow", "3D Surface Display"))

class Figure_Canvas(FigureCanvas):
    def __init__(self,parent=None,width=3.9,height=2.7,dpi=100):
        self.fig=Figure(figsize=(width,height),dpi=100)
        super(Figure_Canvas,self).__init__(self.fig)
        self.ax=self.fig.add_subplot(111)
    def test(self):
        x=[1,2,3,4,5,6,7]
        y=[2,1,3,5,6,4,3]
        self.ax.plot(x,y)

import numpy as np

class ImgDisp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(ImgDisp, self).__init__(parent)
        self.setupUi(self)
        self.Init_Widgets()

    def Init_Widgets(self):
        self.PrepareSamples()
        self.PrepareLineCanvas()
        self.PrepareBarCanvas()
        self.PrepareImgCanvas()
        self.PrepareSurfaceCanvas()

    def PrepareSamples(self):
        self.x = np.arange(-4, 4, 0.02)
        self.y = np.arange(-4, 4, 0.02)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.z = np.sin(self.x)
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2)
        self.Z = np.sin(self.R)

    def PrepareLineCanvas(self):
        self.LineFigure = Figure_Canvas()
        self.LineFigureLayout = QGridLayout(self.LineDisplayGB)
        self.LineFigureLayout.addWidget(self.LineFigure)
        self.LineFigure.ax.set_xlim(-4, 4)
        self.LineFigure.ax.set_ylim(-1, 1)
        self.line = Line2D(self.x, self.z)
        self.LineFigure.ax.add_line(self.line)

    def PrepareBarCanvas(self):
        self.BarFigure = Figure_Canvas()
        self.BarFigureLayout = QGridLayout(self.BarDisplayGB)
        self.BarFigureLayout.addWidget(self.BarFigure)
        self.BarFigure.ax.set_xlim(-4, 4)
        self.BarFigure.ax.set_ylim(-1, 1)
        self.bar = self.BarFigure.ax.bar(np.arange(-4, 4, 0.5), np.sin(np.arange(-4, 4, 0.5)), width=0.4)
        self.patches = self.bar.patches

    def PrepareImgCanvas(self):
        self.ImgFigure = Figure_Canvas()
        self.ImgFigureLayout = QGridLayout(self.ImageDisplayGB)
        self.ImgFigureLayout.addWidget(self.ImgFigure)
        self.ImgFig = self.ImgFigure.ax.imshow(self.Z, cmap='bone')
        self.ImgFig.set_clim(-0.8, 0.8)

    def PrepareSurfaceCanvas(self):
        self.SurfFigure = Figure_Canvas()
        self.SurfFigureLayout = QGridLayout(self.SurfaceDisplayGB)
        self.SurfFigureLayout.addWidget(self.SurfFigure)
        self.SurfFigure.ax.remove()
        self.ax3d = self.SurfFigure.fig.gca(projection='3d')
        self.Surf = self.ax3d.plot_surface(self.X, self.Y, self.Z, cmap='rainbow')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = ImgDisp()
    form.show()
    sys.exit(app.exec_())