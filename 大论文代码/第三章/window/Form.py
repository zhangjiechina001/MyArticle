import random
import sys
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime, QPointF
from PyQt5.QtGui import QIcon, QImage, QPixmap, QColor
from PyQt5.QtWidgets import QDialog, QApplication, QWidget,QMessageBox,QMenu,QLabel,QGraphicsItem
# from qt使用chart import Ui_Form
from PyQt5.QtChart import (QAreaSeries, QBarSet, QChart, QChartView,
                           QLineSeries, QPieSeries, QScatterSeries, QSplineSeries,
                           QStackedBarSeries,QValueAxis)
from formUI import Ui_Form

class MainWindow(QWidget):
    def __init__(self,parent=None):
        super(MainWindow,self).__init__(parent)
        self.ui=Ui_Form()
        self.ui.setupUi(self)
        self.chartFirstUnflood = self.createLineChart()
        self.chartSecondUnflood = self.createLineChart()
        # self.ui.gridLayout.addWidget(QChartView(self.chartFirstUnflood),1,0,1,1)
        # self.ui.gridLayout.addWidget(QChartView(self.chartSecondUnflood),1,1,1,1)


    def createLineChart(self):
        chart = QChart()
        chart.setTitle("像素分布直方图")
        seriesArray=[]



        axisX = QValueAxis()
        axisY = QValueAxis()
        axisX.setRange(0, 255)
        axisY.setRange(0, 10)

        chart.setAxisX(axisX)
        chart.setAxisY(axisY)



        return chart

if __name__=='__main__':
    app = QApplication(sys.argv)
    form = MainWindow()
    size=form.size()
    form.setFixedSize(size)
    form.show()
    sys.exit(app.exec_())