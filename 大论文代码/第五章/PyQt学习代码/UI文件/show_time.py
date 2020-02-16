from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys

class ShowTime(QWidget):
    def __init__(self):
        super(ShowTime,self).__init__()
        self.setWindowTitle('动态显示当前时间')

        self.label=QLabel('显示当前时间')
        self.start_button=QPushButton('开始')
        self.end_button=QPushButton('结束')
        layout=QGridLayout()

        self.timer=QTimer()
        self.timer.timeout.connect(self.show_time)
        layout.addWidget(self.label,0,0,1,2)
        layout.addWidget(self.start_button,1,0)
        layout.addWidget(self.end_button,1,1)
        self.setLayout(layout)
        self.timer.setInterval(10000)

        self.start_button.clicked.connect(self.startTimer)
        self.end_button.clicked.connect(self.endTimer)

    def show_time(self):
        time=QDateTime.currentDateTime()
        time_display=time.toString('yyyy-MM-dd hh:mm:ss dddd')
        self.label.setText(time_display)

    def startTimer(self):
        self.timer.start(2000)
        self.start_button.setEnabled(False)
        self.end_button.setEnabled(True)
        # self.close()
    def endTimer(self):
        self.timer.stop()
        self.start_button.setEnabled(True)
        self.end_button.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = ShowTime()
    main.show()
    sys.exit(app.exec_())