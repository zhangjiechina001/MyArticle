from PyQt5 import  QtCore
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys


class QtableWidget(QWidget):
    def __init__(self):
        super(QtableWidget, self).__init__()
        self.initUI()
        # self.btnCancel.clicked.connect(self.close)

    def initUI(self):
        self.setWindowTitle('在单元格中放置控件')
        self.resize(400, 300)
        layout = QHBoxLayout()
        tableWidget = QTableWidget()
        tableWidget.setRowCount(40)
        tableWidget.setColumnCount(3)
        for i in range(40):
            for j in range(3):
                item='(%d,%d)'%(i+1,j+1)
                tableWidget.setItem(i,j,QTableWidgetItem(item))
        text='(13,1)'
        items=tableWidget.findItems(text,QtCore.Qt.MatchExactly)
        tableWidget.setSpan(0,0,3,3)
        if(len(items))>0:
            item=items[0]
            item.setBackground(QBrush(QColor(0,255,0)))
            row=item.row()
            #定位
            tableWidget.verticalScrollBar().setSliderPosition(row)

        layout.addWidget(tableWidget)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QtableWidget()
    main.show()
    sys.exit(app.exec_())