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
        if(len(items))>0:
            item=items[0]
            item.setBackground(QBrush(QColor(0,255,0)))
            row=item.row()
            #定位
            tableWidget.verticalScrollBar().setSliderPosition(row)



        # tableWidget.setHorizontalHeaderLabels(['姓名', '性别', '体重'])
        # textItem = QTableWidgetItem('小明')
        # tableWidget.setItem(0, 0, textItem)
        #
        # combox = QComboBox()
        # combox.addItem('男')
        # combox.addItem('女')
        # combox.addItem('bt')
        # # QSS
        # combox.setStyleSheet('QComboBox(margin:3px)')
        # tableWidget.setCellWidget(0, 1, combox)
        # modify_button = QPushButton('修改')
        # modify_button.setDown(True)
        #
        # modify_button.setStyleSheet('QPushButton(margin:3px')
        # tableWidget.setCellWidget(0, 2, modify_button)
        layout.addWidget(tableWidget)
        self.setLayout(layout)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QtableWidget()
    main.show()
    sys.exit(app.exec_())