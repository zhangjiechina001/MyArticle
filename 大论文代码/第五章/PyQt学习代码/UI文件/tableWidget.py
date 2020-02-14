from PyQt5.QtWidgets import *
import sys

class QtableWidget(QWidget):
    def __init__(self):
        super(QtableWidget,self).__init__()
        self.initUI()
        # self.btnCancel.clicked.connect(self.close)

    def initUI(self):
        self.setWindowTitle('在单元格中放置控件')
        self.resize(400,300)
        layout=QHBoxLayout()
        tableWidget=QTableWidget()
        tableWidget.setRowCount(4)
        tableWidget.setColumnCount(3)

        tableWidget.setHorizontalHeaderLabels(['姓名','性别','体重'])
        textItem=QTableWidgetItem('小明')
        tableWidget.setItem(0,0,textItem)

        combox=QComboBox()
        combox.addItem('男')
        combox.addItem('女')
        combox.addItem('bt')
        #QSS
        combox.setStyleSheet('QComboBox(margin:3px)')
        tableWidget.setCellWidget(0,1,combox)
        modify_button=QPushButton('修改')
        modify_button.setDown(True)

        modify_button.setStyleSheet('QPushButton(margin:3px')
        tableWidget.setCellWidget(0, 2, modify_button)
        layout.addWidget(tableWidget)
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QtableWidget()
    main.show()
    sys.exit(app.exec_())