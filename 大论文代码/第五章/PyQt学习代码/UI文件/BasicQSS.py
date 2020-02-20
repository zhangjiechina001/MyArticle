'''
QSS基础
QSS(Qt Style Sheets)
Qt样式表
用于设置控件的样式

'''
from PyQt5.QtWidgets import *
import sys

class Basic_QSS(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('QSS样式')
        btn1=QPushButton('按钮1')
        btn2=QPushButton('按钮2')
        vbox=QVBoxLayout()
        vbox.addWidget(btn1)
        vbox.addWidget(btn1)
        vbox.addWidget(btn2)
        self.setLayout(vbox)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Basic_QSS()
    #选择器
    qssStyle='''
    QPushButton {
    background-color:blue
    }
    '''
    main.setStyleSheet(qssStyle)
    main.show()
    sys.exit(app.exec_())