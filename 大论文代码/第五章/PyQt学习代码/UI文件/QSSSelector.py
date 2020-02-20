'''
使用QSS选择器设置控件样式

'''
'''
QSS基础
QSS(Qt Style Sheets)
Qt样式表
用于设置控件的样式

'''
from PyQt5.QtWidgets import *
import sys

class QSSSelectorDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('QSS样式')
        btn1=QPushButton('按钮1')
        btn2=QPushButton('按钮2')
        btn2.setProperty('name','btn2')
        btn3=QPushButton('按钮3')
        btn3.set
        vbox=QVBoxLayout()
        vbox.addWidget(btn1)
        vbox.addWidget(btn2)
        vbox.addWidget(btn3)
        self.setLayout(vbox)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QSSSelectorDemo()
    #选择器
    qssStyle='''
    QPushButton[name=btn2] {
    background-color:blue;
    color:red;
    height:120;
    font-size:120;
    }
    '''
    main.setStyleSheet(qssStyle)
    main.show()
    sys.exit(app.exec_())