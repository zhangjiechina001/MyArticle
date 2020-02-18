from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *
from functools import partial
class AutoSignalSlot(QWidget):
    def __init__(self):
        super(AutoSignalSlot,self).__init__()
        self.okButton=QPushButton('ok')
        self.okButton.setObjectName('okButton')
        self.cancle=QPushButton('Cancel')
        self.cancle.setObjectName('cancelButton')

        layout=QHBoxLayout()
        layout.addWidget(self.okButton)
        layout.addWidget(self.cancle)
        self.setLayout(layout)
        # self.ok_button.clicked.connect(self.ok_button_click)
        #使用lambda表达式打印的button ok
        self.okButton.clicked.connect(lambda :print('这是lambda打印的ok'))
        #使用lambda调用外部
        self.okButton.clicked.connect(lambda :self.on_okButton_click_lambda(10,20))
        #使用partical模块进行调用
        self.okButton.clicked.connect(partial(self.on_okButton_click_partical,10,20))
        QtCore.QMetaObject.connectSlotsByName(self)

    #自动将相应的信号绑定到槽上
    @QtCore.pyqtSlot()
    def on_okButton_clicked(self):
        print('点击了ok按钮！')

    @QtCore.pyqtSlot()
    def on_cancleButton_clicked(self):
        print('点击了cancel按钮！')

    def on_okButton_click_lambda(self,m,n):
        print('这是使用lambda传递的多个参数m+n=',str(m+n))

    def on_okButton_click_partical(self,m,n):
        print('这是使用partical传递的多个参数m+n=',str(m+n))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = AutoSignalSlot()
    main.show()
    sys.exit(app.exec_())