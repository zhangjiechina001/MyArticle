from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *

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
        QtCore.QMetaObject.connectSlotsByName(self)

    #自动将相应的信号绑定到槽上
    @QtCore.pyqtSlot()
    def on_okButton_clicked(self):
        print('点击了ok按钮！')

    @QtCore.pyqtSlot()
    def on_cancleButton_clicked(self):
        print('点击了cancel按钮！')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = AutoSignalSlot()
    main.show()
    sys.exit(app.exec_())