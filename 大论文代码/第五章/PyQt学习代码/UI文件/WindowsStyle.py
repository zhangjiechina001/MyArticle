from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *
from functools import partial
#设置不同的窗口风格 ['windowsvista', 'Windows', 'Fusion']
class WindowsStyleDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('设置窗口风格')
        horizontallayout=QHBoxLayout()
        self.styleLabel=QLabel('设置窗口风格')
        self.styleComBox=QComboBox()
        self.styleComBox.addItems(QStyleFactory.keys())
        print(QApplication.style().objectName())
        index=self.styleComBox.findText(QApplication.style().objectName(),QtCore.Qt.MatchFixedString)
        self.styleComBox.setCurrentIndex(index)
        self.styleComBox.activated[str].connect(self.handleStyleChanged)
        horizontallayout.addWidget(self.styleLabel)
        horizontallayout.addWidget(self.styleComBox)
        self.setLayout(horizontallayout)

    def handleStyleChanged(self,style):
        QApplication.setStyle(style)
        # self.styleComBox.setCurrentIndex(1)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = WindowsStyleDemo()
    main.show()
    print(QStyleFactory.keys())
    sys.exit(app.exec_())