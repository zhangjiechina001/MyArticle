from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *
#拖动控件之间的边界
class Splitter(QWidget):
    def __init__(self):
        super(Splitter,self).__init__()
        self.setWindowTitle('QSplitter的例子')
        self.setGeometry(300,300,300,200)
        topleft=QFrame()
        topleft.setFrameShape(QFrame.StyledPanel)

        bottom = QFrame()
        bottom.setFrameShape(QFrame.StyledPanel)

        splitter1=QSplitter(QtCore.Qt.Horizontal)
        textEdit=QTextEdit()
        splitter1.addWidget(topleft)
        splitter1.addWidget(textEdit)

        splitter2=QSplitter(QtCore.Qt.Vertical)
        # textEdit=QTextEdit()
        splitter2.addWidget(splitter1)
        splitter2.addWidget(bottom)

        hbox=QHBoxLayout()
        hbox.addWidget(splitter2)
        self.setLayout(hbox)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Splitter()
    main.show()
    sys.exit(app.exec_())


