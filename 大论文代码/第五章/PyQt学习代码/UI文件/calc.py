from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *

class Calc(QWidget):
    def __init__(self):
        super(Calc,self).__init__()
        self.setWindowTitle('栅格布局')

        grid=QGridLayout()
        self.setLayout(grid)

        names=['cls','back','close',' ','7','8','9','/',
               '4','5','6','*','1','2','3','-',
               '0','.','=','+']

        positions=[(i,j) for i in range(5) for j in range(4)]
        print(positions)

        for position,name in zip(positions,names):
            if name=='':
                continue
            print(position,name)
            button=QPushButton(name)
            grid.addWidget(button,position[0],position[1])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Calc()
    main.show()
    sys.exit(app.exec_())