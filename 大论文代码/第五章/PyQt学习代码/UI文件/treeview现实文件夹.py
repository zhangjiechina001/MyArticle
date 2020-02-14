from PyQt5 import  QtCore
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model=QDirModel()
    tree=QTreeView()
    tree.setModel(model)
    tree.resize(600,400)
    tree.setWindowTitle('QTreeView')
    tree.show()
    # main.show()

    sys.exit(app.exec_())