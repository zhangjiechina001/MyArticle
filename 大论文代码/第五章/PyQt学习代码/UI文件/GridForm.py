from PyQt5 import QtCore
from PyQt5.QtCore import QTimer, QDateTime, QUrl
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys
from PyQt5.QtWebEngineWidgets import *

class GridForm(QWidget):
    def __init__(self):
        super(GridForm,self).__init__()
        self.setWindowTitle('栅格布局：表单设计')
        title_label=QLabel('标题')
        author_label=QLabel('作者')
        content_label=QLabel('内容')

        title_edit=QLineEdit()
        author_edit = QLineEdit()
        content_edit = QTextEdit()

        grid=QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(title_label,1,0)
        grid.addWidget(title_edit, 1, 1)

        grid.addWidget(author_label, 2, 0)
        grid.addWidget(author_edit, 2, 1)

        grid.addWidget(content_label, 3, 0)
        grid.addWidget(content_edit, 3, 1,5,1)
        self.setLayout(grid)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = GridForm()
    main.show()
    sys.exit(app.exec_())