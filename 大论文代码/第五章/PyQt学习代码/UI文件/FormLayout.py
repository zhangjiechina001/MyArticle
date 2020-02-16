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

        grid=QFormLayout()
        grid.setSpacing(10)
        grid.addRow(title_label,title_edit)
        grid.addRow(author_label,author_edit)
        grid.addRow(content_label,content_edit)
        self.setLayout(grid)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = GridForm()
    main.show()
    sys.exit(app.exec_())