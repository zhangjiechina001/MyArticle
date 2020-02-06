import sys,math
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
class MyComBox(QComboBox):
    def __init__(self):
        super(MyComBox,self).__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        print(e)
        if(e.mimeData().hasText()):
            e.accept()
        else:
            e.ignore()

    def dropEvent(self,e):
        self.addItem(e.mimeData().text())

class DrapDropDemo(QWidget):
    def __init__(self):
        super(DrapDropDemo,self).__init__()
        form_layout=QFormLayout()
        form_layout.addRow(QLabel('请将左边的文本拖曳到右边的下拉列表中'))
        line_edit=QLineEdit()
        line_edit.setDragEnabled(True)

        combo=MyComBox()
        form_layout.addRow(line_edit,combo)
        self.setLayout(form_layout)
        self.setWindowTitle('拖曳案例')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = DrapDropDemo()
    form.show()
    sys.exit(app.exec_())
