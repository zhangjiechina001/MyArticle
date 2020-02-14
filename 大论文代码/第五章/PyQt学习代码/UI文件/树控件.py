from PyQt5 import  QtCore
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import *
import sys


class QtableWidget(QWidget):
    def __init__(self):
        super(QtableWidget, self).__init__()
        self.initUI()
        # self.btnCancel.clicked.connect(self.close)

    def initUI(self):
        button_layout=QHBoxLayout()
        btn_add=QPushButton('添加节点')
        btn_update=QPushButton('修改节点')
        btn_delete=QPushButton('删除节点')
        button_layout.addWidget(btn_add)
        button_layout.addWidget(btn_update)
        button_layout.addWidget(btn_delete)
        self.setWindowTitle('在单元格中放置控件')
        self.resize(400, 300)
        layout = QVBoxLayout()
        self.treeWidget=QTreeWidget()
        self.treeWidget.setColumnCount(2)
        self.treeWidget.setHeaderLabels(['key,value'])
        root=QTreeWidgetItem(self.treeWidget)
        root.setText(0,'根节点')
        child1=QTreeWidgetItem(root)
        child1.setText(0,'子节点1')
        child2=QTreeWidgetItem(child1)
        child2.setText(0,'子节点2')
        # root.setIcon()
        layout.addLayout(button_layout)
        layout.addWidget(self.treeWidget)
        self.setLayout(layout)
        btn_add.clicked.connect(self.addNode)
        btn_update.clicked.connect(self.updateNode)
        btn_delete.clicked.connect(self.deleteNode)

    def on_tree_clicked(self,index):
        item=self.treeWidget.currentItem()
        print(index.row())
        print('key=%s,value=%s'%(item.text(0),item.text(1)))

    def addNode(self):
        item=self.treeWidget.currentItem()
        print(item)
        node=QTreeWidgetItem(item)
        node.setText(0,'新节点')
        node.setText(1,'新值')

    #修改节点
    def updateNode(self):
        item = self.treeWidget.currentItem()
        item.setText(0,'修改节点')
        item.setText(1,'值已经被修改')
        # pass
    #删除节点
    def deleteNode(self):
        item = self.treeWidget.currentItem()
        root=self.treeWidget.invisibleRootItem()
        for item in self.treeWidget.selectedItems():
            (item.parent() or root).removeChild(item)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QtableWidget()
    main.show()
    sys.exit(app.exec_())