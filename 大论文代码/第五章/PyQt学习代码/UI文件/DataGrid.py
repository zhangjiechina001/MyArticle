'''
使用可视化的方式对SQLLite数据库进行增删改查等操作
QTableView
QSqlTableModel
'''
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtSql import *

def initModel(model):
    model.setTable('people2')
    model.setEditStrategy(QSqlTableModel.OnFieldChange)
    model.select()
    model.setHeaderData(0,Qt.Horizontal,'ID')
    model.setHeaderData(1, Qt.Horizontal, 'name')
    model.setHeaderData(2, Qt.Horizontal, 'adress')
    return model

def createView(title,model):
    view=QTableView()
    view.setModel(model)
    view.setWindowTitle(title)
    return view

def findrow(i):
    delrow=i.row()
    print('del row=%s'%str(delrow))

def addRow():
    ret=model.insertRows(model.rowCount(),1)
    print('insert Row=%s'%str(ret))



if __name__=='__main__':
    app=QApplication(sys.argv)
    db=QSqlDatabase.addDatabase('QSQLITE')
    db.setDatabaseName('./db/database.db')
    model=QSqlTableModel()
    delrow=-1
    model=initModel(model)
    view=createView('展示数据',model)
    view.clicked.connect(findrow)
    dlg=QDialog()
    layout=QVBoxLayout()
    layout.addWidget(view)
    addBtn=QPushButton('添加一行')
    addBtn.clicked.connect(addRow)
    delBtn=QPushButton('删除一行')
    delBtn.clicked.connect(lambda :model.removeRow(view.currentIndex().row()))
    layout.addWidget(addBtn)
    layout.addWidget(delBtn)
    dlg.setWindowTitle('DataBase Demo')

    dlg.setLayout(layout)
    dlg.resize(500,400)
    dlg.show()

    sys.exit(app.exec_())

