'''
操作SQLite数据库
'''
import sys
from PyQt5.QtSql import QSqlDatabase,QSqlQuery

def createDB():
    db=QSqlDatabase.addDatabase('QSQLITE')
    #指定SQLITe数据库的文件名
    db.setDatabaseName('./db/database.db')
    if not db.open():
        print('无法建立数据库的连接')
        return False
    query=QSqlQuery()
    query.exec('create table people2(id int primary key,name varchar(10),address varchar(50))')
    # query.exec('insert into people values(2,"张杰","沈阳")')
    db.close()

createDB()