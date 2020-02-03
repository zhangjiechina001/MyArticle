from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import *
import sys

class QLabelBuddy(QDialog):
    def __init__(self):
        super(QLabelBuddy,self).__init__()
        self.initUI()
        self.btnCancel.clicked.connect(self.close)

    def initUI(self):
        self.setWindowTitle('Qlabel与伙伴控件')
        nameLabel=QLabel('&Name',self)
        nameLineEdit=QLineEdit(self)
        #设置伙伴控件
        nameLabel.setBuddy(nameLineEdit)

        passwordLabel=QLabel('&Passworld',self)
        passwordLineEdit=QLineEdit(self)
        #设置伙伴控件
        passwordLabel.setBuddy(passwordLineEdit)
        passwordLineEdit.setPlaceholderText('password')
        passwordLineEdit.setEchoMode(QLineEdit.PasswordEchoOnEdit)

        btnOK=QPushButton('&OK')
        self.btnCancel=QPushButton('&Cancel')
        mainLayout=QGridLayout()
        mainLayout.addWidget(nameLabel,0,0)
        mainLayout.addWidget(nameLineEdit,0,1,1,2)
        mainLayout.addWidget(passwordLabel, 1, 0)
        mainLayout.addWidget(passwordLineEdit, 1, 1, 1, 2)
        mainLayout.addWidget(btnOK,2,1)
        mainLayout.addWidget(self.btnCancel, 2, 2)

        intValidator=QIntValidator(self)
        intValidator.setRange(0,99999)
        passwordLineEdit.setValidator(intValidator)
        self.setLayout(mainLayout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = QLabelBuddy()
    main.show()
    sys.exit(app.exec_())