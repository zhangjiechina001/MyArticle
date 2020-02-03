# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'zhifangtu.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1172, 746)
        self.gridLayout = QtWidgets.QGridLayout(Form)
        self.gridLayout.setObjectName("gridLayout")
        self.widget_10 = QtWidgets.QWidget(Form)
        self.widget_10.setObjectName("widget_10")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget_10)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.v_Layout_dis = QtWidgets.QVBoxLayout()
        self.v_Layout_dis.setObjectName("v_Layout_dis")
        self.frame_2 = QtWidgets.QFrame(self.widget_10)
        self.frame_2.setMinimumSize(QtCore.QSize(351, 0))
        self.frame_2.setMaximumSize(QtCore.QSize(351, 16777215))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(10, 20, 54, 12))
        self.label.setObjectName("label")
        self.listView = QtWidgets.QListView(self.frame_2)
        self.listView.setGeometry(QtCore.QRect(10, 40, 331, 261))
        self.listView.setObjectName("listView")
        self.lst_info = QtWidgets.QListView(self.frame_2)
        self.lst_info.setGeometry(QtCore.QRect(10, 350, 331, 341))
        self.lst_info.setObjectName("lst_info")
        self.layoutWidget = QtWidgets.QWidget(self.frame_2)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 310, 331, 31))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_openOPCUA = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_openOPCUA.setObjectName("btn_openOPCUA")
        self.horizontalLayout.addWidget(self.btn_openOPCUA)
        self.btn_openfile = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_openfile.setObjectName("btn_openfile")
        self.horizontalLayout.addWidget(self.btn_openfile)
        self.btn_openSysfile = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_openSysfile.setObjectName("btn_openSysfile")
        self.horizontalLayout.addWidget(self.btn_openSysfile)
        self.v_Layout_dis.addWidget(self.frame_2)
        self.gridLayout_2.addLayout(self.v_Layout_dis, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.widget_10, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "连接状态"))
        self.btn_openOPCUA.setText(_translate("Form", "连接OPC"))
        self.btn_openfile.setText(_translate("Form", "打开文件"))
        self.btn_openSysfile.setText(_translate("Form", "打开系统文件"))
