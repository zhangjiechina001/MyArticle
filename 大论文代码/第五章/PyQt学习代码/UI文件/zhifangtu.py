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
        self.frame_2 = QtWidgets.QFrame(self.widget_10)
        self.frame_2.setMinimumSize(QtCore.QSize(351, 0))
        self.frame_2.setMaximumSize(QtCore.QSize(351, 16777215))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.toolBox = QtWidgets.QToolBox(self.frame_2)
        self.toolBox.setObjectName("toolBox")
        self.page = QtWidgets.QWidget()
        self.page.setGeometry(QtCore.QRect(0, 0, 329, 610))
        self.page.setObjectName("page")
        self.frame_3 = QtWidgets.QFrame(self.page)
        self.frame_3.setGeometry(QtCore.QRect(0, 150, 321, 451))
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.widget_5 = QtWidgets.QWidget(self.frame_3)
        self.widget_5.setGeometry(QtCore.QRect(10, 10, 301, 421))
        self.widget_5.setObjectName("widget_5")
        self.widget = QtWidgets.QWidget(self.widget_5)
        self.widget.setGeometry(QtCore.QRect(0, 0, 301, 421))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.lst_info = QtWidgets.QListWidget(self.widget)
        self.lst_info.setResizeMode(QtWidgets.QListView.Adjust)
        self.lst_info.setObjectName("lst_info")
        self.verticalLayout.addWidget(self.lst_info)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        self.tab_recongnizeInfo = QtWidgets.QTableWidget(self.widget)
        self.tab_recongnizeInfo.setObjectName("tab_recongnizeInfo")
        self.tab_recongnizeInfo.setColumnCount(0)
        self.tab_recongnizeInfo.setRowCount(0)
        self.verticalLayout.addWidget(self.tab_recongnizeInfo)
        self.widget1 = QtWidgets.QWidget(self.page)
        self.widget1.setGeometry(QtCore.QRect(0, 50, 321, 41))
        self.widget1.setObjectName("widget1")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_7 = QtWidgets.QLabel(self.widget1)
        self.label_7.setObjectName("label_7")
        self.verticalLayout_4.addWidget(self.label_7)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.rbn_module = QtWidgets.QRadioButton(self.widget1)
        self.rbn_module.setObjectName("rbn_module")
        self.horizontalLayout_3.addWidget(self.rbn_module)
        self.rbn_svm = QtWidgets.QRadioButton(self.widget1)
        self.rbn_svm.setObjectName("rbn_svm")
        self.horizontalLayout_3.addWidget(self.rbn_svm)
        self.rbn_cnn = QtWidgets.QRadioButton(self.widget1)
        self.rbn_cnn.setObjectName("rbn_cnn")
        self.horizontalLayout_3.addWidget(self.rbn_cnn)
        self.verticalLayout_4.addLayout(self.horizontalLayout_3)
        self.widget2 = QtWidgets.QWidget(self.page)
        self.widget2.setGeometry(QtCore.QRect(1, 1, 320, 43))
        self.widget2.setObjectName("widget2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget2)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.btn_connectOPC = QtWidgets.QPushButton(self.widget2)
        self.btn_connectOPC.setObjectName("btn_connectOPC")
        self.gridLayout_4.addWidget(self.btn_connectOPC, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.widget2)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)
        self.btn_disconnectOPC = QtWidgets.QPushButton(self.widget2)
        self.btn_disconnectOPC.setObjectName("btn_disconnectOPC")
        self.gridLayout_4.addWidget(self.btn_disconnectOPC, 1, 1, 1, 1)
        self.btn_reconginze = QtWidgets.QPushButton(self.widget2)
        self.btn_reconginze.setObjectName("btn_reconginze")
        self.gridLayout_4.addWidget(self.btn_reconginze, 1, 3, 1, 1)
        self.btn_openFile = QtWidgets.QPushButton(self.widget2)
        self.btn_openFile.setObjectName("btn_openFile")
        self.gridLayout_4.addWidget(self.btn_openFile, 1, 2, 1, 1)
        self.toolBox.addItem(self.page, "")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setGeometry(QtCore.QRect(0, 0, 329, 610))
        self.page_2.setObjectName("page_2")
        self.widget_4 = QtWidgets.QWidget(self.page_2)
        self.widget_4.setGeometry(QtCore.QRect(10, 30, 241, 101))
        self.widget_4.setObjectName("widget_4")
        self.label_8 = QtWidgets.QLabel(self.widget_4)
        self.label_8.setGeometry(QtCore.QRect(10, 10, 61, 21))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.widget_4)
        self.label_9.setGeometry(QtCore.QRect(50, 40, 41, 21))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.widget_4)
        self.label_10.setGeometry(QtCore.QRect(50, 70, 41, 21))
        self.label_10.setObjectName("label_10")
        self.zft_2d_start_y_spinBox = QtWidgets.QSpinBox(self.widget_4)
        self.zft_2d_start_y_spinBox.setGeometry(QtCore.QRect(170, 40, 61, 22))
        self.zft_2d_start_y_spinBox.setMaximum(999999999)
        self.zft_2d_start_y_spinBox.setObjectName("zft_2d_start_y_spinBox")
        self.zft_2d_start_x_spinBox = QtWidgets.QSpinBox(self.widget_4)
        self.zft_2d_start_x_spinBox.setGeometry(QtCore.QRect(100, 40, 61, 22))
        self.zft_2d_start_x_spinBox.setMaximum(999999999)
        self.zft_2d_start_x_spinBox.setObjectName("zft_2d_start_x_spinBox")
        self.zft_2d_end_y_spinBox = QtWidgets.QSpinBox(self.widget_4)
        self.zft_2d_end_y_spinBox.setGeometry(QtCore.QRect(170, 70, 61, 22))
        self.zft_2d_end_y_spinBox.setMaximum(999999999)
        self.zft_2d_end_y_spinBox.setObjectName("zft_2d_end_y_spinBox")
        self.zft_2d_end_x_spinBox = QtWidgets.QSpinBox(self.widget_4)
        self.zft_2d_end_x_spinBox.setGeometry(QtCore.QRect(100, 70, 61, 22))
        self.zft_2d_end_x_spinBox.setMaximum(999999999)
        self.zft_2d_end_x_spinBox.setObjectName("zft_2d_end_x_spinBox")
        self.zft_2d_checkBox = QtWidgets.QCheckBox(self.widget_4)
        self.zft_2d_checkBox.setGeometry(QtCore.QRect(70, 10, 71, 21))
        self.zft_2d_checkBox.setObjectName("zft_2d_checkBox")
        self.zft_2d_bins_start_spinBox = QtWidgets.QSpinBox(self.page_2)
        self.zft_2d_bins_start_spinBox.setGeometry(QtCore.QRect(70, 140, 61, 22))
        self.zft_2d_bins_start_spinBox.setMaximum(180)
        self.zft_2d_bins_start_spinBox.setProperty("value", 180)
        self.zft_2d_bins_start_spinBox.setObjectName("zft_2d_bins_start_spinBox")
        self.zft_2d_bins_end_spinBox = QtWidgets.QSpinBox(self.page_2)
        self.zft_2d_bins_end_spinBox.setGeometry(QtCore.QRect(140, 140, 61, 22))
        self.zft_2d_bins_end_spinBox.setMaximum(256)
        self.zft_2d_bins_end_spinBox.setProperty("value", 256)
        self.zft_2d_bins_end_spinBox.setObjectName("zft_2d_bins_end_spinBox")
        self.label_11 = QtWidgets.QLabel(self.page_2)
        self.label_11.setGeometry(QtCore.QRect(10, 140, 51, 21))
        self.label_11.setObjectName("label_11")
        self.zft_2d_H0_spinBox = QtWidgets.QSpinBox(self.page_2)
        self.zft_2d_H0_spinBox.setGeometry(QtCore.QRect(70, 170, 61, 22))
        self.zft_2d_H0_spinBox.setMaximum(256)
        self.zft_2d_H0_spinBox.setObjectName("zft_2d_H0_spinBox")
        self.zft_2d_H1_spinBox = QtWidgets.QSpinBox(self.page_2)
        self.zft_2d_H1_spinBox.setGeometry(QtCore.QRect(140, 170, 61, 22))
        self.zft_2d_H1_spinBox.setMaximum(180)
        self.zft_2d_H1_spinBox.setProperty("value", 180)
        self.zft_2d_H1_spinBox.setObjectName("zft_2d_H1_spinBox")
        self.label_12 = QtWidgets.QLabel(self.page_2)
        self.label_12.setGeometry(QtCore.QRect(10, 170, 51, 21))
        self.label_12.setObjectName("label_12")
        self.zft_2d_S0_spinBox = QtWidgets.QSpinBox(self.page_2)
        self.zft_2d_S0_spinBox.setGeometry(QtCore.QRect(70, 200, 61, 22))
        self.zft_2d_S0_spinBox.setMaximum(256)
        self.zft_2d_S0_spinBox.setObjectName("zft_2d_S0_spinBox")
        self.zft_2d_S1_spinBox = QtWidgets.QSpinBox(self.page_2)
        self.zft_2d_S1_spinBox.setGeometry(QtCore.QRect(140, 200, 61, 22))
        self.zft_2d_S1_spinBox.setMaximum(256)
        self.zft_2d_S1_spinBox.setProperty("value", 256)
        self.zft_2d_S1_spinBox.setObjectName("zft_2d_S1_spinBox")
        self.label_13 = QtWidgets.QLabel(self.page_2)
        self.label_13.setGeometry(QtCore.QRect(10, 200, 51, 21))
        self.label_13.setObjectName("label_13")
        self.zft_2d_openfile_btn = QtWidgets.QPushButton(self.page_2)
        self.zft_2d_openfile_btn.setGeometry(QtCore.QRect(240, 0, 75, 23))
        self.zft_2d_openfile_btn.setObjectName("zft_2d_openfile_btn")
        self.zft_2d_OK_btn = QtWidgets.QPushButton(self.page_2)
        self.zft_2d_OK_btn.setGeometry(QtCore.QRect(210, 170, 75, 23))
        self.zft_2d_OK_btn.setObjectName("zft_2d_OK_btn")
        self.zft_2d_yszf_btn = QtWidgets.QPushButton(self.page_2)
        self.zft_2d_yszf_btn.setGeometry(QtCore.QRect(210, 200, 75, 23))
        self.zft_2d_yszf_btn.setObjectName("zft_2d_yszf_btn")
        self.frame_5 = QtWidgets.QFrame(self.page_2)
        self.frame_5.setGeometry(QtCore.QRect(0, 240, 321, 361))
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.widget_8 = QtWidgets.QWidget(self.frame_5)
        self.widget_8.setGeometry(QtCore.QRect(10, 10, 301, 341))
        self.widget_8.setObjectName("widget_8")
        self.toolBox.addItem(self.page_2, "")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setGeometry(QtCore.QRect(0, 0, 329, 610))
        self.page_3.setObjectName("page_3")
        self.zft_fx_openfile_btn = QtWidgets.QPushButton(self.page_3)
        self.zft_fx_openfile_btn.setGeometry(QtCore.QRect(230, 10, 75, 23))
        self.zft_fx_openfile_btn.setObjectName("zft_fx_openfile_btn")
        self.zft_fx_ok_btn = QtWidgets.QPushButton(self.page_3)
        self.zft_fx_ok_btn.setGeometry(QtCore.QRect(230, 40, 75, 23))
        self.zft_fx_ok_btn.setObjectName("zft_fx_ok_btn")
        self.label_14 = QtWidgets.QLabel(self.page_3)
        self.label_14.setGeometry(QtCore.QRect(20, 40, 51, 21))
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.page_3)
        self.label_15.setGeometry(QtCore.QRect(20, 10, 51, 21))
        self.label_15.setObjectName("label_15")
        self.zft_fx_S1_spinBox = QtWidgets.QSpinBox(self.page_3)
        self.zft_fx_S1_spinBox.setGeometry(QtCore.QRect(150, 70, 61, 22))
        self.zft_fx_S1_spinBox.setMaximum(256)
        self.zft_fx_S1_spinBox.setProperty("value", 256)
        self.zft_fx_S1_spinBox.setObjectName("zft_fx_S1_spinBox")
        self.zft_fx_S0_spinBox = QtWidgets.QSpinBox(self.page_3)
        self.zft_fx_S0_spinBox.setGeometry(QtCore.QRect(80, 70, 61, 22))
        self.zft_fx_S0_spinBox.setMaximum(256)
        self.zft_fx_S0_spinBox.setObjectName("zft_fx_S0_spinBox")
        self.label_16 = QtWidgets.QLabel(self.page_3)
        self.label_16.setGeometry(QtCore.QRect(20, 70, 51, 21))
        self.label_16.setObjectName("label_16")
        self.zft_fx_bins1_spinBox = QtWidgets.QSpinBox(self.page_3)
        self.zft_fx_bins1_spinBox.setGeometry(QtCore.QRect(150, 10, 61, 22))
        self.zft_fx_bins1_spinBox.setMaximum(256)
        self.zft_fx_bins1_spinBox.setProperty("value", 256)
        self.zft_fx_bins1_spinBox.setObjectName("zft_fx_bins1_spinBox")
        self.zft_fx_H0_spinBox = QtWidgets.QSpinBox(self.page_3)
        self.zft_fx_H0_spinBox.setGeometry(QtCore.QRect(80, 40, 61, 22))
        self.zft_fx_H0_spinBox.setMaximum(256)
        self.zft_fx_H0_spinBox.setObjectName("zft_fx_H0_spinBox")
        self.zft_fx_bins0_spinBox = QtWidgets.QSpinBox(self.page_3)
        self.zft_fx_bins0_spinBox.setGeometry(QtCore.QRect(80, 10, 61, 22))
        self.zft_fx_bins0_spinBox.setMaximum(180)
        self.zft_fx_bins0_spinBox.setProperty("value", 180)
        self.zft_fx_bins0_spinBox.setObjectName("zft_fx_bins0_spinBox")
        self.zft_fx_H1_spinBox = QtWidgets.QSpinBox(self.page_3)
        self.zft_fx_H1_spinBox.setGeometry(QtCore.QRect(150, 40, 61, 22))
        self.zft_fx_H1_spinBox.setMaximum(180)
        self.zft_fx_H1_spinBox.setProperty("value", 180)
        self.zft_fx_H1_spinBox.setObjectName("zft_fx_H1_spinBox")
        self.zft_fx_save_btn = QtWidgets.QPushButton(self.page_3)
        self.zft_fx_save_btn.setGeometry(QtCore.QRect(230, 70, 75, 23))
        self.zft_fx_save_btn.setObjectName("zft_fx_save_btn")
        self.frame_6 = QtWidgets.QFrame(self.page_3)
        self.frame_6.setGeometry(QtCore.QRect(0, 100, 321, 501))
        self.frame_6.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.widget_9 = QtWidgets.QWidget(self.frame_6)
        self.widget_9.setGeometry(QtCore.QRect(10, 10, 301, 481))
        self.widget_9.setObjectName("widget_9")
        self.toolBox.addItem(self.page_3, "")
        self.gridLayout_3.addWidget(self.toolBox, 1, 0, 1, 1)
        self.gridLayout_2.addWidget(self.frame_2, 0, 0, 1, 1)
        self.v_Layout_dis = QtWidgets.QVBoxLayout()
        self.v_Layout_dis.setObjectName("v_Layout_dis")
        self.gridLayout_2.addLayout(self.v_Layout_dis, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.widget_10, 0, 0, 1, 1)

        self.retranslateUi(Form)
        self.toolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_2.setText(_translate("Form", "交互信息"))
        self.label_3.setText(_translate("Form", "识别结果"))
        self.label_7.setText(_translate("Form", "识别算法"))
        self.rbn_module.setText(_translate("Form", "模板匹配"))
        self.rbn_svm.setText(_translate("Form", "SVM"))
        self.rbn_cnn.setText(_translate("Form", "CNN"))
        self.btn_connectOPC.setText(_translate("Form", "连接OPC"))
        self.label.setText(_translate("Form", "连接状态"))
        self.btn_disconnectOPC.setText(_translate("Form", "关闭OPC"))
        self.btn_reconginze.setText(_translate("Form", "开始识别"))
        self.btn_openFile.setText(_translate("Form", "打开文件"))
        self.label_8.setText(_translate("Form", "矩形掩模："))
        self.label_9.setText(_translate("Form", "<html><head/><body><p align=\"right\">起点：</p></body></html>"))
        self.label_10.setText(_translate("Form", "<html><head/><body><p align=\"right\">终点：</p></body></html>"))
        self.zft_2d_checkBox.setText(_translate("Form", "True"))
        self.label_11.setText(_translate("Form", "<html><head/><body><p align=\"right\">Bins：</p></body></html>"))
        self.label_12.setText(_translate("Form", "<html><head/><body><p align=\"right\">H_Ranges：</p></body></html>"))
        self.label_13.setText(_translate("Form", "<html><head/><body><p align=\"right\">S_Ranges：</p></body></html>"))
        self.zft_2d_openfile_btn.setText(_translate("Form", "打开文件"))
        self.zft_2d_OK_btn.setText(_translate("Form", "确定"))
        self.zft_2d_yszf_btn.setText(_translate("Form", "颜色直方图"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), _translate("Form", "2D直方图"))
        self.zft_fx_openfile_btn.setText(_translate("Form", "打开文件"))
        self.zft_fx_ok_btn.setText(_translate("Form", "确定"))
        self.label_14.setText(_translate("Form", "<html><head/><body><p align=\"right\">H_Ranges：</p></body></html>"))
        self.label_15.setText(_translate("Form", "<html><head/><body><p align=\"right\">Bins：</p></body></html>"))
        self.label_16.setText(_translate("Form", "<html><head/><body><p align=\"right\">S_Ranges：</p></body></html>"))
        self.zft_fx_save_btn.setText(_translate("Form", "保存"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_3), _translate("Form", "反向投影"))
