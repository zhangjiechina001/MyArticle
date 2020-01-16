# -*- coding: utf-8 -*-
from gui_demo_ui import Ui_Form
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtCore import QTimer
import cv2
import numpy as np
from matplotlib.patches import Rectangle
import time
import os

class demoWindows(QWidget):
    def __del__(self):
        self.camera.release()# 释放资源
        if self.out_release_flag:
            self.out.release()

    def init_fun(self):
        self.window = Ui_Form()
        self.window.setupUi(self)
        self.showandhideframe()
        self.timer = QTimer()# 定义一个定时器对象
        self.timer.timeout.connect(self.timer_fun) #计时结束调用方法

        self.window.pushButton_2.clicked.connect(self.timer_start)# 这几行就是绑定按键的处理事件
        self.window.pushButton_3.clicked.connect(self.recod_image)
        self.window.pushButton_4.clicked.connect(self.timer_stop)
        self.window.pushButton_5.clicked.connect(self.outfilespath_fun)
        self.out_release_flag = False
        self.tmp_img = np.zeros((720,1080,3), np.uint8)
        self.window.comboBox_type.currentIndexChanged.connect(self.showandhideframe)
        self.window.OK_Btn.clicked.connect(self.draw_lines_fun)
        self.window.pushButton.clicked.connect(self.open_btn_fun)

        self.window.figure1.canvas.mpl_connect("button_press_event", self.figure1_on_press)# 绑定鼠标按下事件
        self.window.figure1.canvas.mpl_connect("button_release_event", self.figure1_on_release)# 绑定鼠标松下按键

    '''
    showandhideframe():这个函数使用来根据选择绘制类型进行显示对应的参数
    '''
    def showandhideframe(self):
        if self.window.comboBox_type.currentText() == "直线":
            self.window.frame_line_range.show()
            self.window.frame_yuan.hide()
            self.window.frame_tuoyuan.hide()
            self.window.frame_wenzi.hide()
            # print(self.comboBox_type.currentText())
        elif self.window.comboBox_type.currentText() == "矩形":
            self.window.frame_line_range.show()
            self.window.frame_yuan.hide()
            self.window.frame_tuoyuan.hide()
            self.window.frame_wenzi.hide()
        elif self.window.comboBox_type.currentText() == "圆":
            self.window.frame_line_range.hide()
            self.window.frame_yuan.show()
            self.window.frame_tuoyuan.hide()
            self.window.frame_wenzi.hide()
        elif self.window.comboBox_type.currentText() == "椭圆":
            self.window.frame_line_range.hide()
            self.window.frame_yuan.hide()
            self.window.frame_tuoyuan.show()
            self.window.frame_wenzi.hide()
        elif self.window.comboBox_type.currentText() == "文字":
            self.window.frame_line_range.hide()
            self.window.frame_yuan.hide()
            self.window.frame_tuoyuan.hide()
            self.window.frame_wenzi.show()
        else:
            # print(self.comboBox_type.currentText())
            self.window.frame_line_range.hide()
            self.window.frame_yuan.hide()
            self.window.frame_tuoyuan.hide()
            self.window.frame_wenzi.hide()

    '''
    figure1_on_press()：鼠标的按下事件的处理函数，参数event是必须的，可以根据event中的参数判断是左键还是右键等
    '''
    def figure1_on_press(self, event):
        self.x0 = int(event.xdata)
        self.y0 = int(event.ydata)
        while len(self.window.figaxes1.patches)>0:
            del self.window.figaxes1.patches[0]
        self.figure1_rect = Rectangle((0,0), 0, 0, linestyle='solid', fill=False, edgecolor='red')
        self.window.figaxes1.add_patch(self.figure1_rect)
    '''
    鼠标的松开事件处理函数，参数event是必须的
    '''
    def figure1_on_release(self, event):
        self.x1 = int(event.xdata)
        self.y1 = int(event.ydata)
        self.figure1_rect.set_width(self.x1 - self.x0 + 1)
        self.figure1_rect.set_height(self.y1 - self.y0 + 1)
        self.figure1_rect.set_xy((self.x0, self.y0))
        self.window.figure1.canvas.draw()

    def open_btn_fun(self):
        fileName = self.open_image_file()
        if fileName:
            self.img = cv2.imread(fileName)
            self.showimg2figaxes(self.img)

    def showimg2figaxes(self,img):
        b, g, r = cv2.split(img)
        imgret = cv2.merge([r,g,b])# 这个就是前面说书的，OpenCV和matplotlib显示不一样，需要转换
        self.window.figaxes1.clear()
        self.window.figaxes1.imshow(imgret)
        self.window.figaxes1.autoscale_view()
        self.window.figure1.canvas.draw()

    def open_image_file(self):
        '''打开一个图像文件'''
        fileName, filetype= QFileDialog.getOpenFileName(self.window.page, 
            "open file", '.', "jpg Files (*.jpg);;png Files (*.png);;All Files (*)")
        return fileName

    def draw_lines_fun(self):
        '''画图形函数'''
        b = self.window.RGB_R_spinBox.value()
        g = self.window.RGB_G_spinBox.value()
        r = self.window.RGB_B_spinBox.value()
        lines_type = self.window.spinBox_linetype.value()
        if lines_type == 0:
            lines_type = -1
        type_lines = self.window.comboBox_type.currentText()
        if hasattr(self, "img"):
            self.openimg2 = self.img
        else:
            self.openimg2 = self.tmp_img

        if type_lines == "直线" or type_lines == "矩形":
            x0 = self.window.line_range_x_start.value()
            y0 = self.window.line_range_y_start.value()
            x1 = self.window.line_range_x_stop.value()
            y1 = self.window.line_range_y_stop.value()
            if type_lines == "直线":
                if lines_type == -1:
                    lines_type = 1
                cv2.line(self.openimg2, (x0,y0), (x1,y1), (r,g,b), lines_type)
            else:
                cv2.rectangle(self.openimg2, (x0,y0), (x1,y1), (r,g,b), lines_type)
        elif type_lines == "圆":
            x0 = self.window.yuanxin_x.value()
            y0 = self.window.yuanxin_y.value()
            banjing = self.window.yuan_banjing.value()
            cv2.circle(self.openimg2, (x0,y0), banjing, (r,g,b), lines_type)
        elif type_lines == "椭圆":
            x0 = self.window.zhongxindian_x.value()
            y0 = self.window.zhongxindian_y.value()
            chang = self.window.changzhou.value()
            duan = self.window.duanzhou.value()
            jiao = self.window.jiaodu.value()
            jiao_start = self.window.jiaodu_start.value()
            jiao_end = self.window.jiaodu_end.value()
            cv2.ellipse(self.openimg2, (x0,y0), (chang, duan), jiao, jiao_start, jiao_end, (r,g,b), lines_type)
        elif type_lines == "文字":
            wenzi = self.window.putText.text()
            x0 = self.window.putText_x.value()
            y0 = self.window.putText_y.value()
            font_type = self.window.putText_ziti.currentText()
            if font_type == "FONT_HERSHEY_TRIPLEX":
                font = cv2.FONT_HERSHEY_TRIPLEX
            elif font_type == "FONT_HERSHEY_PLAIN":
                font = cv2.FONT_HERSHEY_PLAIN
            elif font_type == "FONT_HERSHEY_DUPLEX":
                font = cv2.FONT_HERSHEY_DUPLEX
            elif font_type == "FONT_HERSHEY_COMPLEX":
                font = cv2.FONT_HERSHEY_COMPLEX
            elif font_type == "FONT_HERSHEY_SIMPLEX":
                font = cv2.FONT_HERSHEY_SIMPLEX
            elif font_type == "FONT_HERSHEY_COMPLEX_SMALL":
                font = cv2.FONT_HERSHEY_COMPLEX_SMALL

            wenzi_size = self.window.putText_daxiao.value()
            wenzi_type = self.window.putText_type.currentText()
            if wenzi_type == "LINE_AA":
                ziti_type = cv2.LINE_AA
            elif wenzi_type == "LINE_8":
                ziti_type = cv2.LINE_8
            elif wenzi_type == "LINE_4":
                ziti_type = cv2.LINE_4

            cv2.putText(self.openimg2, wenzi, (x0,y0), font, wenzi_size, (r,g,b), ziti_type)

        else:
            pass
        self.showimg2figaxes(self.openimg2)

    def timer_fun(self):
        ret, frame = self.camera.read()
        if ret:
            if self.recod_flag :
                # frame = cv2.flip(frame, 1)# 在帧上进行操作 左右翻转了
                self.out.write(frame) # 保存视频
            self.showimg2figaxes(frame)
        else:
            self.timer.stop()

    def timer_start(self):
        self.recod_flag = False
        if self.window.radioButton.isChecked():
            # if not self.camera.isOpened():
            #     self.camera.open()
            self.camera = cv2.VideoCapture(0)
        else:
            fileName, filetype= QFileDialog.getOpenFileName(self.window.page, 
                "open file", '.', "avi Files (*.avi);;mp4 Files (*.mp4);;All Files (*)")
            if fileName:
                self.camera = cv2.VideoCapture(fileName)
            else:
                return None
        self.timer.start(25) #设置计时间隔并启动

    def timer_stop(self):
        if self.recod_flag :
            self.recod_flag = False
            self.out.release()
            self.out_release_flag = False
        self.timer.stop()

    def recod_image(self):
        self.recod_flag = True
        self.out_release_flag = True
        self.fourcc = cv2.VideoWriter_fourcc(*"MPEG")
        # filename = self.get_time_fimename("./")
        filename = self.get_time_fimename("\\")
        # filename = filename + ".avi"
        if not os.path.exists(self.window.lineEdit.text()):
            os.makedirs(self.window.lineEdit.text())
        filename = self.window.lineEdit.text() + filename + ".avi"
        # print(filename)
        self.out = cv2.VideoWriter(filename,self.fourcc,20,(640,480))
        self.timer.start(25) #设置计时间隔并启动

    def get_time_fimename(self, option):
        tmp_time = time.localtime()
        tmp_nyr = str(tmp_time.tm_year)+str(tmp_time.tm_mon)+str(tmp_time.tm_mday)
        tmp_sfm = str(tmp_time.tm_hour)+str(tmp_time.tm_min)+str(tmp_time.tm_sec)
        return option+tmp_nyr+tmp_sfm

    def outfilespath_fun(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.window.lineEdit.setText(dirname)

if __name__ == '__main__':
    
    import sys
    from PyQt5.QtWidgets import QApplication , QMainWindow

    app = QApplication(sys.argv)
    mainW = QMainWindow()
    mainW.resize(1191, 686)
    ui = demoWindows(mainW)
    ui.init_fun()
    mainW.show()
    sys.exit(app.exec_())
