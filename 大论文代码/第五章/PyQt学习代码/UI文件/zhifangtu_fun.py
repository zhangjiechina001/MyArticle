import datetime
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
import matplotlib.pyplot as plt
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QSizePolicy, QFileDialog, QTableWidgetItem
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
import cv2
from matplotlib.figure import Figure
from recongnize_image import recongnize_image

from zhifangtu import Ui_Form

import cv2 as cv
from opcua import ua, Server

class ua_server():
    #初始化opcua，订阅时间和触发事件方法
    def __init__(self):
        # setup our server
        self.server = Server()
        self.server.set_endpoint("opc.tcp://0.0.0.0:4840/freeopcua/server/")
        # setup our own namespace, not really necessary but should as spec
        uri = "http://examples.freeopcua.github.io"
        self.idx = self.server.register_namespace(uri)
        self.init_event()
        self.init_value_change()


    def init_event(self):
        # get Objects node, this is where we should put our custom stuff
        objects = self.server.get_objects_node()

        # populating our address space
        myobj = objects.add_object(self.idx, "MyObject")

        # Creating a custom event: Approach 1
        # The custom event object automatically will have members from its parent (BaseEventType)
        etype = self.server.create_custom_event_type(self.idx, 'MyFirstEvent', ua.ObjectIds.BaseEventType,
                                                [('MyNumericProperty', ua.VariantType.Float),
                                                 ('MyStringProperty', ua.VariantType.String)])
        self.myevgen = self.server.get_event_generator(etype, myobj)
        #配置数据监测
        # 取得根节点
        # objects = self.server.get_objects_node()
        # 添加自我节点
        # myobj = objects.add_object(self.idx, "MyObject")
        self.myvar = myobj.add_variable(self.idx, "my_change_data", ua.Variant(0, ua.VariantType.ByteString))
        self.myvar.set_writable()
        self.server.start()

    def init_value_change(self):
        pass

    def trigger_event(self,message):
        self.myevgen.event.Message = ua.LocalizedText(message)
        self.myevgen.event.Severity = 0
        self.myevgen.event.MyNumericProperty = 0
        self.myevgen.event.MyStringProperty = "Property " + str(0)
        self.myevgen.trigger()

    def trigger_data_value(self,message):
        self.myvar.set_value(message)


class DealImgThread(QThread):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # 定义五个信号
    # 源图像
    signal_sourceImg = pyqtSignal(QImage)
    # 两次处理的曲线
    signal_2cut = pyqtSignal(list)
    # 展开并且形态学处理后的图像
    signal_morphology = pyqtSignal(list)
    # 定位识别效果图
    signal_recongnize_result = pyqtSignal(list)
    # 识别信息
    signal_recongnize_info = pyqtSignal(str)
    #识别结果回调
    signal_recongnize_str_result=pyqtSignal(list)

    def __init__(self, parent=None,recongnize_fun=None):
        super(DealImgThread, self).__init__(parent)
        self.image = None
        self.garryIsOpen = False
        self.threadIsOpen = True
        self.histCatch = 0
        self.recongnize = recongnize_image()
        self.recongnize_fun=recongnize_fun

    def get_image(self, img):
        self.image = img

    def openGarry(self):
        if (self.garryIsOpen == False):
            self.garryIsOpen = True

    def end(self):
        if (self.threadIsOpen):
            self.threadIsOpen = False

    def sort_and_merge(self, img_list_info):
        _img_list_info = sorted(img_list_info, key=lambda img_list_info: img_list_info[0])
        ret_str = ''
        for info in _img_list_info:
            x, code = info
            ret_str += code
        return ret_str

    def run(self):
        #记录开始识别的时间
        start = datetime.datetime.now()
        copyImg = self.image.copy()
        smallImg = self.recongnize.reduceImg(copyImg)
        self.signal_recongnize_info.emit('两次金字塔缩小处理完成')
        thresh, binary = self.recongnize.binaryImage(smallImg, binary_type=cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        self.signal_recongnize_info.emit('二值化处理完成,阈值为：{0}'.format(str(thresh)))
        circle_point, circle_r, drawImg = self.recongnize.getPointAndR(binary)
        self.signal_recongnize_info.emit("圆心:{0},半径：{1}".format(str(circle_point), str(circle_r)))

        # 两次展开
        unflood_img1 = self.recongnize.unfloodImage(smallImg, point=circle_point, unfloodR=circle_r, width=40,
                                                    startTheta=0)
        unflood_img1 = cv2.Canny(unflood_img1, 100, 200)
        # 绘制直方分布图
        ret_max1, ret_theta1, hist1 = self.recongnize.calcMax(unflood_img1)

        # 2.展开角为90°
        import math
        unflood_img2 = self.recongnize.unfloodImage(smallImg, point=circle_point, unfloodR=circle_r, width=40,
                                                    startTheta=math.pi / 2)
        unflood_img2 = cv2.Canny(unflood_img2, 100, 200)
        ret_max2, ret_theta2, hist2 = self.recongnize.calcMax(unflood_img2)
        self.signal_recongnize_info.emit(
            "两次展开完成，第一次角度：{0},max:{1},第二次角度：{2},max:{3}".format(str(ret_theta1)[0:4], str(ret_max1),
                                                                str(ret_theta2)[0:4], str(ret_max2)))
        dis_list = [hist1, hist2]
        self.signal_2cut.emit(dis_list)
        # 确定展开角度
        if (ret_max1 > ret_max2):
            theta = ret_theta1
        else:
            theta = ret_theta2 + math.pi / 2
        # 做了两次缩小后的图片，现在进行两次放大
        point = (circle_point[0] * 4, circle_point[1] * 4)
        r = circle_r * 4
        # 看圆的半径，得出型号，一个为109，一个为208,208的字符高度为50左右，109的为80左右
        unflood_srcImg = self.recongnize.unfloodImagePro(copyImg, point=point, unfloodR=r, width=130,
                                                         startTheta=theta - 0.03)
        self.signal_recongnize_info.emit('圆心为:{0},半径为:{1}'.format(str(point), str(r)))
        if (circle_r > 204):
            unflood_srcImg = unflood_srcImg[40:110, :]
        else:
            unflood_srcImg = unflood_srcImg[0:100, :]
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binaryUnfloodImg = cv2.adaptiveThreshold(unflood_srcImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                                 blockSize=17, C=10)
        dst_img = binaryUnfloodImg
        # CLOSE先膨胀再腐蚀
        dst_img = cv2.morphologyEx(dst_img, cv2.MORPH_CLOSE, kernel)
        # OPEN先腐蚀再膨胀
        dst_img = cv2.morphologyEx(dst_img, cv2.MORPH_OPEN, kernel)
        self.signal_morphology.emit([dst_img])
        # 字符切割
        _, contours, _ = cv2.findContours(dst_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        unflood_srcImg = cv2.cvtColor(unflood_srcImg, cv2.COLOR_GRAY2BGR)

        if (circle_r > 204):
            h_thresh = 42
            code_high = 22
            code_size = 1
        else:
            h_thresh = 78
            code_high = 25
            code_size = 1.5
        fun = self.recongnize_fun
        import 第四章.模板匹配算法.formated.模板匹配识别字符 as detect
        import 第四章.svm.svm识别字符模型训练 as svm_detect
        img_list = []
        img_info_list = []

        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if (area > 100):
                x, y, w, h = cv2.boundingRect(contours[i])
                if (h > h_thresh):
                    count += 1
                    unflood_srcImg_copy = unflood_srcImg.copy()
                    cv2.rectangle(unflood_srcImg, (x - 3, y - 3), (x + w + 3, y + h + 3), color=(255, 0, 0))
                    cutNum = 0
                    # tempImg=unflood_srcImg[x-cutNum:x+w+cutNum,y-cutNum:y+h+cutNum,:]
                    tempImg = unflood_srcImg[y - cutNum:y + h + cutNum, x - cutNum:x + w + cutNum, :]
                    tempImg = cv2.cvtColor(tempImg, cv2.COLOR_RGB2GRAY)
                    cutNum = 3
                    tempImg_1 = unflood_srcImg_copy[y - cutNum:y + h + cutNum, x - cutNum:x + w + cutNum, :]
                    tempImg_1 = cv2.cvtColor(tempImg_1, cv2.COLOR_RGB2GRAY)
                    img_list.append(tempImg_1)
                    if fun == 'template matching':
                        ret_code = detect.detect_code(tempImg, show_plt=False)
                        unflood_srcImg = cv2.putText(unflood_srcImg, str(ret_code), (x, y + code_high),
                                                     cv2.FONT_HERSHEY_COMPLEX, code_size, (255, 0, 0), 2)
                    if fun == 'svm':
                        # if(tempImg.width>0):
                        ret_code = svm_detect.dettect_code(tempImg)
                        # unflood_srcImg=cv2.putText(unflood_srcImg,str(ret_code),(x,y+code_high),cv2.FONT_HERSHEY_COMPLEX,code_size,(255,0,0),2)
                    img_info_list.append((x, ret_code))

        ret_str = self.sort_and_merge(img_info_list)
        self.signal_recongnize_result.emit([unflood_srcImg])
        self.signal_recongnize_info.emit('{0}识别结束，共{1}个字符，结果为：{2}'.format(fun, str(count), str(ret_str)))
        end = datetime.datetime.now()
        totalMs = str((end - start).total_seconds()*1000)[0:8]
        self.signal_recongnize_str_result.emit([count,totalMs,ret_str])

# 首先定义一个继承自FigureCanvas的类
class Mydemo(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.rcParams['font.family'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 创建一个2*2布局的表格
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        w = 2
        h = 2
        self.axes_1 = self.fig.add_subplot(3, h, 1)
        self.axes_2 = self.fig.add_subplot(3, h, 2)
        self.axes_3 = self.fig.add_subplot(3, 1, 2)
        self.axes_4 = self.fig.add_subplot(3, 1, 3)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.fig.tight_layout()
        #设置识别方法SVM或者模板匹配

# #创建一个和opcua通讯相关的类
# class opcua_communication():

import cv2
# 使用多线程进行控制，后台线程负责数据处理，主线程GUI负责数据显示刷新
class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.vlayout = QtWidgets.QVBoxLayout(self)
        # 创建图表的实例
        self.cavas = Mydemo(width=5, height=4, dpi=100)
        self.widget_toolbar = NavigationToolbar(self.cavas,self.widget_5)
        self.v_Layout_dis.addWidget(self.widget_toolbar)
        self.v_Layout_dis.addWidget(self.cavas)
        self.btn_openFile.clicked.connect(self.btn_open_file_ywzf_clicked)
        self.label.setText("未连接")  # 红色
        self.label.setStyleSheet("background-color:red;")
        self.btn_connectOPC.clicked.connect(self.connect_opcUA)
        self.btn_disconnectOPC.clicked.connect(self.disconnect_opcUA)
        self.radioAyyay = [self.rbn_module, self.rbn_svm, self.rbn_cnn]
        self.set_axes_title()
        self.statusTip()
        self.cvThread = DealImgThread()
        # 定义一个待处理的image
        self.source_image = None
        self.cvThread.signal_recongnize_info.connect(self.addItemAndFocusIndex)
        self.cvThread.signal_2cut.connect(self.two_cut_display)
        self.cvThread.signal_morphology.connect(self.mor_image_display)
        self.cvThread.signal_recongnize_result.connect(self.result_image_display)
        self.cvThread.signal_recongnize_str_result.connect(self.result_tab_item_add)
        self.btn_reconginze.clicked.connect(self.openThread)
        self.ua_server=ua_server()

    def result_tab_item_add(self,items):
        #获取tab行数
        row=self.tab_recongnizeInfo.rowCount()
        self.tab_recongnizeInfo.insertRow(row)
        clumn=0
        items[1]=items[1]+'ms'
        for item in items:
            self.tab_recongnizeInfo.setItem(row,clumn,QTableWidgetItem(str(item)))
            clumn+=1

    def result_image_display(self, image):
        self.cavas.axes_4.cla()
        image = image[0]
        self.cavas.axes_4.set_title('定位识别结果')
        self.cavas.axes_4.imshow(image, cmap='gray')
        self.cavas.draw()

    def mor_image_display(self, image):
        self.cavas.axes_3.cla()
        image = image[0]
        self.cavas.axes_3.set_title('形态学操作')
        self.cavas.axes_3.imshow(image, cmap='gray')
        self.cavas.draw()

    def two_cut_display(self, lines):
        self.cavas.axes_2.cla()
        self.cavas.axes_2.set_title('两次展开位置(r:1,g:2)')
        self.cavas.axes_2.plot(lines[0], label="第一次分割", color="r")
        self.cavas.axes_2.plot(lines[1], label="第二次分割", color="g")
        self.cavas.draw()

    def radio_clicked(self, sender):
        pass

    def open_image_file(self):
        '''打开一个图像文件'''
        fileName, filetype = QFileDialog.getOpenFileName(self,
                                                         "open file", '.',
                                                         "jpg Files (*.jpg);;png Files (*.png);;All Files (*)")

        item = '打开文件:' + fileName
        self.addItemAndFocusIndex(item)
        return fileName

    def disconnect_opcUA(self):
        self.label.setText("未连接")  # 红色
        self.label.setStyleSheet("background-color:red;")

    def connect_opcUA(self):
        self.label.setText("已连接")  # 红色
        self.label.setStyleSheet("background-color:green;")

    def set_axes_title(self):
        self.cavas.axes_1.set_title('原图')
        self.cavas.axes_2.set_title('两次展开位置')
        self.cavas.axes_3.set_title('形态学操作')
        self.cavas.axes_4.set_title('定位识别结果')

    def btn_open_file_ywzf_clicked(self):
        import numpy as np
        file_name = self.open_image_file()
        if file_name == '':
            return
        image = cv2.imread(filename=file_name, flags=cv2.IMREAD_GRAYSCALE)
        # 图片显示
        self.cavas.axes_1.imshow(image, cmap='gray')
        # self.cavas.axes_1.set_title('原图')
        # self.cavas.axes_2.hist(image.ravel(),256,[0,255],color='r')
        # self.cavas.axes_2.set_title('直方图')
        # self.cavas.axes_3.hist(image.ravel(),256,[0,255],color='g')
        # 在界面上显示
        self.cavas.draw()
        # 将图片赋给局部变量
        self.source_image = image
        # 将图片赋给cvThread中的图片
        self.cvThread.get_image(image)

    def addItemAndFocusIndex(self, item):
        self.ua_server.trigger_event(str(item))
        self.ua_server.trigger_data_value(str(item))
        self.lst_info.addItem(item)
        self.lst_info.setCurrentRow(self.lst_info.count() - 1)

    def openThread(self):
        #template matching
        self.recongnize_fun = 'template matching'
        if(self.rbn_module.isChecked()):
            self.recongnize_fun='template matching'
        elif(self.rbn_svm.isChecked()):
            self.recongnize_fun='svm'
        elif(self.rbn_cnn.isChecked()):
            self.recongnize_fun='template matching'

        if (self.cvThread.isRunning() == False):
            self.cvThread.recongnize_fun=self.recongnize_fun
            self.cvThread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())
