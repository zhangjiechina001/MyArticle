import sys
import test
from PyQt5.QtWidgets import  QApplication,QWidget

if __name__=='__main__':
    #创建QApplication
    app=QApplication(sys.argv)
    #创建一个窗口

    w=QWidget()
    #设置窗口尺寸
    w.resize(480,540)
    #移动窗口
    w.move(960,540)

    #设置窗口标题
    w.setWindowTitle('first application gui')
    w.show()
    #进入程序主循环，并通过exit函数确保主循环安全结束
    sys.exit(app.exec_())