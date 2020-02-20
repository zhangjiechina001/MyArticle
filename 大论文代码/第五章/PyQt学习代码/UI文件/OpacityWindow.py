from PyQt5.Qt import *
import sys

if __name__=='__main__':
    app=QApplication(sys.argv)
    win=QMainWindow()
    win.setWindowTitle('窗口的透明度设置')
    win.setWindowOpacity(1.0)
    btn=QPushButton('我的按钮',win)
    radio=1.0
    def fun():
        global radio
        radio=radio-0.1
        return radio
    btn.clicked.connect(lambda :
                        win.setWindowOpacity(fun()))
    win.resize(300,200)
    win.show()
    sys.exit(app.exec_())