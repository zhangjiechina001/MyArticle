import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QDesktopWidget,QPushButton,QHBoxLayout,QWidget

class CenterForm(QMainWindow):
    def __init__(self):
        super(CenterForm,self).__init__()

        self.setWindowTitle('first Form App')
        self.resize(400,300)
        ##添加一个button
        self.button1=QPushButton('退出应用程序')
        #将button放到布局，再将布局放到QWidget中，最后将mainForm放到主窗体中
        self.button1.clicked.connect(self.cliclbutton)
        layout=QHBoxLayout()
        layout.addWidget(self.button1)

        mainForm=QWidget()
        mainForm.setLayout(layout)
        self.setCentralWidget(mainForm)
    #按钮单击时间的方法
    def cliclbutton(self):
        sender=self.sender()
        print(sender.text()+'按钮被按下')
        app=QApplication.instance()
        app.quit()

    def center(self):
        screen=QDesktopWidget().screenGeometry()

        # size=self.geometry()
        # newLeft=


if __name__=='__main__':
    app=QApplication(sys.argv)
    main=CenterForm()
    main.show()
    sys.exit(app.exec_())