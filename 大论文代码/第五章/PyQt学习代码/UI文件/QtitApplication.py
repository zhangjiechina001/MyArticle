import sys
from PyQt5.QtWidgets import QHBoxLayout,QMainWindow,QApplication,QPushButton,QWidget

class QuitApplication(QMainWindow):
    def __init__(self):
        super(QuitApplication,self).__init__()
        self.resize(300,200)
        self.setWindowTitle('退出应用程序')
        self.button=QPushButton('退出应用程序')
        layout=QHBoxLayout()
        layout.addWidget(self.button)

        mainFrame=QWidget()
        mainFrame.setLayout(layout)
        self.setCentralWidget(mainFrame)
        self.button.clicked.connect(self.closeBtnClick)
        self.button.setToolTip('今天天气不错！')

    def closeBtnClick(self):
        sender=self.sender()
        print(sender.text()+'按钮被按下')
        app=QApplication.instance()
        app.quit()

if __name__=='__main__':
    app=QApplication(sys.argv)
    main=QuitApplication()
    main.show()
    sys.exit(app.exec_())
