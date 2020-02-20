from PyQt5.Qt import *
import sys

class LabelButtonBackground(QWidget):
    def __init__(self):
        super(LabelButtonBackground,self).__init__()
        self.setWindowTitle('使用QSS为文本添加标签')
        label=QLabel()
        label.setToolTip('这是一个文本标签')
        label.setStyleSheet('QLabel{border-image:url(./img_python.jpg);}')
        label.setFixedWidth(400)
        label.setFixedHeight(300)

        btn1=QPushButton()
        btn1.setText('btn1')
        btn1.setObjectName('btn1')
        btn1.setMaximumSize(50,50)
        btn1.setMinimumSize(50,50)

        btn2=QPushButton()
        btn2.setText('btn2')
        btn2.setObjectName('btn2')
        btn2.setMaximumSize(50,50)
        btn2.setMinimumSize(50,50)

        style='''
        QPushButton#btn2
        {
        border-radius:25px;
        background-image:url(./black.png);
        }
        QPushButton#btn2:Pressed{
        background-image:url(./img_python.jpg);
        }'''
        btn2.setStyleSheet(style)

        style='''
        QPushButton#btn1
        {
        border-radius:10px;
        background-image:url(./black.png);
        }
        QPushButton#btn1:Pressed{
        background-image:url(./img_python.jpg);
        }'''
        btn1.setStyleSheet(style)

        vbox=QVBoxLayout()
        vbox.addWidget(label)
        vbox.addStretch()
        vbox.addWidget(btn1)
        vbox.addWidget(btn2)
        self.setLayout(vbox)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = LabelButtonBackground()
    main.show()
    sys.exit(app.exec_())