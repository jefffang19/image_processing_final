import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import cv2

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Frame(object):
    def setupUi(self, Frame):
        Frame.setObjectName("Frame")
        Frame.resize(1026, 785)
        self.verticalLayoutWidget = QtWidgets.QWidget(Frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(70, 20, 451, 251))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.horizontalSlider = QtWidgets.QSlider(Frame)
        self.horizontalSlider.setGeometry(QtCore.QRect(190, 659, 561, 41))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(Frame)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(49, 289, 221, 331))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(Frame)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(300, 290, 160, 331))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(Frame)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(490, 289, 160, 331))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label = QtWidgets.QLabel(Frame)
        self.label.setGeometry(QtCore.QRect(667, 70, 261, 261))
        self.label.setObjectName("label")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(Frame)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(680, 359, 241, 191))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_2 = QtWidgets.QLabel(Frame)
        self.label_2.setGeometry(QtCore.QRect(790, 670, 58, 15))
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Frame)
        QtCore.QMetaObject.connectSlotsByName(Frame)

    def retranslateUi(self, Frame):
        _translate = QtCore.QCoreApplication.translate
        Frame.setWindowTitle(_translate("Frame", "Frame"))
        self.pushButton.setText(_translate("Frame", "Select dataset"))
        self.pushButton_2.setText(_translate("Frame", "run"))
        self.label.setText(_translate("Frame", "TextLabel"))
        self.label_2.setText(_translate("Frame", "TextLabel"))

        self.setImg(0)

        # set slider
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(20)
        self.horizontalSlider.setValue(0)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider.setTickInterval(1)
        self.horizontalSlider.valueChanged.connect(self.valuechange)

    def setImg(self, n_img):

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = cv2.imread('test/9/T1/{}.jpg'.format(n_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc.axes.imshow(img, 'gray')
        self.verticalLayout_4.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = cv2.imread('test/9/T2/{}.jpg'.format(n_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc.axes.imshow(img, 'gray')
        self.verticalLayout_4.addWidget(sc)

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = cv2.imread('test/9/CT/{}.jpg'.format(n_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc.axes.imshow(img, 'gray')
        self.verticalLayout_5.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = cv2.imread('test/9/FT/{}.jpg'.format(n_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc.axes.imshow(img, 'gray')
        self.verticalLayout_5.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = cv2.imread('test/9/MN/{}.jpg'.format(n_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc.axes.imshow(img, 'gray')
        self.verticalLayout_5.addWidget(sc)

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = cv2.imread('test/9/CT/{}.jpg'.format(n_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc.axes.imshow(img, 'gray')
        self.verticalLayout_6.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = cv2.imread('test/9/FT/{}.jpg'.format(n_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc.axes.imshow(img, 'gray')
        self.verticalLayout_6.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = cv2.imread('test/9/MN/{}.jpg'.format(n_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc.axes.imshow(img, 'gray')
        self.verticalLayout_6.addWidget(sc)

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = cv2.imread('test/9/T1/{}.jpg'.format(n_img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sc.axes.imshow(img, 'gray')
        self.verticalLayout_7.addWidget(sc)

    def valuechange(self):
        s = self.horizontalSlider.value()
        print(s)
        # clear layout
        for i in reversed(range(self.verticalLayout_4.count())):
            self.verticalLayout_4.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.verticalLayout_5.count())):
            self.verticalLayout_5.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.verticalLayout_6.count())):
            self.verticalLayout_6.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.verticalLayout_7.count())):
            self.verticalLayout_7.itemAt(i).widget().setParent(None)

        # change image
        self.setImg(s)

import sys
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_Frame()

ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())