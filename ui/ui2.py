import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import cv2
import numpy as np

from model import inference

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
        self.label.setText(_translate("Frame", "dice"))
        self.label_2.setText(_translate("Frame", "0/0"))

        # set slider
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(20)
        self.horizontalSlider.setValue(0)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider.setTickInterval(1)
        self.horizontalSlider.valueChanged.connect(self.valuechange)

        # connect choose dataset
        self.pushButton.clicked.connect(self.choose_path)
        self.pushButton_2.clicked.connect(self.run_inference)

    def setImg(self, n_img):

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(self.xs[n_img][...,0], 'gray')
        self.verticalLayout_4.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(self.xs[n_img][...,1], 'gray')
        self.verticalLayout_4.addWidget(sc)

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(self.ys[n_img][...,0], 'gray')
        self.verticalLayout_5.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(self.ys[n_img][...,1], 'gray')
        self.verticalLayout_5.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(self.ys[n_img][...,2], 'gray')
        self.verticalLayout_5.addWidget(sc)

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(self.preds[n_img][...,0], 'gray')
        self.verticalLayout_6.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(self.preds[n_img][...,1], 'gray')
        self.verticalLayout_6.addWidget(sc)
        sc = MplCanvas(self, width=5, height=4, dpi=100)
        sc.axes.imshow(self.preds[n_img][...,2], 'gray')
        self.verticalLayout_6.addWidget(sc)

        sc = MplCanvas(self, width=5, height=4, dpi=100)
        img = self.stacks[n_img]
        sc.axes.imshow(img)
        self.verticalLayout_7.addWidget(sc)

        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("Frame", "Sequence DC(mean):\nMedian nerve: {}\nFlexor tendons: {}\nCarpal tunnel: {} \nimage DC:\nMedian nerve: {}\nFlexor tendons: {}\nCarpal tunnel: {}".format(self.mean_mn, self.mean_ft, self.mean_ct, self.mns[n_img], self.fts[n_img], self.cts[n_img])))

    def valuechange(self):
        s = self.horizontalSlider.value()
        # print(s)
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

        # change indexer
        self.label_2.setText('{}/{}'.format(s, len(self.cts)))

    def choose_path(self):
        file = str(QtWidgets.QFileDialog.getExistingDirectory(None, "Select Directory"))
        f = open('dataset_path.txt', 'w')
        f.write(file)
        f.close()

    def run_inference(self):
        self.xs, self.ys, self.preds, self.cts, self.fts, self.mns, self.stacks = inference()

        self.mean_ct = np.mean(np.array(self.cts))
        self.mean_ft = np.mean(np.array(self.fts))
        self.mean_mn = np.mean(np.array(self.mns))

        # change image
        self.setImg(0)

        # change indexer
        self.label_2.setText('0/{}'.format(len(self.cts)))
        self.horizontalSlider.setMaximum(len(self.cts) -1 )

import sys
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
ui = Ui_Frame()

ui.setupUi(MainWindow)
MainWindow.show()
sys.exit(app.exec_())