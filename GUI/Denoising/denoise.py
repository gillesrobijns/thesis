# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(814, 508)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralWidget)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.picturebutton = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picturebutton.sizePolicy().hasHeightForWidth())
        self.picturebutton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.picturebutton.setFont(font)
        self.picturebutton.setObjectName("picturebutton")
        self.verticalLayout.addWidget(self.picturebutton)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.dAbutton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.dAbutton.setFont(font)
        self.dAbutton.setObjectName("dAbutton")
        self.verticalLayout_2.addWidget(self.dAbutton)
        self.dvaebutton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.dvaebutton.setFont(font)
        self.dvaebutton.setObjectName("dvaebutton")
        self.verticalLayout_2.addWidget(self.dvaebutton)
        self.bm3dbutton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.bm3dbutton.setFont(font)
        self.bm3dbutton.setObjectName("bm3dbutton")
        self.verticalLayout_2.addWidget(self.bm3dbutton)
        self.originalbutton = QtWidgets.QPushButton(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        self.originalbutton.setFont(font)
        self.originalbutton.setObjectName("originalbutton")
        self.verticalLayout_2.addWidget(self.originalbutton)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 3, 1, 2)
        self.imagelabel = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imagelabel.sizePolicy().hasHeightForWidth())
        self.imagelabel.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.imagelabel.setFont(font)
        self.imagelabel.setAlignment(QtCore.Qt.AlignCenter)
        self.imagelabel.setObjectName("imagelabel")
        self.gridLayout.addWidget(self.imagelabel, 0, 1, 2, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.cutoffslider = QtWidgets.QSlider(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.cutoffslider.setFont(font)
        self.cutoffslider.setMaximum(100)
        self.cutoffslider.setProperty("value", 30)
        self.cutoffslider.setOrientation(QtCore.Qt.Horizontal)
        self.cutoffslider.setObjectName("cutoffslider")
        self.gridLayout_2.addWidget(self.cutoffslider, 2, 1, 1, 1)
        self.sharpenbox = QtWidgets.QCheckBox(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.sharpenbox.setFont(font)
        self.sharpenbox.setObjectName("sharpenbox")
        self.gridLayout_2.addWidget(self.sharpenbox, 0, 3, 1, 1)
        self.amountslider = QtWidgets.QSlider(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.amountslider.setFont(font)
        self.amountslider.setMinimum(3)
        self.amountslider.setMaximum(18)
        self.amountslider.setProperty("value", 5)
        self.amountslider.setOrientation(QtCore.Qt.Horizontal)
        self.amountslider.setObjectName("amountslider")
        self.gridLayout_2.addWidget(self.amountslider, 1, 3, 1, 1)
        self.cutofflabel = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.cutofflabel.setFont(font)
        self.cutofflabel.setAlignment(QtCore.Qt.AlignCenter)
        self.cutofflabel.setObjectName("cutofflabel")
        self.gridLayout_2.addWidget(self.cutofflabel, 2, 0, 1, 1)
        self.amountlabel = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.amountlabel.setFont(font)
        self.amountlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.amountlabel.setObjectName("amountlabel")
        self.gridLayout_2.addWidget(self.amountlabel, 1, 2, 1, 1)
        self.gainlabel = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.gainlabel.setFont(font)
        self.gainlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.gainlabel.setObjectName("gainlabel")
        self.gridLayout_2.addWidget(self.gainlabel, 1, 0, 1, 1)
        self.gainslider = QtWidgets.QSlider(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.gainslider.setFont(font)
        self.gainslider.setMaximum(20)
        self.gainslider.setProperty("value", 10)
        self.gainslider.setOrientation(QtCore.Qt.Horizontal)
        self.gainslider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.gainslider.setTickInterval(0)
        self.gainslider.setObjectName("gainslider")
        self.gridLayout_2.addWidget(self.gainslider, 1, 1, 1, 1)
        self.sigmoid = QtWidgets.QCheckBox(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Open Sans")
        font.setPointSize(12)
        self.sigmoid.setFont(font)
        self.sigmoid.setObjectName("sigmoid")
        self.gridLayout_2.addWidget(self.sigmoid, 0, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 4, 0, 1, 5)
        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.picturebutton.setText(_translate("MainWindow", "Import Picture"))
        self.label.setText(_translate("MainWindow", "Denoising method"))
        self.dAbutton.setText(_translate("MainWindow", "dA"))
        self.dvaebutton.setText(_translate("MainWindow", "DVAE"))
        self.bm3dbutton.setText(_translate("MainWindow", "BM3D"))
        self.originalbutton.setText(_translate("MainWindow", "Original"))
        self.imagelabel.setText(_translate("MainWindow", "Import image to start"))
        self.sharpenbox.setText(_translate("MainWindow", "Sharpening"))
        self.cutofflabel.setText(_translate("MainWindow", "Cutoff"))
        self.amountlabel.setText(_translate("MainWindow", "Amount"))
        self.gainlabel.setText(_translate("MainWindow", "Gain"))
        self.sigmoid.setText(_translate("MainWindow", "Sigmoid Transform"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

