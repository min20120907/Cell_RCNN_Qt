# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MaskRCNN.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(937, 650)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.train_btn = QtWidgets.QPushButton(self.centralwidget)
        self.train_btn.setGeometry(QtCore.QRect(30, 240, 151, 51))
        self.train_btn.setObjectName("train_btn")
        self.conf_rate = QtWidgets.QTextEdit(self.centralwidget)
        self.conf_rate.setGeometry(QtCore.QRect(150, 40, 161, 31))
        self.conf_rate.setObjectName("conf_rate")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 40, 121, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(30, 80, 111, 31))
        self.label_2.setObjectName("label_2")
        self.epochs = QtWidgets.QTextEdit(self.centralwidget)
        self.epochs.setGeometry(QtCore.QRect(150, 80, 161, 31))
        self.epochs.setObjectName("epochs")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(30, 180, 871, 41))
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.upload_sets = QtWidgets.QPushButton(self.centralwidget)
        self.upload_sets.setGeometry(QtCore.QRect(340, 40, 131, 31))
        self.upload_sets.setObjectName("upload_sets")
        self.upload_weight = QtWidgets.QPushButton(self.centralwidget)
        self.upload_weight.setGeometry(QtCore.QRect(340, 80, 131, 31))
        self.upload_weight.setObjectName("upload_weight")
        self.upload_det = QtWidgets.QPushButton(self.centralwidget)
        self.upload_det.setGeometry(QtCore.QRect(480, 40, 171, 31))
        self.upload_det.setObjectName("upload_det")
        self.gpu_train = QtWidgets.QRadioButton(self.centralwidget)
        self.gpu_train.setGeometry(QtCore.QRect(820, 50, 115, 22))
        self.gpu_train.setObjectName("gpu_train")
        self.cpu_train = QtWidgets.QRadioButton(self.centralwidget)
        self.cpu_train.setGeometry(QtCore.QRect(820, 70, 115, 22))
        self.cpu_train.setChecked(True)
        self.cpu_train.setObjectName("cpu_train")
        self.roi_convert = QtWidgets.QPushButton(self.centralwidget)
        self.roi_convert.setGeometry(QtCore.QRect(660, 40, 151, 31))
        self.roi_convert.setObjectName("roi_convert")
        self.detect_btn = QtWidgets.QPushButton(self.centralwidget)
        self.detect_btn.setGeometry(QtCore.QRect(220, 240, 181, 51))
        self.detect_btn.setObjectName("detect_btn")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(30, 310, 881, 261))
        self.textBrowser.setObjectName("textBrowser")
        self.mrcnn_btn = QtWidgets.QPushButton(self.centralwidget)
        self.mrcnn_btn.setGeometry(QtCore.QRect(490, 80, 151, 31))
        self.mrcnn_btn.setObjectName("mrcnn_btn")
        self.clear_logs = QtWidgets.QPushButton(self.centralwidget)
        self.clear_logs.setGeometry(QtCore.QRect(30, 580, 89, 25))
        self.clear_logs.setObjectName("clear_logs")
        self.export_logs = QtWidgets.QPushButton(self.centralwidget)
        self.export_logs.setGeometry(QtCore.QRect(130, 580, 89, 25))
        self.export_logs.setObjectName("export_logs")
        self.anot_btn = QtWidgets.QPushButton(self.centralwidget)
        self.anot_btn.setGeometry(QtCore.QRect(450, 240, 181, 51))
        self.anot_btn.setObjectName("anot_btn")
        self.mask_btn = QtWidgets.QPushButton(self.centralwidget)
        self.mask_btn.setGeometry(QtCore.QRect(690, 240, 181, 51))
        self.mask_btn.setObjectName("mask_btn")
        self.output_dir = QtWidgets.QPushButton(self.centralwidget)
        self.output_dir.setGeometry(QtCore.QRect(660, 80, 151, 31))
        self.output_dir.setObjectName("output_dir")
        self.format_txt = QtWidgets.QTextEdit(self.centralwidget)
        self.format_txt.setGeometry(QtCore.QRect(400, 580, 161, 31))
        self.format_txt.setObjectName("format_txt")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(270, 580, 121, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(600, 580, 121, 31))
        self.label_4.setObjectName("label_4")
        self.train_mode = QtWidgets.QTextEdit(self.centralwidget)
        self.train_mode.setGeometry(QtCore.QRect(710, 580, 161, 31))
        self.train_mode.setObjectName("train_mode")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 130, 141, 31))
        self.label_5.setObjectName("label_5")
        self.steps = QtWidgets.QTextEdit(self.centralwidget)
        self.steps.setGeometry(QtCore.QRect(180, 130, 161, 31))
        self.steps.setObjectName("steps")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 937, 22))
        self.menubar.setObjectName("menubar")
        self.menuMaskRCNN_Trainer = QtWidgets.QMenu(self.menubar)
        self.menuMaskRCNN_Trainer.setObjectName("menuMaskRCNN_Trainer")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionUpload_weights = QtWidgets.QAction(MainWindow)
        self.actionUpload_weights.setObjectName("actionUpload_weights")
        self.actionUpload_dataset = QtWidgets.QAction(MainWindow)
        self.actionUpload_dataset.setObjectName("actionUpload_dataset")
        self.actionConvert_ImageJ_ROIs = QtWidgets.QAction(MainWindow)
        self.actionConvert_ImageJ_ROIs.setObjectName("actionConvert_ImageJ_ROIs")
        self.menuMaskRCNN_Trainer.addAction(self.actionUpload_weights)
        self.menuMaskRCNN_Trainer.addAction(self.actionUpload_dataset)
        self.menuMaskRCNN_Trainer.addAction(self.actionConvert_ImageJ_ROIs)
        self.menubar.addAction(self.menuMaskRCNN_Trainer.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "RCNN Cell Segmentation"))
        self.train_btn.setText(_translate("MainWindow", "Train it!"))
        self.conf_rate.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">0.9</span></p></body></html>"))
        self.label.setText(_translate("MainWindow", "Confidence Rate: "))
        self.label_2.setText(_translate("MainWindow", "Traning Epochs: "))
        self.epochs.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">100</span></p></body></html>"))
        self.upload_sets.setText(_translate("MainWindow", "Upload datasets"))
        self.upload_weight.setText(_translate("MainWindow", "Upload weights"))
        self.upload_det.setText(_translate("MainWindow", "Upload detection images"))
        self.gpu_train.setText(_translate("MainWindow", "GPU Training"))
        self.cpu_train.setText(_translate("MainWindow", "CPU Training"))
        self.roi_convert.setText(_translate("MainWindow", "Convert ImageJ ROIs"))
        self.detect_btn.setText(_translate("MainWindow", "Detect it!"))
        self.mrcnn_btn.setText(_translate("MainWindow", "MaskRCNN Directory"))
        self.clear_logs.setText(_translate("MainWindow", "Clear Logs"))
        self.export_logs.setText(_translate("MainWindow", "Export Logs"))
        self.anot_btn.setText(_translate("MainWindow", "Annotate it!"))
        self.mask_btn.setText(_translate("MainWindow", "Mask it!"))
        self.output_dir.setText(_translate("MainWindow", "Save ROIs"))
        self.format_txt.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">.jpg</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "Detect Format:"))
        self.label_4.setText(_translate("MainWindow", "Training Mode:"))
        self.train_mode.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">train</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "Traning Steps per Epoch: "))
        self.steps.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Noto Sans\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">10</span></p></body></html>"))
        self.menuMaskRCNN_Trainer.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionOpen.setText(_translate("MainWindow", "Open..."))
        self.actionUpload_weights.setText(_translate("MainWindow", "Upload weights"))
        self.actionUpload_dataset.setText(_translate("MainWindow", "Upload dataset"))
        self.actionConvert_ImageJ_ROIs.setText(_translate("MainWindow", "Convert ImageJ ROIs"))

