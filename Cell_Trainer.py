# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import batchDetectThread
import batch_cocoThread
import anotThread
import BWThread
import cocoThread
import detectingThread
import trainingThread
import threading
from os.path import dirname
import io
import read_roi
import json
from datetime import datetime
from main_ui import Ui_MainWindow
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMainWindow, QApplication, QListView, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import progressbar
import mlrose
import cv2
from skimage import feature
import skimage
from PIL import Image
import numpy
import glob
from PymageJ.roi import ROIEncoder, ROIRect, ROIPolygon
from zipfile import ZipFile
import codecs
import skimage.io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import time
import re
import math
import random
import sys
import struct
import os
import subprocess

class Cell(QMainWindow, Ui_MainWindow):
    # Global Variables
    epoches = 100
    confidence = 0.9
    DEVICE = "/cpu:0"
    dataset_path = ""
    weight_path = ""
    WORK_DIR = ""
    ROI_PATH = ""
    DETECT_PATH = ""
    coco_path = ""
    steps_num = 1
    # Json read

    def load_profile(self):
        with open("profile.json") as f:
            data = json.loads(f.read())
        f = data
        self.epoches = f['epoches']
        self.epochs.setText(str(f['epoches']))
        self.confidence = f['confidence']
        self.conf_rate.setText(str(f['confidence']))
        self.DEVICE = f['DEVICE']
        if(self.DEVICE == "/cpu:0"):
            self.cpu_train.toggle()
        elif(self.DEVICE == "/gpu:0"):
            self.gpu_train.toggle()
        self.dataset_path = f['dataset_path']
        self.WORK_DIR = f['WORK_DIR']
        self.ROI_PATH = f['ROI_PATH']
        self.DETECT_PATH = f['DETECT_PATH']
        self.coco_path = f['coco_path']
        self.weight_path = f['weight_path']
        self.steps_num = f['steps']
        self.format_txt.setText(f['txt'])
        self.steps.setText(str(f['steps']))
        self.append("Json profile loaded!")
    # Json write

    def save_profile(self):
        tmp = dict()
        tmp['epoches'] = int(self.epochs.toPlainText())
        tmp['confidence'] = float(self.conf_rate.toPlainText())
        tmp['DEVICE'] = self.DEVICE
        tmp['dataset_path'] = self.dataset_path
        tmp['WORK_DIR'] = self.WORK_DIR
        tmp['ROI_PATH'] = self.ROI_PATH
        tmp['DETECT_PATH'] = self.DETECT_PATH
        tmp['coco_path'] = self.coco_path
        tmp['weight_path'] = self.weight_path
        tmp['steps'] = self.steps.toPlainText()
        tmp['txt'] = self.format_txt.toPlainText()
        with open('profile.json', 'w') as json_file:
            json.dump(tmp, json_file)
        self.append("Json Profile saved!")

    def __init__(self, parent=None):

        super(Cell, self).__init__(parent)

        self.setupUi(self)
        # TextArea Events

        # Button Events
        self.train_btn.clicked.connect(self.train_t)
        self.detect_btn.clicked.connect(self.detect)

        self.gpu_train.clicked.connect(self.gpu_train_func)
        self.cpu_train.clicked.connect(self.cpu_train_func)
        self.clear_logs.clicked.connect(self.clear)
        self.upload_sets.clicked.connect(self.get_sets)
        self.upload_weight.clicked.connect(self.get_weight)
        self.upload_det.clicked.connect(self.get_detect)
        self.mrcnn_btn.clicked.connect(self.get_mrcnn)
        self.output_dir.clicked.connect(self.save_ROIs)
        self.roi_convert.clicked.connect(self.zip2coco)
        self.l_profile.clicked.connect(self.load_profile)
        self.s_profile.clicked.connect(self.save_profile)
        self.batch_coco.clicked.connect(self.cocoBatch)
        self.batch_detect.clicked.connect(self.detectBatch)
        ################################################

    def detectBatch(self):
        self.myThread = QtCore.QThread()
        self.thread = batchDetectThread.batchDetectThread(DETECT_PATH=self.DETECT_PATH, ROI_PATH=self.ROI_PATH, txt=self.format_txt.toPlainText(
        ), weight_path=self.weight_path, dataset_path=self.dataset_path, WORK_DIR=self.WORK_DIR,DEVICE=self.DEVICE,conf_rate=self.conf_rate.toPlainText(),
        epoches=self.epochs.toPlainText(),step=self.steps.toPlainText() )
        self.thread.append.connect(self.append)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.thread.progressBar.connect(self.progressBar.setValue)
        self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
        self.myThread.start()
        self.myThread.exit(0)
        self.thread.exit(0)

    def zip2coco(self):
        self.get_coco()
        self.myThread = QtCore.QThread()
        self.thread = cocoThread.cocoThread(
            coco_path=self.coco_path, txt=self.format_txt.toPlainText())
        self.thread.append_coco.connect(self.append)
        self.thread.progressBar.connect(self.progressBar.setValue)
        self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()
        self.myThread.exit(0)
        self.thread.exit(0)

    def cocoBatch(self):
        self.get_coco()
        self.myThread = QtCore.QThread()
        self.thread = batch_cocoThread.batch_cocoThread(
            coco_path=self.coco_path, txt=self.format_txt.toPlainText())
        self.thread.append_coco.connect(self.append)
        self.thread.progressBar.connect(self.progressBar.setValue)
        self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()
        self.myThread.exit(0)
        self.thread.exit(0)

    def append(self, a):
        now = datetime.now()
        current_time = now.strftime("[%m-%d-%Y %H:%M:%S]")
        self.textBrowser.setText(
            self.textBrowser.toPlainText() + current_time + a + "\n")

    def clear(self):
        self.textBrowser.clear()

    def gpu_train_func(self):
        self.append("Training in GPU...")
        self.DEVICE = "/gpu:0"

    def cpu_train_func(self):
        self.append("Training in CPU...")
        self.DEVICE = "/cpu:0"

    def train_t(self):
        self.epoches = int(self.epochs.toPlainText())
        self.confidence = float(self.conf_rate.toPlainText())
        self.myThread = QtCore.QThread()
        self.thread = trainingThread.trainingThread(test=1, steps=self.steps_num, train_mode=self.train_mode.toPlainText(
        ), dataset_path=self.dataset_path, confidence=self.confidence, epoches=self.epoches, WORK_DIR=self.WORK_DIR, weight_path=self.weight_path)
        self.thread.update_training_status.connect(self.append)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()
        self.myThread.exit(0)
        self.thread.exit(0)

    def detect(self):
        self.myThread = QtCore.QThread()
        self.thread = detectingThread.detectingThread(DETECT_PATH=self.DETECT_PATH, ROI_PATH=self.ROI_PATH, txt=self.format_txt.toPlainText(
        ), weight_path=self.weight_path, dataset_path=self.dataset_path, WORK_DIR=self.WORK_DIR,DEVICE=self.DEVICE,conf_rate=self.conf_rate.toPlainText(),
        epoches=self.epochs.toPlainText(),step=self.steps.toPlainText() )
        self.thread.append.connect(self.append)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.thread.progressBar.connect(self.progressBar.setValue)
        self.thread.progressBar_setMaximum.connect(self.progressBar.setMaximum)
        self.myThread.start()
        self.myThread.exit(0)
        self.thread.exit(0)
    #not opened
    def detect_anot(self):
        self.myThread = QtCore.QThread()
        self.thread = anotThread.anotThread()
        self.thread.append.connect(self.append)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()
        self.myThread.exit(0)
        self.thread.exit(0)
    #not opened
    def detect_BW(self):
        self.myThread = QtCore.QThread()
        self.thread = BWThread.BWThread()
        self.thread.append.connect(self.append)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()
        self.myThread.exit(0)
        self.thread.exit(0)
###############################################

    def get_sets(self):
        dir_choose = QFileDialog.getExistingDirectory(
            self, "Select an input directory...", self.dataset_path
        )
        if dir_choose == "":
            self.append("Cancel")
            return
        self.append("Selected:")
        self.append(dir_choose)
        self.dataset_path = dir_choose
################################################

    def get_output(self):
        dir_choose = QFileDialog.getExistingDirectory(
            self, "Select an output directory...", self.output_path
        )
        if dir_choose == "":
            self.append("Cancel")
            return
        self.append("Selected:")
        self.append(dir_choose)
        self.output_dir = dir_choose
###################################################

    def get_detect(self):
        dir_choose = QFileDialog.getExistingDirectory(
            self, "Select an detecting directory...", self.DETECT_PATH
        )
        if dir_choose == "":
            self.append("Cancel")
            return
        self.append("Selected:")
        self.DETECT_PATH = dir_choose
        self.append(self.DETECT_PATH)

##################################################
    def get_mrcnn(self):
        dir_choose = QFileDialog.getExistingDirectory(
            self, "Select an working directory...", self.WORK_DIR
        )
        if dir_choose == "":
            self.append("Cancel")
            return
        self.append("Selected:")
        self.append(dir_choose)
        self.WORK_DIR = dir_choose

    def get_coco(self):
        dir_choose = QFileDialog.getExistingDirectory(
            self, "Select an COCO directory...", self.coco_path
        )
        if dir_choose == "":
            self.append("Cancel")
            return
        self.append("Selected:")
        self.append(dir_choose)
        self.coco_path = dir_choose
#####################################################

    def get_weight(self):
        fileName_choose, filetype = QFileDialog.getOpenFileName(
            self, "Select Weight...", self.weight_path, " COCO Weight Files (*.h5)"
        )

        if fileName_choose == "":
            self.append("Cancel")
            return
        self.append("Selected Weight: "+str(fileName_choose))
        self.weight_path = fileName_choose
######################################################

    def save_ROIs(self):
        dir_choose = QFileDialog.getExistingDirectory(
            self, "Select an COCO directory...", self.ROI_PATH
        )
        if dir_choose == "":
            self.append("Cancel")
            return
        self.append("Selected:")
        self.append(dir_choose)
        self.ROI_PATH = dir_choose
######################################################


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = Cell()
    window.show()
    sys.exit(app.exec_())
