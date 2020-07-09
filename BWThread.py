import os
import struct
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io
import codecs
from zipfile import ZipFile
from PymageJ.roi import ROIEncoder, ROIRect, ROIPolygon
import glob
import numpy
from PIL import Image
import skimage
from skimage import feature
import cv2
import progressbar
import time
import logging
logging.getLogger('tensorflow').disabled = True
#PyQt5 Dependencies
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QListView, QFileDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import pyqtSlot, QThread
#UI
from main_ui import Ui_MainWindow
#time
from datetime import datetime
import json
import read_roi
import io
from os.path import dirname
import json
import threading

class detectingThread(QtCore.QThread):
    def __init__(self, parent=None, WORK_DIR = '', weight_path = '',dataset_path='',ROI_PATH=''):
        super(detectingThread, self).__init__(parent)
        self.WORK_DIR = WORK_DIR
        self.weight_path = weight_path
        self.dataset_path = dataset_path
        self.ROI_PATH=ROI_PATH
    append = QtCore.pyqtSignal(str)
    progressbar = QtCore.pyqtSignal(int)
    def run(self):
        ROOT_DIR = os.path.abspath(self.WORK_DIR)
        print(ROOT_DIR)
        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        # import training functions
        
        import mrcnn.utils
        import mrcnn.visualize
        import mrcnn.visualize
        import mrcnn.model as modellib
        from mrcnn.model import log
        from samples.cell import cell
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        # Path to Ballon trained weights
        # You can download this file from the Releases page
        # https://github.com/matterport/Mask_RCNN/releases
        CELL_WEIGHTS_PATH = self.weight_path  # TODO: update this path
        
        DEVICE =self.DEVICE
        config = cell.CustomConfig()

        # Override the training configurations with a few
        # changes for inferencing.
        def parseInt(a):
            filenum=""
            if int(a) >= 100 and int(a) < 1000:
                filenum = "0" + str(a)
            elif int(a) >= 10 and int(a) < 100:
                filenum = "00" + str(a)
            elif int(a) >= 1 and int(a) < 10:
                filenum = "000" + str(a)
            elif int(a) >= 1000 and int(a) < 10000:
                 filenum = str(a)
            else:
                filenum="0000"
            return filenum
        class InferenceConfig(config.__class__):
            # Run detection on one image at a time
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
        config.display()

        # Device to load the neural network on.
        # Useful if you're training a model on the same
        # machine, in which case use CPU and leave the
        # GPU for training.

        # Inspect the model in training or inference modes
        # values: 'inference' or 'training'
        # TODO: code for 'training' test mode not ready yet
        TEST_MODE = "inference"

        # Create model in inference mode
        with tf.device(DEVICE):
            model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)

        # Or, load the last model you trained
        weights_path = self.weight_path

        # Load weights
        self.append.emit("Loading weights "+str(weights_path))
        model.load_weights(weights_path, by_name=True)
        self.append.emit("loaded weights!")
        filenames = []

        for f in glob.glob(self.DETECT_PATH+"/*"+self.format_txt.toPlainText()):
            filenames.append(f)
        #bar = progressbar.ProgressBar(max_value=len(filenames))
        self.progressBar.setMaximum(len(filenames))
        #filenames = sorted(filenames, key=lambda a : int(a.replace(self.format_txt.toPlainText(), "").replace("-", " ").split(" ")[6]))
        filenames.sort()
        file_sum=0
        self.append.emit(str(np.array(filenames)))
        for j in range(len(filenames)):
            self.progressBar.setValue(j)
            image = skimage.io.imread(os.path.join(filenames[j]))
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            data = numpy.array(r['masks'], dtype=numpy.bool)
            # self.append.emit(data.shape)
            edges = []
            for a in range(len(r['masks'][0][0])):
                mask = (numpy.array(r['masks'][:, :, a]*255)).astype(numpy.uint8)
                img = Image.fromarray(mask, 'L')
                img.save("1202-2017-BW/"+os.path.basename(filenames[j]).replace(self.format_txt.toPlainText(),"")+str(a)+self.format_txt.toPlainText())
