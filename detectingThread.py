import subprocess
import os
import struct
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
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
import csv
class detectingThread(QtCore.QThread):
    def __init__(self, parent=None, WORK_DIR = '',txt='', weight_path = '',dataset_path='',ROI_PATH='',DETECT_PATH='',DEVICE=':/gpu', conf_rate=0.9, epoches=10, step=100):
        super(detectingThread, self).__init__(parent)
        self.DETECT_PATH=DETECT_PATH
        self.WORK_DIR = WORK_DIR
        self.weight_path = weight_path
        self.dataset_path = dataset_path
        self.ROI_PATH=ROI_PATH
        self.txt = txt
        self.DEVICE=DEVICE
        self.conf_rate=conf_rate
        self.epoches=epoches
        self.step = step
    append = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)
    def run(self):
        #WORK_DIR="/media/min20120907/Resources/Linux/MaskRCNN"
        ROOT_DIR = os.path.abspath(self.WORK_DIR)
        #self.append.emit(ROOT_DIR)
        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        # import training functions
        
        import mrcnn.utils
        import mrcnn.visualize
        import mrcnn.visualize
        import mrcnn.model as modellib
        from mrcnn.model import log
        import cell
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        # Path to Ballon trained weights
        # You can download this file from the Releases page
        # https://github.com/matterport/Mask_RCNN/releases
        CELL_WEIGHTS_PATH = self.weight_path  # TODO: update this path
        
        DEVICE =self.DEVICE
        config = cell.CustomConfig()

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

        for f in glob.glob(self.DETECT_PATH+"/*"+self.txt):
            filenames.append(f)

        #bar = progressBar.progressBar(max_value=len(filenames))
        self.progressBar_setMaximum.emit(len(filenames))
        #filenames = sorted(filenames, key=lambda a : int(a.replace(self.format_txt.toPlainText(), "").replace("-", " ").split(" ")[6]))
        filenames.sort()
        file_sum=0
        self.append.emit(str(np.array(filenames)))
        RG_result = []
        for j in range(len(filenames)):
            self.progressBar.emit((j + 1) * 100 // len(filenames))
            image = skimage.io.imread(os.path.join(filenames[j]))
            # R_img = image[:,:,0]
            # G_img = image[:,:,1]
            # Run object detection
            try:
                image2 = cv2.cvtColor(image,cv2. COLOR_GRAY2RGB)
                results = model.detect([image2], verbose=0)
            except:
                results = model.detect([image], verbose=0)

            r = results[0]

            data = numpy.array(r['masks'], dtype=numpy.bool)
            # self.append.emit(data.shape)
            edges = []
            
            for a in range(len(r['masks'][0][0])):

                # self.append.emit(data.shape)
                # data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
                mask = (numpy.array(r['masks'][:, :, a]*255)).astype(numpy.uint8)
                img = Image.fromarray(mask, 'L')
                g = cv2.Canny(np.array(img),10,100)
                contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                self.progressBar.emit(j)
                #print(len(contours))
                # R_channel = 0
                # G_channel = 0
                
                # print(R_img.shape)
                # print(G_img.shape)
                # for iii in range(len(mask)):
                #    for jjj in range(len(mask[0])):
                #        R_channel += int(mask[iii][jjj]/255) * R_img[iii][jjj]
                #        G_channel += int(mask[iii][jjj]/255) * G_img[iii][jjj]
                # RG_result.append([R_channel,G_channel])
                # print([R_channel,G_channel])
                roi_count = 0
                for contour in contours:
                    file_sum+=1
                    x = [i[0][0] for i in contour]
                    y = [i[0][1] for i in contour]
                    roi_count += 1
                    filename = '{:04d}-{:04d}-{:04d}.roi'.format(j+1, file_sum, roi_count)
                    roi_obj = ROIPolygon(x, y)
                    with ROIEncoder(filename, roi_obj) as roi:
                        roi.write()
                    with ZipFile(self.ROI_PATH+"/"+os.path.basename(self.DETECT_PATH)+"-"+str(self.conf_rate)+"-"+str(self.epoches)+"-"+str(self.step)+".zip", 'a') as myzip:
                        myzip.write(filename)
                        self.append.emit("Compressed " + filename)
                    os.remove(filename)
                # with open(self.ROI_PATH+"/"+os.path.basename(self.DETECT_PATH)+"-"+str(self.conf_rate)+"-"+str(self.epoches)+"-"+str(self.step)+".csv", 'w', newline='') as csvfile:
                # 建立 CSV 檔寫入器
                #  writer = csv.writer(csvfile)
                #  writer.writerow(["Red","Green"])
                #  writer.writerows(RG_result)
