# -*- coding: utf-8 -*-
def launch_Selenium_Thread(self):
        t = threading.Thread(target=self.log)
        t.start()
#ImageJ tensorflow Python 3.8 Dependencies
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
import mlrose
import progressbar
import time
import logging
logging.getLogger('tensorflow').disabled = True
#PyQt5 Dependencies
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QListView, QFileDialog
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import pyqtSlot
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
import trainingThread
class Cell(QMainWindow, Ui_MainWindow):
    #Global Variables
    epoches = 100
    confidence = 0.9
    DEVICE = "/cpu:0"
    dataset_path = ""
    weight_path = ""
    WORK_DIR=""
    ROI_PATH=""
    DETECT_PATH=""
    coco_path = ""
    steps_num = 1
    #Json read
    def load_profile(self):
        with open("profile.json") as f:
            data = json.loads(f.read())
        f= data
        self.epoches = f['epoches']
        self.epochs.setText(str(f['epoches']))
        self.confidence = f['confidence']
        self.conf_rate.setText(str(f['confidence']))
        self.DEVICE = f['DEVICE']
        if(self.DEVICE=="/cpu:0"):
            self.cpu_train.toggle()
        elif(self.DEVICE=="/gpu:0"):
            self.gpu_train.toggle()
        self.dataset_path = f['dataset_path']
        self.WORK_DIR = f['WORK_DIR']
        self.ROI_PATH = f['ROI_PATH']
        self.DETECT_PATH = f['DETECT_PATH']
        self.coco_path = f['coco_path']
        self.weight_path = f['weight_path']
        self.steps_num = f['steps']
        self.steps.setText(str(f['steps']))
        self.append("Json profile loaded!")
    #Json write
    def save_profile(self):
    	tmp = dict()
    	tmp['epoches'] = int(self.epochs.toPlainText())
    	tmp['confidence']=float(self.conf_rate.toPlainText())
    	tmp['DEVICE'] = self.DEVICE
    	tmp['dataset_path'] = self.dataset_path
    	tmp['WORK_DIR'] = self.WORK_DIR
    	tmp['ROI_PATH'] = self.ROI_PATH
    	tmp['DETECT_PATH'] = self.DETECT_PATH
    	tmp['coco_path'] = self.coco_path
    	tmp['weight_path'] = self.weight_path
    	tmp['steps'] = self.steps.toPlainText()
    	with open('profile.json', 'w') as json_file:
            json.dump(tmp, json_file)
    	self.append("Json Profile saved!")
    def __init__(self, parent=None):

        super(Cell, self).__init__(parent)

        self.setupUi(self)
        #TextArea Events
        
        #Button Events
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
        ################################################
    def zip2coco(self):
        try:
            epoches = int(self.epochs.toPlainText())
            confidence = float(self.conf_rate.toPlainText())
            self.append(str(self.epoches))
            self.append(str(self.confidence))
            self.get_coco()
            os.chdir(self.coco_path)
            path ="."
            
            # ROI arrays
            filenames = []
            zips = []
            dirs =[]
            # scanning
            for d in os.walk(path):
                for dir in d:
                    for r,d,f in os.walk(str(dir)):
                        for file in f:
                            if self.format_txt.toPlainText() in file:
                                filenames.append(os.path.join(r, file))
                            elif ".zip" in file:
                                zips.append(os.path.join(r, file))
                    # Sorting
                    zips.sort()
                    filenames.sort()
                    # looping and decoding...
                    print(zips)
                    for j in range(len(zips)):
                        for i in range(len(filenames)):
                            # declare ROI file
                            roi = read_roi.read_roi_zip(zips[j])
                            roi_list = list(roi.values())

                            # ROI related file informations

                            filename = filenames[i].replace("./", "")
                            im = cv2.imread("./"+filename)
                            h, w, c = im.shape
                            size = os.path.getsize(filename)
                            try:
                                f = open("via_region_data.json")
                                original = json.loads(f.read())
                                print("Writing..."+str(zips[j]))
                                # Do something with the file
                            except FileNotFoundError:
                                print("File not exisited, creating new file...")
                                original = {}

                            data = {
                                filename
                                + str(size): {
                                    "fileref": "",
                                    "size": size,
                                    "filename": filename,
                                    "base64_img_data": "",
                                    "file_attributes": {},
                                    "regions": {},
                                }
                            }

                            # write json

                            length = len(list(roi.values()))
                            self.progressBar.setMaximum(length)
                            for a in range(length):
                                self.progressBar.setValue(a+1)
                                filename2 = filename.replace(self.format_txt.toPlainText(), "").replace("-", " ").split(" ")
                                roi_name = roi_list[a]["name"].replace("-", " ").split(" ")
                                filenum = ""

                                if int(filename2[-1]) > 10 and int(filename2[-1]) < 100:
                                    filenum = "00" + str(filename2[-1])
                                elif int(filename2[-1]) > 100 and int(filename2[-1]) < 1000:
                                    filenum = "0" + str(filename2[-1])
                                elif int(filename2[-1]) > 1 and int(filename2[-1]) < 10:
                                    filenum = "000" + str(filename2[-1])
                                elif int(filename2[-1]) > 1000 and int(filename2[-1]) < 10000:
                                    filenum = str(filename2[-1])

                                if filenum == roi_name[0]:
                                    print("roi_name: ", roi_name[0], "filename: ", filenum)
                                    x_list = roi_list[a]["x"]
                                    y_list = roi_list[a]["y"]
                                    for l in range(len(x_list)):
                                        if x_list[l] >= w:
                                            x_list[l] = w
                                        #  print(x_list[j])
                                    for k in range(len(y_list)):
                                        if y_list[k] >=h:
                                            y_list[k] = h
                                    #                print(y_list[k])
                                    # parameters

                                    x_list.append(roi_list[a]["x"][0])
                                    y_list.append(roi_list[a]["y"][0])
                                    regions = {
                                        str(a): {
                                            "shape_attributes": {
                                                "name": "polygon",
                                                "all_points_x": x_list,
                                                "all_points_y": y_list,
                                            },
                                            "region_attributes": {"name": dirname(dir).replace("-ROI", " ")+"-"+str(j)},
                                        }
                                    }
                                    data[filename + str(size)]["regions"].update(regions)
                                    original.update(data)
                            with io.open("via_region_data.json", "w", encoding="utf-8") as f:
                                f.write(json.dumps(original, ensure_ascii=False))
        except:
            self.append("Conversion failed partially!")
    
    def append(self, a):
        now = datetime.now()
        current_time = now.strftime("[%m-%d-%Y %H:%M:%S]")
        self.textBrowser.setText(self.textBrowser.toPlainText() + current_time + a + "\n")

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
        self.thread = trainingThread.trainingThread(test=1,steps=self.steps_num, train_mode=self.train_mode.toPlainText(), dataset_path=self.dataset_path,confidence=self.confidence,epoches=self.epoches, WORK_DIR=self.WORK_DIR, weight_path=self.weight_path)
        self.thread.update_training_status.connect(self.append)
        self.thread.moveToThread(self.myThread)
        self.myThread.started.connect(self.thread.run)
        self.myThread.start()

    def detect(self):
        #WORK_DIR="/media/min20120907/Resources/Linux/MaskRCNN"
        ROOT_DIR = os.path.abspath(self.WORK_DIR)
        #print(ROOT_DIR)
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
        print("Loading weights "+str(weights_path))
        model.load_weights(weights_path, by_name=True)
        print("loaded weights!")
        filenames = []

        for f in glob.glob(self.DETECT_PATH+"/*"+self.format_txt.toPlainText()):
            filenames.append(f)

        #bar = progressbar.ProgressBar(max_value=len(filenames))
        self.progressBar.setMaximum(len(filenames))
        #filenames = sorted(filenames, key=lambda a : int(a.replace(self.format_txt.toPlainText(), "").replace("-", " ").split(" ")[6]))
        filenames.sort()
        file_sum=0
        print(str(np.array(filenames)))
        for j in range(len(filenames)):
            self.progressBar.setValue(j)
            image = skimage.io.imread(os.path.join(filenames[j]))
            # Run object detection
            results = model.detect([image], verbose=0)

            r = results[0]

            data = numpy.array(r['masks'], dtype=numpy.bool)
            # print(data.shape)
            edges = []
            for a in range(len(r['masks'][0][0])):

                # print(data.shape)
                # data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
                mask = (numpy.array(r['masks'][:, :, a]*255)).astype(numpy.uint8)
                img = Image.fromarray(mask, 'L')
                g = cv2.Canny(np.array(img),10,100)
                contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                self.progressBar.setValue(j)
                for contour in contours:
                    file_sum+=1

                    x = [i[0][0] for i in contour]
                    y = [i[0][1] for i in contour]
                    if(len(x)>=100):
                        roi_obj = ROIPolygon(x, y)
                        with ROIEncoder(parseInt(j+1)+"-"+parseInt(file_sum)+"-0000"+".roi", roi_obj) as roi:
                            roi.write()
                        with ZipFile(self.ROI_PATH, 'a') as myzip:
                            myzip.write(parseInt(j+1)+"-"+parseInt(file_sum)+"-0000"+".roi")
                            print("Compressed "+parseInt(j+1)+"-"+parseInt(file_sum)+"-0000"+".roi")
                        os.remove(parseInt(j+1)+"-"+parseInt(file_sum)+"-0000"+".roi")
    def detect_anot(self):
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
        self.append("Loading weights "+str(weights_path))
        model.load_weights(weights_path, by_name=True)
        self.append("loaded weights!")
        filenames = []

        for f in glob.glob(self.DETECT_PATH+"/*"+self.format_txt.toPlainText()):
            filenames.append(f)

        #bar = progressbar.ProgressBar(max_value=len(filenames))
        self.progressBar.setMaximum(len(filenames))
        #filenames = sorted(filenames, key=lambda a : int(a.replace(self.format_txt.toPlainText(), "").replace("-", " ").split(" ")[6]))
        filenames.sort()
        file_sum=0
        self.append(str(np.array(filenames)))
        for j in range(len(filenames)):
            self.progressBar.setValue(j)
            image = skimage.io.imread(os.path.join(filenames[j]))
            # Run object detection
            results = model.detect([image], verbose=0)

            r = results[0]
            mrcnn.visualize.save_image(image, str(j)+"-anot"+self.format_txt.toPlainText(),r['rois'], r['masks'], r['class_ids'], r['scores'],r['class_names'], save_dir="anotated",mode=0)
    
    def detect_BW(self):
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
        self.append("Loading weights "+str(weights_path))
        model.load_weights(weights_path, by_name=True)
        self.append("loaded weights!")
        filenames = []

        for f in glob.glob(self.DETECT_PATH+"/*"+self.format_txt.toPlainText()):
            filenames.append(f)
        #bar = progressbar.ProgressBar(max_value=len(filenames))
        self.progressBar.setMaximum(len(filenames))
        #filenames = sorted(filenames, key=lambda a : int(a.replace(self.format_txt.toPlainText(), "").replace("-", " ").split(" ")[6]))
        filenames.sort()
        file_sum=0
        self.append(str(np.array(filenames)))
        for j in range(len(filenames)):
            self.progressBar.setValue(j)
            image = skimage.io.imread(os.path.join(filenames[j]))
            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]
            data = numpy.array(r['masks'], dtype=numpy.bool)
            # self.append(data.shape)
            edges = []
            for a in range(len(r['masks'][0][0])):
                mask = (numpy.array(r['masks'][:, :, a]*255)).astype(numpy.uint8)
                img = Image.fromarray(mask, 'L')
                img.save("1202-2017-BW/"+os.path.basename(filenames[j]).replace(self.format_txt.toPlainText(),"")+str(a)+self.format_txt.toPlainText())
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
        fileName_choose, filetype = QFileDialog.getSaveFileName(
            self, "Save ROIs zip file...", self.ROI_PATH, "ROIs Archives (*.zip)"
        )

        if fileName_choose == "":
            self.append("\nCanceled")
            return
        self.ROI_PATH = fileName_choose
######################################################

if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = Cell()
    window.show()
    sys.exit(app.exec_())
