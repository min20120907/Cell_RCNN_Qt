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
    def __init__(self, parent=None):

        super(Cell, self).__init__(parent)

        self.setupUi(self)
        #TextArea Events
        self.epoches = int(self.epochs.toPlainText())
        self.confidence = float(self.conf_rate.toPlainText())
        #Button Events
        self.train_btn.clicked.connect(self.train)
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
        
        ################################################
    def zip2coco(self):
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
                        if ".png" in file:
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

                        filename = filenames[i].replace(".\\", "").replace(".\\","")
                        im = cv2.imread(".\\"+filename)
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
                            filename2 = filename.replace(".png", "").replace("-", " ").split(" ")
                            roi_name = roi_list[a]["name"].replace("-", " ").split(" ")
                            filenum = ""
                            index = 2
                            if int(filename2[index]) > 10 and int(filename2[index]) < 100:
                                filenum = "00" + str(filename2[index])
                            elif int(filename2[index]) > 100 and int(filename2[index]) < 1000:
                                filenum = "0" + str(filename2[index])
                            elif int(filename2[index]) > 1 and int(filename2[index]) < 10:
                                filenum = "000" + str(filename2[index])
                            elif int(filename2[index]) > 1000 and int(filename2[index]) < 10000:
                                filenum = str(filename2[index])

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

    def train(self):
        self.epoches = int(self.epochs.toPlainText())
        self.confidence = float(self.conf_rate.toPlainText())
        # Root directory of the project
        ROOT_DIR = os.path.abspath(self.WORK_DIR)
        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        from mrcnn.config import Config
        from mrcnn import model as modellib, utils

        # Path to trained weights file
        COCO_WEIGHTS_PATH = os.path.join(self.weight_path)

        # Directory to save logs and model checkpoints, if not provided
        # through the command line argument --logs
        DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

        ############################################################
        #  Configurations
        ############################################################


        class CustomConfig(Config):
            """Configuration for training on the toy  dataset.
            Derives from the base Config class and overrides some values.
            """
            # Give the configuration a recognizable name
            NAME = "cell"

            # We use a GPU with 12GB memory, which can fit two images.
            # Adjust down if you use a smaller GPU.
            IMAGES_PER_GPU = 1

            # Number of classes (including background)
            NUM_CLASSES = 1 + 1 # Background + toy

            # Number of training steps per epoch
            STEPS_PER_EPOCH = self.epoches

            # Skip detections with < 90% confidence
            DETECTION_MIN_CONFIDENCE = self.confidence


        ############################################################
        #  Dataset
        ############################################################

        class CustomDataset(utils.Dataset):
        
            def load_custom(self, dataset_dir, subset):
                """Load a subset of the bottle dataset.
                dataset_dir: Root directory of the dataset.
                subset: Subset to load: train or val
                """
                # Add classes. We have only one class to add.
                self.add_class("cell", 1, "cell")

                # Train or validation dataset?
                assert subset in ["train", "val"]
                dataset_dir = os.path.join(dataset_dir, subset)

                # Load annotations
                # VGG Image Annotator saves each image in the form:
                # { 'filename': '28503151_5b5b7ec140_b.png',
                #   'regions': {
                #       '0': {
                #           'region_attributes': {},
                #           'shape_attributes': {
                #               'all_points_x': [...],
                #               'all_points_y': [...],
                #               'name': 'polygon'}},
                #       ... more regions ...
                #   },
                #   'size': 100202
                # }
                # We mostly care about the x and y coordinates of each region
                annotations1 = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
                # print(annotations1)
                annotations = list(annotations1.values())  # don't need the dict keys

                # The VIA tool saves images in the JSON even if they don't have any
                # annotations. Skip unannotated images.
                annotations = [a for a in annotations if a['regions']]

                # Add images
                for a in annotations:
                    # print(a)
                    # Get the x, y coordinaets of points of the polygons that make up
                    # the outline of each object instance. There are stores in the
                    # shape_attributes (see json format above)
                    polygons = [r['shape_attributes'] for r in a['regions'].values()]

                    # load_mask() needs the image size to convert polygons to masks.
                    # Unfortunately, VIA doesn't include it in JSON, so we must read
                    # the image. This is only managable since the dataset is tiny.
                    image_path = os.path.join(dataset_dir, a['filename'])
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]

                    self.add_image(
                        "cell",  ## for a single class just add the name here
                        image_id=a['filename'],  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons)

            def load_mask(self, image_id):
                """Generate instance masks for an image.
               Returns:
                masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
                """
                # If not a bottle dataset image, delegate to parent class.
                image_info = self.image_info[image_id]
                if image_info["source"] != "cell":
                    return super(self.__class__, self).load_mask(image_id)

                # Convert polygons to a bitmap mask of shape
                # [height, width, instance_count]
                info = self.image_info[image_id]
                mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                                dtype=np.uint8)
                for i, p in enumerate(info["polygons"]):
                    # Get indexes of pixels inside the polygon and set them to 1
                    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                    mask[rr, cc, i] = 1

                # Return mask, and array of class IDs of each instance. Since we have
                # one class ID only, we return an array of 1s
                return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

            def image_reference(self, image_id):
                """Return the path of the image."""
                info = self.image_info[image_id]
                if info["source"] == "cell":
                    return info["path"]
                else:
                    super(self.__class__, self).image_reference(image_id)


        def train(model):
            """Train the model."""
            # Training dataset.
            dataset_train = CustomDataset()
            dataset_train.load_custom(self.dataset_path,"train")
            dataset_train.prepare()

            # Validation dataset
            dataset_val = CustomDataset()
            dataset_val.load_custom(self.dataset_path, "val")
            dataset_val.prepare()

            # *** This training schedule is an example. Update to your needs ***
            # Since we're using a very small dataset, and starting from
            # COCO trained weights, we don't need to train too long. Also,
            # no need to train all layers, just the heads should do it.
            self.append("Training network heads")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=self.epoches,
                        layers='heads')

        ############################################################
        #  Training
        ############################################################

        if __name__ == '__main__':
            # Validate arguments
            if self.train_mode.toPlainText() == "train":
                assert self.dataset_path, "Argument --dataset is required for training"

            self.append("Dataset: "+self.dataset_path)
            self.append("Logs: "+self.WORK_DIR+"/logs")

            # Configurations
            if self.train_mode.toPlainText() == "train":
                config = CustomConfig()
            else:
                class InferenceConfig(CustomConfig):
                    # Set batch size to 1 since we'll be running inference on
                    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                    GPU_COUNT = 1
                    IMAGES_PER_GPU = 1
                config = InferenceConfig()
            config.display()

            # Create model
            if self.train_mode.toPlainText() == "train":
                model = modellib.MaskRCNN(mode="training", config=config,
                                          model_dir=self.WORK_DIR+"/logs")
            else:
                model = modellib.MaskRCNN(mode="inference", config=config,
                                          model_dir=self.WORK_DIR+"/logs")

            weights_path = COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)

            # Load weights
            print("Loading weights ", weights_path)

            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
            # Train or evaluate
            train(model)
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

        for f in glob.glob(self.DETECT_PATH+"/*.png"):
            filenames.append(f)

        #bar = progressbar.ProgressBar(max_value=len(filenames))
        self.progressBar.setMaximum(len(filenames))
        #filenames = sorted(filenames, key=lambda a : int(a.replace(".png", "").replace("-", " ").split(" ")[6]))
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

                # self.append(data.shape)
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
                            self.append("Compressed "+parseInt(j+1)+"-"+parseInt(file_sum)+"-0000"+".roi")
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

        for f in glob.glob(self.DETECT_PATH+"/*.png"):
            filenames.append(f)

        #bar = progressbar.ProgressBar(max_value=len(filenames))
        self.progressBar.setMaximum(len(filenames))
        #filenames = sorted(filenames, key=lambda a : int(a.replace(".png", "").replace("-", " ").split(" ")[6]))
        filenames.sort()
        file_sum=0
        self.append(str(np.array(filenames)))
        for j in range(len(filenames)):
            self.progressBar.setValue(j)
            image = skimage.io.imread(os.path.join(filenames[j]))
            # Run object detection
            results = model.detect([image], verbose=0)

            r = results[0]
            mrcnn.visualize.save_image(image, str(j)+"-anot.png",r['rois'], r['masks'], r['class_ids'], r['scores'],r['class_names'], save_dir="anotated",mode=0)
    
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

        for f in glob.glob(self.DETECT_PATH+"/*.png"):
            filenames.append(f)
        #bar = progressbar.ProgressBar(max_value=len(filenames))
        self.progressBar.setMaximum(len(filenames))
        #filenames = sorted(filenames, key=lambda a : int(a.replace(".png", "").replace("-", " ").split(" ")[6]))
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
                img.save("1202-2017-BW/"+os.path.basename(filenames[j]).replace(".png","")+str(a)+".png")
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
