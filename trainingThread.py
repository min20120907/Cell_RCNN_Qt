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
import imgaug.augmenters as iaa
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
import tensorflow.keras as keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 在程序中使用 keras 模块

from solve_cudnn_error import *
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", message="Operation .* was changed by setting attribute after it was run by a session")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
class trainingThread(QtCore.QThread):
    def __init__(self, parent=None, test=0, epoches=100,
     confidence=0.9, WORK_DIR = '', weight_path = '',dataset_path='',train_mode="train",steps=1):
        super(trainingThread, self).__init__(parent)
        self.test = test
        self.epoches = epoches
        self.WORK_DIR = WORK_DIR
        self.weight_path = weight_path
        self.confidence = confidence
        self.dataset_path = dataset_path
        self.train_mode = train_mode
        self.steps = steps
    update_training_status = QtCore.pyqtSignal(str)

    def run(self):
        solve_cudnn_error()
        self.update_training_status.emit("Training started!")
        print("started input stream")
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
            IMAGES_PER_GPU = 4
#            GPU_COUNT = 2
            # Number of classes (including background)
            NUM_CLASSES = 1 + 1 # Background + cell

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
                # { 'filename': '28503151_5b5b7ec140_b.jpg',
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
                # self.update_training_status.emit(annotations1)
                annotations = list(annotations1.values())  # don't need the dict keys

                # The VIA tool saves images in the JSON even if they don't have any
                # annotations. Skip unannotated images.
                annotations = [a for a in annotations if a['regions']]

                # Add images
                for a in annotations:
                    # self.update_training_status.emit(a)
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
                # If not a balloon dataset image, delegate to parent class.
                image_info = self.image_info[image_id]
                if image_info["source"] != "cell":
                    return super(self.__class__, self).load_mask(image_id)

                # Convert polygons to a bitmap mask of shape
                # [height, width, instance_count]
                info = self.image_info[image_id]
                # print(info)
                mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                                dtype=np.uint8)
                for i, p in enumerate(info["polygons"]):
                    # Get indexes of pixels inside the polygon and set them to 1
                    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                    try:
                        mask[rr, cc, i] = 1
                    except IndexError:
                        print("Index Error")

                # Return mask, and array of class IDs of each instance. Since we have
                # one class ID only, we return an array of 1s
                return mask, np.ones([mask.shape[-1]], dtype=np.int32)

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
            self.update_training_status.emit("Training network heads")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=int(self.steps),
                        layers='heads',
                        augmentation = iaa.Sometimes(5/6, iaa.OneOf([
                        iaa.Fliplr(1),
                        iaa.Flipud(1),
                        iaa.Affine(rotate=(-45, 45)),
                        iaa.Affine(rotate=(-90, 90)),
                        iaa.Affine(scale=(0.5, 1.5))
                        ]))
                        )
            #gc.collect()
        '''
	augmentation = iaa.Sometimes(5/6, iaa.OneOf([
                        iaa.Fliplr(1),
                        iaa.Flipud(1),
                        iaa.Affine(rotate=(-45, 45)),
                        iaa.Affine(rotate=(-90, 90)),
                        iaa.Affine(scale=(0.5, 1.5))
                        ]))
        '''
        ############################################################
        #  Training
        ############################################################

        
        # Validate arguments
        self.update_training_status.emit("Dataset: "+self.dataset_path)
        self.update_training_status.emit("Logs: "+self.WORK_DIR+"/logs")
        # Configurations
        if self.train_mode == "train":
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
        if self.train_mode == "train":
            model = modellib.MaskRCNN(mode="training", config=config,
                                      model_dir=self.WORK_DIR+"/logs")
        else:
            model = modellib.MaskRCNN(mode="inference", config=config,
                                      model_dir=self.WORK_DIR+"/logs")
        weights_path = COCO_WEIGHTS_PATH
        # Download weights filet
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
        # Load weights
        self.update_training_status.emit("Loading weights "+str(weights_path))
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
        # Train or evaluate
        train(model)
        #while(True):
        #    time.sleep(2)
        #    self.update_training_status.emit('training' + str(self.test))
