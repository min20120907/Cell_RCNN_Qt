import multiprocessing
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
from multiprocessing import Pool, cpu_count

def load_annotations(args):
    annotation, subset_dir, class_id = args
    # Load annotations from JSON file
    annotations = json.load(open(os.path.join(subset_dir, annotation)))
    annotations = list(annotations.values()) 
    annotations = [a for a in annotations if a['regions']]

    # Add images
    images = []
    for a in annotations:
        # Get the x, y coordinates of points of the polygons that make up
        # the outline of each object instance. These are stored in the
        # shape_attributes (see JSON format above)
        if type(a['regions']) is dict:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]
        else:
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes'] for s in a['regions']]
        num_ids = []
        for _ in objects:
            try:
                num_ids.append(class_id)
            except:
                pass
        # load_mask() needs the image size to convert polygons to masks.
        # Unfortunately, VIA doesn't include it in JSON, so we must read
        # the image. This is only manageable since the dataset is tiny.
        image_path = os.path.join(subset_dir, a['filename'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        images.append({
            'image_id': a['filename'],  # use file name as a unique image id
            'path': image_path,
            'width': width,
            'height': height,
            'polygons': polygons,
            'num_ids': num_ids
        })

    return images
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
            IMAGES_PER_GPU = 2
#            GPU_COUNT = 2
            # Number of classes (including background)
            NUM_CLASSES = 1 + 2 # Background + cell + chromosome
            # NUM_CLASSES = 1 + 1 # Background + cell
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
                self.add_class("cell", 2, "chromosome")
                self.add_class("cell", 3, "nuclear")
                # Train or validation dataset?
                assert subset in ["train", "val"]
                subset_dir = os.path.join(dataset_dir, subset)

                # Load annotations from all JSON files using multiprocessing
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                annotations1 = ['via_region_data.json']
                annotations2 = ['via_region_chromosome.json']
                annotations3 = ['via_region_nuclear.json']
                regions = [(a, subset_dir, 1) for a in annotations1] + [(a, subset_dir, 2) for a in annotations2] + [(a, subset_dir, 3) for a in annotations3]
                results = pool.map(load_annotations, regions)
                pool.close()
                pool.join()

                # Add images
                for images in results:
                    for image in images:
                        self.add_image(
                            'cell',
                            image_id=image['image_id'],  # use file name as a unique image id
                            path=image['path'],
                            width=image['width'], height=image['height'],
                            polygons=image['polygons'],
                            num_ids=image['num_ids'])
                            

            def load_mask(self, image_id):
                """Generate instance masks for an image.
               Returns:
                masks: A bool array of shape [height, width, instance count] with
                    one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks.
                """
                # If not a cell dataset image, delegate to parent class.
                image_info = self.image_info[image_id]

                # Convert polygons to a bitmap mask of shape
                # [height, width, instance_count]
                info = self.image_info[image_id]
                mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                                dtype=np.uint8)
                for i, p in enumerate(info["polygons"]):
                    # Get indexes of pixels inside the polygon and set them to 1
                    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                    # print(f"i={i}, rr={rr}, cc={cc}, len(cc)={len(cc)}")
                    try:
                        mask[rr, cc, i] = 1
                    except:
                        pass
                        # print("Error Occured")
                        # print(f"i={i}, rr={rr}, cc={cc}, len(cc)={len(cc)}")
                # Return mask, and array of class IDs of each instance. Since we have
                # one class ID only, we return an array of 1s
                return mask.astype(np.bool), np.array(info['num_ids'], dtype=np.int32)
            def image_reference(self, image_id):
                """Return the path of the image."""
                info = self.image_info[image_id]
                return info["path"]



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
                        iaa.Affine(scale=(0.5, 1.5)),
                        iaa.Fliplr(0.5), # 左右翻轉概率為0.5
                        iaa.Flipud(0.5), # 上下翻轉概率為0.5
                        iaa.Affine(rotate=(-10, 10)), # 隨機旋轉-10°到10°
                        iaa.Affine(scale=(0.8, 1.2)), # 隨機縮放80%-120%
                        iaa.Crop(percent=(0, 0.1)), # 隨機裁剪，裁剪比例為0%-10%
                        iaa.GaussianBlur(sigma=(0, 0.5)), # 高斯模糊，sigma值在0到0.5之間
                        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), # 添加高斯噪聲，噪聲標準差為0到0.05的像素值
                        iaa.ContrastNormalization((0.5, 1.5)), # 對比度調整，調整因子為0.5到1.5
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
