
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

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# import training functions
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from samples.cell import cell
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
CELL_WEIGHTS_PATH = "../../mask_rcnn_cell_0010.h5"  # TODO: update this path

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
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# Or, load the last model you trained
weights_path = "../../mask_rcnn_cell_0010.h5"

# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)
print("loaded weights!")
filenames = []

for f in glob.glob("/home/min20120907/Linux/low-res/*.png"):
    filenames.append(f)

bar = progressbar.ProgressBar(max_value=len(filenames))

#filenames = sorted(filenames, key=lambda a : int(a.replace(".jpg", "").replace("-", " ").split(" ")[6]))
filenames.sort()
file_sum=0
for j in range(len(filenames)):
    
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
        bar.update(j)
        for contour in contours:
            file_sum+=1
            
            x = [i[0][0] for i in contour]
            y = [i[0][1] for i in contour]
            if(len(x)>=100):
                roi_obj = ROIPolygon(x, y)
                with ROIEncoder(parseInt(j+1)+"-"+parseInt(file_sum)+"-0000"+".roi", roi_obj) as roi:
                    roi.write()
                with ZipFile("low-res-AI"+".zip", 'a') as myzip:
                    myzip.write(parseInt(j+1)+"-"+parseInt(file_sum)+"-0000"+".roi")
                    #print("Compressed "+parseInt(j+1)+"-"+parseInt(file_sum)+"-0000"+".roi")
                os.remove(parseInt(j+1)+"-"+parseInt(file_sum)+"-0000"+".roi")
        
