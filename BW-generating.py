
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
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from samples.cell import cell
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn.visualize import display_images
from mrcnn import visualize
from mrcnn import utils
# import training functions

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
CELL_WEIGHTS_PATH = "../../mask_rcnn_cell_0010.h5"  # TODO: update this path

config = cell.CustomConfig()
CELL_DIR = os.path.join(ROOT_DIR, "samples/cell/dataset")

# Override the training configurations with a few
# changes for inferencing.


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
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# Load validation dataset
dataset = cell.CustomDataset()
dataset.load_custom(CELL_DIR, "val")

# Must call before using the dataset
dataset.prepare()
#print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

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
for f in glob.glob(ROOT_DIR+"\\samples\\cell\\B10-S2\\1\\*.jpg"):
    filenames.append(f)
print(filenames)
for j in range(len(filenames)):
    image = skimage.io.imread(os.path.join(filenames[j]))
    # image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    #  modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    #info = dataset.image_info[image_id]
    # print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
    #                                      dataset.image_reference(image_id)))

    # Run object detection
    results = model.detect([image], verbose=0)

    # Display results
    ax = get_ax(1)
    r = results[0]
    # print(r['masks'].shape)
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],                            dataset.class_names, r['scores'], ax=ax, title="Predictions")
    # for a in range(len(r['rois'])):

    #w, h = len(r['masks'][0]),len(r['masks'])
    #print("weight=", w,"\theihgt: ",h,"\tfile: ",r['masks'])
    data = numpy.array(r['masks'], dtype=numpy.bool)
    # print(data.shape)
    edges = []
    for a in range(len(r['masks'][0][0])):

        # print(data.shape)
        # data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
        mask = (numpy.array(r['masks'][:, :, a]*255)).astype(numpy.uint8)
        img = Image.fromarray(mask, 'L')
        img.save("1202-2017-BW/"+os.path.basename(filenames[j]).replace(".jpg","")+str(a)+".jpg")