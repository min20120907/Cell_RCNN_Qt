import os
import sys
import numpy as np
import tensorflow as tf
import skimage.io
import imgaug.augmenters as iaa
import skimage
import time
#PyQt5 Dependencies
from PyQt5 import QtCore
#UI
#time
import json
import cv2
from tqdm import tqdm
import json
import ray
from CustomCroppingDataset import CustomCroppingDataset
from CustomDataset import CustomDataset
from mrcnn import model as modellib, utils
import tensorflow.keras as keras

from solve_cudnn_error import *
from multiprocessing import Pool, cpu_count
# from mrcnn.MeanAveragePrecisionCallback import MeanAveragePrecisionCallback
from tensorflow.keras import backend as K
import sys
sys.setrecursionlimit(5000)  # Set a higher recursion limit
# Configure the TensorFlow cluster
# cluster_spec = tf.train.ClusterSpec({
#     'worker': ['192.168.50.227:12345', '192.168.50.65:12345']
# })
# 
# # Create a server for the current node
# server = tf.distribute.Server(cluster_spec, job_name='worker', task_index=0)
# # Set number of intra-op and inter-op parallelism threads
# tf.config.threading.set_intra_op_parallelism_threads(32)
# tf.config.threading.set_inter_op_parallelism_threads(32)
# 
# # Allow GPU memory growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)
# 
# # Set Keras session to the current TensorFlow session
# tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto()))
import signal

signal.signal(signal.SIGTERM, signal.SIG_DFL)

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def generate_mask_subset(args):
    height, width, subset = args
    mask = np.zeros([height, width, len(subset)], dtype=np.uint8)
    for i, j in enumerate(range(subset[0], subset[1])):
        start = subset[j]['all_points'][:-1]
        rr, cc = skimage.draw.polygon(start[:, 1], start[:, 0])
        mask[rr, cc, i] = 1
    return mask



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
        # ray.init(
        # _system_config={
        #     "object_spilling_config": json.dumps(
        #         {
        #           "type": "filesystem",
        #           "params": {
        #             # Multiple directories can be specified to distribute
        #             # IO across multiple mounted physical devices.
        #             "directory_path": [
        #               "/mnt/800GB-DISK-1/ray/spill",
        #             ]
        #           },
        #         }
        #     ),
        # },
        # ignore_reinit_error=True, 
        # object_store_memory=2*1024**3,_memory=32*1024**3,
        # )
        from ray.util.joblib import register_ray

        register_ray()
        ray.init()
        # Get the physical devices and set memory growth for GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        solve_cudnn_error()
        self.update_training_status.emit("Training started!")
        print("started input stream")
        # Root directory of the project
        ROOT_DIR = os.path.abspath(self.WORK_DIR)

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        from mrcnn.config import Config
        from mrcnn import utils

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
            MAX_GT_INSTANCES = 1
            IMAGE_RESIZE_MODE = "square"
            # Give the configuration a recognizable name
            NAME = "cell"
            # We use a GPU with 12GB memory, which can fit two images.
            # Adjust down if you use a smaller GPU.
            IMAGES_PER_GPU = 10
            # IMAGE_CHANNEL_COUNT = 1
#            GPU_COUNT = 2
            USE_MINI_MASK = False
            # Number of classes (including background)
            NUM_CLASSES = 1 + 3 # Background + cell + chromosome
            # NUM_CLASSES = 1 + 1 # Background + cell
            # Number of training steps per epoch
            STEPS_PER_EPOCH = self.epoches
            IMAGE_MAX_DIM = 64
            IMAGE_MIN_DIM = 64
            # Backbone network architecture
            BACKBONE = "resnet101"

            # Number of validation steps per epoch
            VALIDATION_STEPS = 50


        ############################################################
        #  Dataset
        ############################################################

        def train(model):

            """Train the model."""
            # Training dataset.
            print("Loading training dataset")
            dataset_train = CustomCroppingDataset()
            dataset_train.load_custom(self.dataset_path,"train")

            dataset_train.prepare()
            print("Loading validation dataset")
            # Validation dataset
            dataset_val = CustomDataset()
            dataset_val.load_custom(self.dataset_path, "val")

            dataset_val.prepare()
            
            # *** This training schedule is an example. Update to your needs ***
            # Since we're using a very small dataset, and starting from
            # COCO trained weights, we don't need to train too long. Also,
            # no need to train all layers, just the heads should do it.
            aug = iaa.Sometimes(5/6, iaa.OneOf([
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
                        iaa.LinearContrast((0.5, 1.5)), # 對比度調整，調整因子為0.5到1.5
                        ]))
            # tf.compat.v1.enable_eager_execution()
            # add callback to calculate the result of accuracy
            
            # mean_average_precision_callback = MeanAveragePrecisionCallback(model,
            #   model_inference, dataset_val, calculate_map_at_every_X_epoch=1, verbose=1, dataset_limit=100)
            
            self.update_training_status.emit("Training network heads")
            
            model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=int(self.steps),
                    layers='heads',
                    # custom_callbacks=[mean_average_precision_callback],
                    augmentation = aug,
                    )
           
        ############################################################
        #  Training
        ############################################################

        
        # Validate arguments
        print("Loading Training Configuation...")
        self.update_training_status.emit("Dataset: "+self.dataset_path)
        self.update_training_status.emit("Logs: "+self.WORK_DIR+"/logs")
        # Configurations
        if self.train_mode == "train":
            config = CustomConfig()
        config.display()
        
 
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=self.WORK_DIR+"/logs")
        print("Loading Inference Configuration...")
        class InferenceConfig(CustomConfig):
            # Number of GPUs to use for inference
            GPU_COUNT = 1
            # Number of images to process on each GPU
            IMAGES_PER_GPU = 1
            USE_MINI_MASK = False
        model_inference = modellib.MaskRCNN(mode="inference", config=InferenceConfig(),
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
        train(model)
