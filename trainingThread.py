import os
import sys
import numpy as np
import tensorflow as tf
import skimage.io
import imgaug.augmenters as iaa
import skimage
import time
import logging
logging.getLogger('tensorflow').disabled = True
#PyQt5 Dependencies
from PyQt5 import QtCore
#UI
#time
import json
from tqdm import tqdm
import json
import ray

import tensorflow.keras as keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from solve_cudnn_error import *
from multiprocessing import Pool, cpu_count
import psutil

ray.init(
    _system_config={
        "object_spilling_config": json.dumps(
            {
              "type": "filesystem",
              "params": {
                # Multiple directories can be specified to distribute
                # IO across multiple mounted physical devices.
                "directory_path": [
                  "/mnt/800GB-DISK-1/ray/spill",
                  "/mnt/8TB-DISK-4/tmp/ray/spill",
                ],
                "buffer_size": 1_000_000,
              },
            }
        ),
    },
    ignore_reinit_error=True, object_store_memory=2.5*1024**3,_memory=32*1024**3, num_cpus=None, num_gpus=1
)

def generate_mask_subset(args):
    height, width, subset = args
    mask = np.zeros([height, width, len(subset)], dtype=np.uint8)
    for i, j in enumerate(range(subset[0], subset[1])):
        start = subset[j]['all_points'][:-1]
        rr, cc = skimage.draw.polygon(start[:, 1], start[:, 0])
        mask[rr, cc, i] = 1
    return mask

@ray.remote
def load_annotations(annotation, subset_dir, class_id):
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
            MAX_GT_INSTANCES = 100
            # IMAGE_RESIZE_MODE = "none"
            # Give the configuration a recognizable name
            NAME = "cell"
            # We use a GPU with 12GB memory, which can fit two images.
            # Adjust down if you use a smaller GPU.
            IMAGES_PER_GPU = 4
            # IMAGE_CHANNEL_COUNT = 1
#            GPU_COUNT = 2
            # Number of classes (including background)
            NUM_CLASSES = 1 + 3 # Background + cell + chromosome
            # NUM_CLASSES = 1 + 1 # Background + cell
            # Number of training steps per epoch
            STEPS_PER_EPOCH = self.epoches

            # Backbone network architecture
            BACKBONE = "resnet50"

            # Number of validation steps per epoch
            VALIDATION_STEPS = 50


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

                def to_iterator(obj_ids):
                    while obj_ids:
                        done, obj_ids = ray.wait(obj_ids)
                        yield ray.get(done[0])
                
                # Load annotations from all JSON files using Ray multiprocessing
                annotations = [f for f in os.listdir(subset_dir) if f.startswith("via_region_") and f.endswith(".json")]
                futures = [load_annotations.remote(a, subset_dir, 1) for a in annotations if "data_" in a] + \
                          [load_annotations.remote(a, subset_dir, 2) for a in annotations if "chromosome_" in a] + \
                          [load_annotations.remote(a, subset_dir, 3) for a in annotations if "nuclear_" in a]
                # Showing the progressbar
                for _ in tqdm(to_iterator(futures), total=len(futures)):
                    pass
                results = ray.get(futures)
                

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
                        rr = np.clip(rr, 0, info["height"] - 1)  # Clip row indices to valid range
                        cc = np.clip(cc, 0, info["width"] - 1)   # Clip column indices to valid range
                        mask[rr, cc, i] = 1
                        # print("Error Occured")
                        # print(f"i={i}, rr={rr}, cc={cc}, len(cc)={len(cc)}")
                # Return mask, and array of class IDs of each instance. Since we have
                # one class ID only, we return an array of 1s
                return mask.astype(np.bool), np.array(info['num_ids'], dtype=np.int32)


            def image_reference(self, image_id):
                """Return the path of the image."""
                info = self.image_info[image_id]
                return info["path"]



        def train(model, model_inference):

            """Train the model."""
            # Training dataset.
            print("Loading training dataset")
            dataset_train = CustomDataset()
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
            
            # add callback to calculate the result of accuracy
            mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model,
               model_inference, dataset_val, calculate_map_at_every_X_epoch=1, verbose=1)
            
            self.update_training_status.emit("Training network heads")
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=int(self.steps),
                        layers='heads',
                        custom_callbacks=[mean_average_precision_callback],
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
        # Create model
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=self.WORK_DIR+"/logs")
        
        print("Loading Inference Configuration...")
        class InferenceConfig(CustomConfig):
            # Give the configuration a recognizable name
            NAME = "cell_inference"

            # Number of GPUs to use for inference
            GPU_COUNT = 1
            # Number of images to process on each GPU
            IMAGES_PER_GPU = 1

            # Backbone network architecture
            BACKBONE = "resnet101"

            # Size of image channels
            IMAGE_CHANNEL_COUNT = 3

            # Mean pixel values for image normalization
            MEAN_PIXEL = [123.7, 116.8, 103.9]

            # Resize mode for image resizing
            IMAGE_RESIZE_MODE = "square"

            # Maximum dimension of image
            IMAGE_MAX_DIM = 1024

            # Minimum dimension of image
            IMAGE_MIN_DIM = 768

            # Scale factor for image pyramid
            IMAGE_MIN_SCALE = 0

            # Shape of input image
            IMAGE_SHAPE = [1024, 1024, 3]

            # Size of image meta data
            IMAGE_META_SIZE = 16

            # Anchor scales for the Region Proposal Network (RPN)
            RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

            # Anchor ratios for the RPN
            RPN_ANCHOR_RATIOS = [0.5, 1, 2]

            # Anchor stride for the RPN
            RPN_ANCHOR_STRIDE = 1

            # Number of anchors used in training
            RPN_TRAIN_ANCHORS_PER_IMAGE = 256

            # Max number of ground truth instances to use in one image
            MAX_GT_INSTANCES = 100

            # Max number of detection instances to use in one image
            DETECTION_MAX_INSTANCES = 100

            # Minimum probability for a detection to be considered positive
            DETECTION_MIN_CONFIDENCE = 0.7

            # Non-maximum suppression threshold for detection
            DETECTION_NMS_THRESHOLD = 0.3

            # Size of mask pool
            MASK_POOL_SIZE = 14

            # Size of mask shape
            MASK_SHAPE = [28, 28]

            # Size of minimum mask shape
            MINI_MASK_SHAPE = (56, 56)

            # Positive fraction of ROI proposals used in training
            ROI_POSITIVE_RATIO = 0.33

            # Size of top-down pyramid in pixels
            TOP_DOWN_PYRAMID_SIZE = 256

            # Number of fully connected layers in the classification head
            FPN_CLASSIF_FC_LAYERS_SIZE = 1024

            # Size of pooling region in the mask head
            POOL_SIZE = 7

            # Whether to use batch normalization in training
            TRAIN_BN = False

            # Whether to use mini masks
            USE_MINI_MASK = True

            # Whether to use RPN ROIs
            USE_RPN_ROIS = True

            # Loss weights for different components
            LOSS_WEIGHTS = {
                "rpn_class_loss": 1.0,
                "rpn_bbox_loss": 1.0,
                "mrcnn_class_loss": 1.0,
                "mrcnn_bbox_loss": 1.0,
                "mrcnn_mask_loss": 1.0,
            }

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
        # Train or evaluate
        train(model, model_inference)
        #while(True):
        #    time.sleep(2)
        #    self.update_training_status.emit('training' + str(self.test))
