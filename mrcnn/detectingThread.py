import os
import sys
import glob
from zipfile import ZipFile
from datetime import datetime
import numpy as np
import cv2
import skimage
import tensorflow as tf
from PymageJ.roi import ROIEncoder, ROIPolygon
import ray
from mrcnn.config import Config
from PyQt5 import QtCore
from multiprocessing import cpu_count

import mrcnn.utils
import mrcnn.visualize
import mrcnn.visualize
import mrcnn.model as modellib
from mrcnn.model import log
import cell

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases

now = datetime.now()
formatted_date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

# 定義工作函數
@ray.remote
def process_image(DEVICE, MODEL_DIR, weights_path, config, ROI_PATH, DETECT_PATH, image_path, j, conf_rate, epoches, step):
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                      config=config)
    model.load_weights(weights_path, by_name=True)
    file_sum=0
    image = skimage.io.imread(image_path)
    try:
        image2 = cv2.cvtColor(image,cv2. COLOR_GRAY2RGB)
        results = model.detect([image2], verbose=0)
    except:
        results = model.detect([image], verbose=0)
    r = results[0]
    for a in range(len(r['masks'][0][0])):
        # self.append.emit(data.shape)
        # data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
        mask = (np.array(r['masks'][:, :, a]*255)).astype(np.uint8)
        contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        roi_count = 0
        for contour in contours:
            file_sum+=1
            x = [i[0][0] for i in contour]
            y = [i[0][1] for i in contour]
            roi_count += 1
            filename = '{:04d}-{:04d}-{:04d}.roi'.format(j+1, file_sum, roi_count)
            roi_obj = ROIPolygon(x, y)
            roi_class = r['class_ids'][a]
            with ROIEncoder(filename, roi_obj) as roi:
                roi.write()
            if(roi_class==1):
                with ZipFile(ROI_PATH+"/"+os.path.basename(DETECT_PATH)+"-[cell]-"+str(conf_rate)+"-"+str(epoches)+"-"+str(step)+f"[{formatted_date_time}]"+".zip", 'a') as myzip:
                    myzip.write(filename)
                    # append.emit("Compressed class " + str(roi_class) +" "+ filename)
            elif(roi_class==2):
                with ZipFile(ROI_PATH+"/"+os.path.basename(DETECT_PATH)+"-[chromosome]-"+str(conf_rate)+"-"+str(epoches)+"-"+str(step)+f"[{formatted_date_time}]"+".zip", 'a') as myzip:
                    myzip.write(filename)
                    # append.emit("Compressed class " + str(roi_class) +" "+ filename)
            elif(roi_class==3):
                with ZipFile(ROI_PATH+"/"+os.path.basename(DETECT_PATH)+"-[nuclear]-"+str(conf_rate)+"-"+str(epoches)+"-"+str(step)+f"[{formatted_date_time}]"+".zip", 'a') as myzip:
                    myzip.write(filename)
                    # append.emit("Compressed class " + str(roi_class) +" "+ filename)
            os.remove(filename)
    file_sum=0
# 建立 DetectingThread 類別
class detectingThread(QtCore.QThread):
    def __init__(self, parent=None, WORK_DIR='', txt='', weight_path='', confidence='0.9', dataset_path='', ROI_PATH='', DETECT_PATH='', DEVICE=':/gpu', conf_rate=0.9, epoches=10, step=100):
        super(detectingThread, self).__init__(parent)
        self.DETECT_PATH = DETECT_PATH
        self.WORK_DIR = WORK_DIR
        self.weight_path = weight_path
        self.dataset_path = dataset_path
        self.ROI_PATH = ROI_PATH
        self.txt = txt
        self.DEVICE = DEVICE
        self.conf_rate = conf_rate
        self.epoches = epoches
        self.step = step
        self.confidence = confidence
    append = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)

    def run(self):
        class CustomConfig(Config):
            """Configuration for training on the toy  dataset.
            Derives from the base Config class and overrides some values.
            """
            # Give the configuration a recognizable name
            NAME = "cell"
            # Skip detections with < 90% confidence
            DETECTION_MIN_CONFIDENCE = self.confidence
            # We use a GPU with 12GB memory, which can fit two images.
            # Adjust down if you use a smaller GPU.
            IMAGES_PER_GPU = 1
            GPU_COUNT = 1
            # Number of classes (including background)
            NUM_CLASSES = 1 + 3 # Background + cell + chromosome + nuclear

        config = CustomConfig()


        # config = InferenceConfig()
        config.display()


        TEST_MODE = "inference"
        # 初始化 Ray
        #WORK_DIR="/media/min20120907/Resources/Linux/MaskRCNN"
        ROOT_DIR = os.path.abspath(self.WORK_DIR)
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        #self.append.emit(ROOT_DIR)
        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
        # import training functions
        DEVICE =self.DEVICE
        # Or, load the last model you trained
        weights_path = self.weight_path
        # Load weights
        self.append.emit("Loading weights "+str(weights_path))
        # Create model in inference mode
        
        self.append.emit("Loaded weights!")
        # 其他程式碼...

        # 讀取圖像並處理
        filenames = glob.glob(self.DETECT_PATH + "/*" + self.txt)
        filenames.sort()

        # 使用 Ray Task 加速圖像讀取和處理
        for j, filename in enumerate(filenames):
            
            ray.get(process_image.remote(
                DEVICE, MODEL_DIR, weights_path, config, 
             self.ROI_PATH, self.DETECT_PATH,
                filename, j, self.conf_rate, 
                self.epoches, self.step))
            self.progressBar.emit(j)
        # 其他程式碼...

        # 關閉 Ray
        ray.shutdown()
        self.progressBar.emit(len(filenames))

