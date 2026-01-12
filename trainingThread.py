import os
import sys
import numpy as np
import tensorflow as tf
import skimage.io
import imgaug.augmenters as iaa
import skimage
import time
from PyQt5 import QtCore
import json
import cv2
import ray
from CustomCroppingDataset import CustomCroppingDataset
from CustomDataset import CustomDataset
from mrcnn import model as modellib, utils
import tensorflow.keras as keras
from solve_cudnn_error import *
from tensorflow.keras import backend as K
import signal
import logging

sys.setrecursionlimit(5000)
signal.signal(signal.SIGTERM, signal.SIG_DFL)

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Custom Callbacks for GUI ---

class LiveStatusCallback(tf.keras.callbacks.Callback):
    """
    Callback to emit live training status and loss data to the GUI
    instead of printing to the console.
    """
    def __init__(self, thread_instance):
        super().__init__()
        self.thread = thread_instance
        self.start_time = 0
        self.batch_count = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        self.batch_count = 0

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        elapsed = time.time() - self.start_time
        steps = self.params.get('steps', 0)
        
        # Calculate ETA
        if self.batch_count > 0 and steps > 0:
            time_per_batch = elapsed / self.batch_count
            remaining = steps - self.batch_count
            eta = int(remaining * time_per_batch)
        else:
            eta = 0
            
        # Format string like Keras: 
        # ETA: 26s - batch: 474 - loss: 0.6944 ...
        msg_parts = [f"ETA: {eta}s"]
        msg_parts.append(f"batch: {batch + 1}/{steps}")
        
        # Add metrics
        for k, v in logs.items():
            if k in ['batch', 'size']: continue
            msg_parts.append(f"{k}: {v:.4f}")
            
        final_msg = " - ".join(msg_parts)
        
        # Emit signal to update Status Bar text
        self.thread.update_status_bar.emit(final_msg)
        
        # Emit signal to update Graph (send just the loss value)
        if 'loss' in logs:
            self.thread.update_plot_data.emit(float(logs['loss']))

class MeanAveragePrecisionCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_model, inference_model, dataset,
                 calculate_map_at_every_X_epoch=1, dataset_limit=None,
                 verbose=1, thread_instance=None):
        super().__init__()
        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.calculate_map_at_every_X_epoch = calculate_map_at_every_X_epoch
        self.dataset_limit = len(self.dataset.image_ids)
        if dataset_limit is not None:
            self.dataset_limit = dataset_limit
        self.dataset_image_ids = self.dataset.image_ids.copy()
        self.thread = thread_instance 

        if inference_model.config.BATCH_SIZE != 1:
            raise ValueError("This callback only works with the bacth size of 1")

        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def on_epoch_end(self, epoch, logs=None):
        self._verbose_print("Calculating mAP...")
        self._load_weights_for_model()
        mAPs = self._calculate_mean_average_precision()
        mAP = np.mean(mAPs)
        if logs is not None:
            logs["mean_average_precision"] = mAP
            self._verbose_print("mAP at epoch {0} is: {1:.4f}".format(epoch+1, mAP))
        
        if self.thread and "steps" in self.params:
             self.thread.progressBar_setMaximum.emit(self.params['steps'])

        super().on_epoch_end(epoch, logs)

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self.inference_model.load_weights(last_weights_path, by_name=True)

    def _calculate_mean_average_precision(self):
        class_APs = {}
        overall_APs = []
        
        np.random.shuffle(self.dataset_image_ids)
        target_ids = self.dataset_image_ids[:self.dataset_limit]
        
        if self.thread:
            self.thread.progressBar_setMaximum.emit(len(target_ids))
            self.thread.progressBar.emit(0)

        for i, image_id in enumerate(target_ids):
            if self.thread:
                self.thread.progressBar.emit(i + 1)
            
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(self.dataset, self.inference_model.config, image_id)
            molded_images = np.expand_dims(modellib.mold_image(image, self.inference_model.config), 0)
            results = self.inference_model.detect(molded_images, verbose=0)
            r = results[0]
            for class_id in np.unique(gt_class_id):
                AP, _, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                                  r["class_ids"], r["scores"], r['masks'])
                class_APs[class_id] = AP
                overall_APs.append(AP)
        
        return overall_APs

def split_dataset(dataset, train_percentage, val_percentage, test_percentage):
    train_size = int(len(dataset.image_ids) * train_percentage)
    val_size = int(len(dataset.image_ids) * val_percentage)
    shuffled_image_ids = np.random.permutation(dataset.image_ids)

    train_set = CustomCroppingDataset()
    train_set.prepare()
    train_set.image_ids = shuffled_image_ids[:train_size]
    train_set.image_info = {id: dataset.image_info[id] for id in train_set.image_ids}
    train_set.class_info = dataset.class_info
    train_set.source_class_ids = dataset.source_class_ids
    train_set.num_classes = dataset.num_classes

    val_set = CustomCroppingDataset()
    val_set.prepare()
    val_set.image_ids = shuffled_image_ids[train_size:train_size + val_size]
    val_set.image_info = {id: dataset.image_info[id] for id in val_set.image_ids}
    val_set.class_info = dataset.class_info
    val_set.source_class_ids = dataset.source_class_ids
    val_set.num_classes = dataset.num_classes

    test_set = CustomCroppingDataset()
    test_set.prepare()
    test_set.image_ids = shuffled_image_ids[train_size + val_size:]
    test_set.image_info = {id: dataset.image_info[id] for id in test_set.image_ids}
    test_set.class_info = dataset.class_info
    test_set.source_class_ids = dataset.source_class_ids
    test_set.num_classes = dataset.num_classes
    return train_set, val_set, test_set


class trainingThread(QtCore.QThread):
    update_training_status = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)
    
    # NEW SIGNALS
    update_status_bar = QtCore.pyqtSignal(str)
    update_plot_data = QtCore.pyqtSignal(float)

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
    
    def run(self):
        from ray.util.joblib import register_ray
        register_ray()
        
        ray.init(ignore_reinit_error=True, logging_level=logging.ERROR, log_to_driver=False) 
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        solve_cudnn_error()
        self.update_training_status.emit("Training started!")
        
        ROOT_DIR = os.path.abspath(self.WORK_DIR)
        sys.path.append(ROOT_DIR)
        from mrcnn.config import Config
        from mrcnn import utils
        COCO_WEIGHTS_PATH = os.path.join(self.weight_path)
        
        class EvalInferenceConfig(Config):
            NAME = "cell"
            TEST_MODE = "inference"
            IMAGE_RESIZE_MODE = "pad64"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1 
            NUM_CLASSES = 1 + 2
            USE_MINI_MASK = False
            IMAGE_MAX_DIM = 4096
            IMAGE_MIN_DIM = 1024
        class CustomConfig(Config):
            NAME = "cell"
    
            # --- 1. 顯卡安全設定 ---
            # 如果你有 24GB VRAM 可以試試 2，不然建議維持 1
            IMAGES_PER_GPU = 1  
    
            # --- 2. 解析度與模型架構 ---
            # 這是正確的決定！保持 1024 可以看清楚細胞
            IMAGE_MIN_DIM = 1024  # 建議設 800，讓 resize 彈性一點，或維持 1024
            IMAGE_MAX_DIM = 1024
            IMAGE_RESIZE_MODE = "square" # 建議用 pad64，比 square 穩定，能保持長寬比
            BACKBONE = "resnet101"
    
            # --- 3. 關鍵修正：Ground Truth 數量 ---
            MAX_GT_INSTANCES = 50
            DETECTION_MAX_INSTANCES = 50 # 檢測時也允許抓出這麼多
    
            # --- 4. 針對小細胞的優化 ---
            # 你原本沒貼這行，但這對細胞極度重要！
            # 如果不加這個，模型預設最小只抓 32px 的物體，會漏掉小細胞
            RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128) 

            # --- 5. 其他 ---
            NUM_CLASSES = 1 + 2 
            USE_MINI_MASK = False # 關閉是好的，雖然吃記憶體但分割邊緣會更準
    
            # 這行建議在主程式計算，不要寫死，或者設為一個固定值如 100
            # STEPS_PER_EPOCH = 100 
            VALIDATION_STEPS = 10

        def train(model):
            # Define dataset loading (same as before)
            def dataset_progress_callback(current, total):
                self.progressBar_setMaximum.emit(total)
                self.progressBar.emit(current)

            split= True
            if split:
                dataset = CustomCroppingDataset()
                dataset.load_custom(self.dataset_path, "train", progress_callback=dataset_progress_callback)
                # dataset.load_custom(self.dataset_path, "val", progress_callback=dataset_progress_callback)
                # dataset.load_custom(self.dataset_path, "test", progress_callback=dataset_progress_callback)
                dataset.prepare()
                train_set, val_set, test_set = split_dataset(dataset, 0.07, 0.015, 0.015)
                dataset_train = train_set
                dataset_val = val_set
                dataset_test = test_set
            else:
                dataset_train = CustomCroppingDataset()
                dataset_train.load_custom(self.dataset_path,"train", progress_callback=dataset_progress_callback)
                dataset_train.prepare()
                dataset_val = CustomCroppingDataset()
                dataset_val.load_custom(self.dataset_path, "val", progress_callback=dataset_progress_callback)
                dataset_val.prepare()
                dataset_test = CustomDataset()
                dataset_test.load_custom(self.dataset_path, "test", progress_callback=dataset_progress_callback)
                dataset_test.prepare()

            aug = iaa.Sometimes(5/6, iaa.OneOf([
                        iaa.Fliplr(1),
                        iaa.Flipud(1),
                        iaa.Affine(rotate=(-45, 45)),
                        iaa.Affine(rotate=(-90, 90)),
                        iaa.Affine(scale=(0.5, 1.5)),
                        iaa.Fliplr(0.5),
                        iaa.Flipud(0.5),
                        iaa.Affine(rotate=(-10, 10)),
                        iaa.Affine(scale=(0.8, 1.2)),
                        iaa.Crop(percent=(0, 0.1)),
                        iaa.GaussianBlur(sigma=(0, 0.5)),
                        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
                        iaa.LinearContrast((0.5, 1.5)),
                        ]))

            model_inference = modellib.MaskRCNN(mode="inference", config=EvalInferenceConfig(),
                                     model_dir=self.WORK_DIR+"/logs")
            
            # Callbacks
            mean_average_precision_callback = MeanAveragePrecisionCallback(
                model, model_inference, dataset_test, 
                calculate_map_at_every_X_epoch=5, verbose=1, dataset_limit=100,
                thread_instance=self
            )
            # Use the new Live Status callback instead of the old Progress Bar
            live_status_callback = LiveStatusCallback(self)
	    
            self.update_training_status.emit("Fine tune heads layers")
            
            model.train(dataset_train, dataset_val,
                    epochs=300,
                    layers='heads',
                    custom_callbacks=[mean_average_precision_callback, live_status_callback],
                    # augmentation = aug,
                    verbose=0
                    )

        print("Loading Training Configuation...")
        self.update_training_status.emit("Dataset: "+self.dataset_path)
        self.update_training_status.emit("Logs: "+self.WORK_DIR+"/logs")
        if self.train_mode == "train":
            config = CustomConfig()
        config.display()
        
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=self.WORK_DIR+"/logs")
        print("Loading Inference Configuration...")
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
        self.update_training_status.emit("Loading weights "+str(weights_path))
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
        train(model)
