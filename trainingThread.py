import os
import sys
import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa
import time
from PyQt5 import QtCore
import ray
import signal
import matplotlib
# 設定 Matplotlib 後端，防止與 PyQt 衝突
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 🔥 [修正] 補上 patches
import matplotlib.patches as patches
import cv2
from skimage import measure
from datetime import datetime

# Import MRCNN
from mrcnn import model as modellib, utils
from mrcnn.config import Config

# Import Custom Datasets
from CustomCroppingDataset import CustomCroppingDataset
from CustomDataset import CustomDataset

from solve_cudnn_error import *

# 防止遞迴深度錯誤
sys.setrecursionlimit(5000)
signal.signal(signal.SIGTERM, signal.SIG_DFL)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# =================================================================================
# Callbacks (座標系對齊 & 標準化 & Log優化版)
# =================================================================================

class TrainingVisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_model, inference_model, dataset, output_dir, thread_instance):
        super().__init__()
        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.output_dir = output_dir
        self.thread = thread_instance
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs=None):
        try:
            last_weights_path = self.train_model.find_last()
            self.inference_model.load_weights(last_weights_path, by_name=True)
            
            sample_ids = np.random.choice(self.dataset.image_ids, 3)
            
            for i, image_id in enumerate(sample_ids):
                # 🔥 [標準化] Evaluation 用 load_image_gt 確保座標系與 Resize/Padding 一致
                image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                    self.dataset, self.inference_model.config, image_id
                )
                
                # 直接用處理過的 image 進行偵測
                results = self.inference_model.detect([image], verbose=0)
                r = results[0]
                
                save_path = os.path.join(self.output_dir, f"epoch_{epoch+1}_sample_{i}.png")
                self._save_plot(image, gt_mask, r['masks'], r['rois'], save_path, epoch+1)
                
                # Notify GUI
                self.thread.update_gallery_signal.emit(save_path, i)
                
                # 🔥 [優化] Sanity Log 移入迴圈，每張 Sample 都印出來檢查
                if len(r['class_ids']) > 0:
                    print(f"[Epoch {epoch+1} Sample {i}] Preds: {len(r['class_ids'])} | Max Score: {np.max(r['scores']):.4f} | GTs: {len(gt_class_id)}")
                else:
                    print(f"[Epoch {epoch+1} Sample {i}] No predictions found (GTs: {len(gt_class_id)})")
                
        except Exception as e:
            print(f"Viz Callback Error: {e}")
            import traceback
            traceback.print_exc()

    def _save_plot(self, image, gt_mask, pred_mask, pred_rois, save_path, epoch):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.imshow(image)
        
        # 畫 Ground Truth (綠色輪廓)
        if gt_mask.shape[-1] > 0:
            for j in range(gt_mask.shape[-1]):
                if np.sum(gt_mask[..., j]) < 5: continue
                for contour in measure.find_contours(gt_mask[..., j], 0.5):
                    ax.plot(contour[:, 1], contour[:, 0], '-g', linewidth=1, alpha=0.5)
        
        # 畫 Prediction (紅色輪廓 + 紅色 BBox)
        n_inst = min(pred_mask.shape[-1], pred_rois.shape[0])
        
        if n_inst > 0:
            for j in range(n_inst):
                # Mask
                mask = pred_mask[..., j]
                if np.sum(mask) < 5: continue 
                for contour in measure.find_contours(mask, 0.5):
                    ax.plot(contour[:, 1], contour[:, 0], '-r', linewidth=1.5)
                
                # BBox
                y1, x1, y2, x2 = pred_rois[j]
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, alpha=0.7, linestyle="dashed", edgecolor='r', facecolor='none')
                ax.add_patch(p)

        ax.set_title(f"Epoch {epoch} Preview", fontsize=10)
        ax.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)

class LiveStatusCallback(tf.keras.callbacks.Callback):
    def __init__(self, thread_instance):
        super().__init__()
        self.thread = thread_instance
        self.start_time = 0
        self.batch_count = 0

    def on_train_begin(self, logs=None):
        steps = self.params.get('steps', 1000)
        self.thread.progressBar_setMaximum.emit(steps)
        self.thread.progressBar.emit(0)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        self.batch_count = 0
        self.thread.progressBar.emit(0)

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        elapsed = time.time() - self.start_time
        steps = self.params.get('steps', 0)
        
        self.thread.progressBar.emit(self.batch_count)

        if self.batch_count > 0 and steps > 0:
            time_per_batch = elapsed / self.batch_count
            remaining = steps - self.batch_count
            eta = int(remaining * time_per_batch)
        else:
            eta = 0
            
        msg_parts = [f"ETA: {eta}s"]
        msg_parts.append(f"batch: {batch + 1}/{steps}")
        
        loss_dict = {}
        for k, v in logs.items():
            if k in ['batch', 'size']: continue
            msg_parts.append(f"{k}: {v:.4f}")
            if 'loss' in k:
                loss_dict[k] = float(v)
            
        final_msg = " - ".join(msg_parts)
        self.thread.update_status_bar.emit(final_msg)
        
        if loss_dict:
            self.thread.update_plot_data.emit(loss_dict)

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
            raise ValueError("This callback only works with the batch size of 1")
        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.calculate_map_at_every_X_epoch != 0:
            return 

        self._verbose_print(f"Calculating mAP (Limit: {self.dataset_limit} images)...")
        self._load_weights_for_model()
        mAPs = self._calculate_mean_average_precision()
        mAP = np.mean(mAPs)
        
        if logs is not None:
            logs["mean_average_precision"] = mAP
            self._verbose_print("mAP at epoch {0} is: {1:.4f}".format(epoch+1, mAP))
        
        if self.thread and "steps" in self.params:
             self.thread.progressBar_setMaximum.emit(self.params['steps'])

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self.inference_model.load_weights(last_weights_path, by_name=True)

    def _calculate_mean_average_precision(self):
        overall_APs = []
        np.random.shuffle(self.dataset_image_ids)
        target_ids = self.dataset_image_ids[:self.dataset_limit]
        
        if self.thread:
            self.thread.progressBar_setMaximum.emit(len(target_ids))
            self.thread.progressBar.emit(0)

        for i, image_id in enumerate(target_ids):
            if self.thread:
                self.thread.progressBar.emit(i + 1)
            
            # 🔥 [標準化] mAP 也用 load_image_gt
            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                self.dataset, self.inference_model.config, image_id
            )
            
            results = self.inference_model.detect([image], verbose=0)
            r = results[0]
            
            AP, _, _, _ = utils.compute_ap(
                gt_bbox, gt_class_id, gt_mask, 
                r["rois"], r["class_ids"], r["scores"], r['masks']
            )
            overall_APs.append(AP)
            
        return overall_APs

# 🔥 [穩健] Split Dataset 重建邏輯
def split_dataset(dataset, train_percentage, val_percentage, test_percentage):
    all_ids = np.array(dataset.image_ids)
    np.random.shuffle(all_ids)

    n = len(all_ids)
    train_size = int(n * train_percentage)
    val_size = int(n * val_percentage)

    train_ids = all_ids[:train_size]
    val_ids   = all_ids[train_size:train_size + val_size]
    test_ids  = all_ids[train_size + val_size:]

    def make_subset(sub_ids):
        sub = CustomCroppingDataset()
        sub.class_info = list(dataset.class_info)
        sub.source_class_ids = dict(dataset.source_class_ids)
        sub.num_classes = dataset.num_classes

        # 注意：這裡使用 enumerate 重編號為 0..N-1 是必須的
        # 因為 MRCNN Dataset 用 List 存 info，如果用 old_id 當 index 會 out of range
        # 關鍵在於把 old info 內容完整 copy 過去
        for new_id, old_id in enumerate(sub_ids):
            info = dataset.image_info[old_id]
            src = info.get("source", "cell")
            
            sub.add_image(
                src,
                image_id=new_id,
                path=info["path"],
                width=info["width"],
                height=info["height"],
                polygons=info.get("polygons", []), # 確保 polygons 被帶過去
                num_ids=info.get("num_ids", []),
            )
        sub.prepare()
        return sub

    return make_subset(train_ids), make_subset(val_ids), make_subset(test_ids)

# =================================================================================
# Main Training Thread
# =================================================================================
class trainingThread(QtCore.QThread):
    update_training_status = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)
    update_status_bar = QtCore.pyqtSignal(str)
    update_plot_data = QtCore.pyqtSignal(object) 
    update_gallery_signal = QtCore.pyqtSignal(str, int) 

    def __init__(self, parent=None, test=0, epoches=100,
                 confidence=0.9, WORK_DIR='', weight_path='', dataset_path='', train_mode="train", steps=1000):
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
        ray.init(ignore_reinit_error=True, log_to_driver=True)
        solve_cudnn_error()
        self.update_training_status.emit("Training Initializing...")
        
        ROOT_DIR = os.path.abspath(self.WORK_DIR)
        VIZ_DIR = os.path.join(ROOT_DIR, "viz_logs")
        
        # 1. CustomConfig
        class CustomConfig(Config):
            NAME = "cell"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1 
            NUM_CLASSES = 1 + 2
            
            # 🔥 [Mean Pixel] 優先使用 COCO 標準值以利泛化
            # 若發現 Loss 很怪或收斂極慢，可改回 dataset 專屬值: np.array([85.6, 85.6, 85.6])
            MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
            
            RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
            BACKBONE_STRIDES = [4, 8, 16, 32, 64]
            RPN_ANCHOR_RATIOS = [0.75, 1, 1.33, 1.6]
            TRAIN_ROIS_PER_IMAGE = 512 
            RPN_TRAIN_ANCHORS_PER_IMAGE = 512
            PRE_NMS_LIMIT = 9000 
            MAX_GT_INSTANCES = 100
            DETECTION_MAX_INSTANCES = 100
            
            USE_MINI_MASK = False        
            MASK_POOL_SIZE = 14          
            
            DETECTION_MIN_CONFIDENCE = 0.5 
            RPN_NMS_THRESHOLD = 0.7
            
            IMAGE_MIN_DIM = 512
            IMAGE_MAX_DIM = 512
            IMAGE_RESIZE_MODE = "square"
            
            LOSS_WEIGHTS = {
                "rpn_class_loss": 1.0, "rpn_bbox_loss": 1.5,
                "mrcnn_class_loss": 1.0, "mrcnn_bbox_loss": 1.5, "mrcnn_mask_loss": 1.0
            }
            LEARNING_RATE = 0.001
            STEPS_PER_EPOCH = self.steps
            VALIDATION_STEPS = 50
            WEIGHT_DECAY = 0.0001

        class EvalInferenceConfig(CustomConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.5

        def dataset_progress_callback(current, total):
            self.progressBar_setMaximum.emit(total)
            self.progressBar.emit(current)

        # 2. Load Dataset
        dataset = CustomCroppingDataset()
        dataset.load_custom(self.dataset_path, "train", progress_callback=dataset_progress_callback)
        dataset.prepare()
        
        # Split Dataset
        train_set, val_set, test_set = split_dataset(dataset, 0.7, 0.15, 0.15)
        dataset_train = train_set
        dataset_val = val_set
        dataset_test = test_set

        self.update_training_status.emit("🔍 Running pre-training sanity check...")
        check_limit = min(2000, len(dataset_train.image_ids))
        dirty_count = 0
        for i in range(check_limit):
            image_id = dataset_train.image_ids[i]
            info = dataset_train.image_info[image_id]
            polygons = info.get('polygons', [])
            for p in polygons:
                xs, ys = p.get('all_points_x', []), p.get('all_points_y', [])
                if not xs: continue
                if (max(xs) - min(xs)) <= 1 or (max(ys) - min(ys)) <= 1:
                    dirty_count += 1
        
        if dirty_count > 0:
            msg = f"⚠️ WARNING: Found {dirty_count} invalid annotations! Proceeding anyway as requested..."
            print(msg)
            self.update_training_status.emit(msg)
        else:
            self.update_training_status.emit("✅ Sanity Check Passed.")

        aug_cell = iaa.Sequential([
            iaa.Fliplr(0.5), iaa.Flipud(0.5), 
            iaa.SomeOf((0, 2), [
                iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270),
                iaa.Affine(rotate=(-20, 20)), iaa.Affine(scale=(0.8, 1.2)),
                iaa.GaussianBlur(sigma=(0.0, 1.0)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)),
                iaa.LinearContrast((0.8, 1.2)),
            ])
        ])

        # 3. Model Init
        config = CustomConfig()
        config.display()
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=self.WORK_DIR+"/logs")
        
        weights_path = self.weight_path
        if not os.path.exists(weights_path): utils.download_trained_weights(weights_path)
        self.update_training_status.emit(f"Loading weights: {os.path.basename(weights_path)}")
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask", "rpn_model"])

        # 4. Callbacks Init
        model_inference = modellib.MaskRCNN(mode="inference", config=EvalInferenceConfig(), model_dir=self.WORK_DIR+"/logs")
        
        mean_average_precision_callback = MeanAveragePrecisionCallback(
            model, model_inference, dataset_test, 
            calculate_map_at_every_X_epoch=5, 
            verbose=1, dataset_limit=20, thread_instance=self
        )
        
        live_status_callback = LiveStatusCallback(self)
        
        viz_callback = TrainingVisualizationCallback(
            model, model_inference, dataset_val, VIZ_DIR, self
        )
        
        callbacks_list = [mean_average_precision_callback, live_status_callback, viz_callback]
        # -------------------------
        # Stage 1: Training Heads (Warm-up)
        # -------------------------
        self.update_training_status.emit("Stage 1: Training Heads (Warm-up)")

        model.config.LEARNING_RATE = 1e-3  # Stage1 通常可稍大
        model.train(
            dataset_train, dataset_val,
            epochs=80,                 # 會被 EarlyStopping 提前停
            layers='heads',
            augmentation=None,   # 建議 light aug（如果你沒有就先用 aug_cell）
            custom_callbacks=callbacks_list,
            verbose=0
        )

        # -------------------------
        # Stage 2: Fine-tune higher layers (4+)
        # -------------------------
        self.update_training_status.emit("Stage 2: Fine-tuning 4+ layers")

        model.config.LEARNING_RATE = 1e-4
        model.train(
            dataset_train, dataset_val,
            epochs=220,
            layers='4+',
            augmentation=aug_cell,      # 你原本那套 heavy aug
            custom_callbacks=callbacks_list,
            verbose=0
        )

        # -------------------------
        # Stage 3: Fine-tune all layers (Final polish)
        # -------------------------
        self.update_training_status.emit("Stage 3: Fine-tuning all layers")

        model.config.LEARNING_RATE = 1e-5
        model.train(
            dataset_train, dataset_val,
            epochs=400,                 # 你原本 400 的精神放這裡更合理
            layers='all',
            augmentation=aug_cell,      # 或者你可以再更「保守」一點的 aug
            custom_callbacks=callbacks_list,
            verbose=0
        )
