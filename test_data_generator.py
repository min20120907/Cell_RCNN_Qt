import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import ray

# 引用你的檔案
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from CustomCroppingDataset import CustomCroppingDataset

# =================設定區=================
DATASET_PATH = "/media/e814/DATA-500G/dataset-png"  # <--- 請改成你的 Dataset 路徑 (包含 train/val 資料夾的那層)
SUBSET = "test"                          # 測試 train 或 val
# =======================================

# 1. 簡易 Config (必須與 trainingThread 中的設定一致)
class DebugConfig(Config):
    NAME = "cell_debug"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1        # 測試時設為 1 比較好觀察
    NUM_CLASSES = 1 + 2       # 背景 + cell + chromosome
    
    # 這是你在 trainingThread 設定的參數
    IMAGE_MIN_DIM = 128       
    IMAGE_MAX_DIM = 128
    USE_MINI_MASK = False
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    # MEAN_PIXEL 必須與 trainingThread 一致，否則顏色會怪怪的
    MEAN_PIXEL = np.array([85.6, 85.6, 85.6]) 

def display_batch(images, masks, boxes, class_ids):
    """將 Generator 吐出來的資料畫成圖"""
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        # 1. 還原圖片 (Unmold): float32 -> uint8, 加回 Mean Pixel
        image = images[i] + np.array([85.6, 85.6, 85.6])
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # 2. 準備繪圖
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # --- 左圖：原始圖片 + Bounding Box ---
        ax[0].imshow(image)
        ax[0].set_title(f"Image {i} (Shape: {image.shape})")
        
        # 畫 Box
        H, W = image.shape[:2]
        n_boxes = 0
        for box_idx, box in enumerate(boxes[i]):
            y1, x1, y2, x2 = box
            # 過濾掉 Padding 的 0 數據
            if (y2 - y1) <= 0 or (x2 - x1) <= 0: 
                continue
            
            n_boxes += 1
            # 畫紅框
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax[0].add_patch(p)
            ax[0].text(x1, y1, f"ID: {class_ids[i][box_idx]}", color='yellow', fontsize=8, weight='bold')

        # --- 右圖：Mask疊加 ---
        # Mask 形狀是 [H, W, N_Instances]
        mask_layer = np.zeros((H, W), dtype=np.float32)
        
        # 把所有 Instance 的 Mask 疊在一起顯示
        current_masks = masks[i]
        if current_masks.shape[-1] > 0:
            for m_idx in range(current_masks.shape[-1]):
                # 只有當該 Instance 有對應的 Box 時才畫 (過濾 Padding)
                if np.sum(boxes[i][m_idx]) > 0:
                    mask_layer += current_masks[:, :, m_idx].astype(np.float32) * (m_idx + 1)
        
        ax[1].imshow(image)
        ax[1].imshow(mask_layer, alpha=0.5, cmap='jet') # 半透明疊加
        ax[1].set_title(f"Ground Truth Masks (Count: {n_boxes})")
        
        plt.show()

def test_generator():
    # 初始化 Ray (因為 CustomCroppingDataset 有用到)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    print("--- 1. Loading Dataset ---")
    config = DebugConfig()
    dataset = CustomCroppingDataset()
    dataset.load_custom(DATASET_PATH, SUBSET)
    dataset.prepare()
    print(f"Loaded {len(dataset.image_ids)} images.")

    print("--- 2. Creating DataGenerator ---")
    # 使用 model.py 中的 DataGenerator
    data_gen = modellib.DataGenerator(dataset, config, shuffle=True, augmentation=None)
    
    # 取得一個 Batch 的資料
    # __getitem__(0) 會回傳 (inputs, outputs)
    print("--- 3. Fetching first batch ---")
    inputs, _ = data_gen.__getitem__(0)
    
    # inputs 順序參考 model.py 的 DataGenerator:
    # [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, 
    #  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
    batch_images = inputs[0]
    batch_gt_class_ids = inputs[4]
    batch_gt_boxes = inputs[5]
    batch_gt_masks = inputs[6]
    
    print(f"Batch Image Shape: {batch_images.shape}")
    print(f"Batch Mask Shape: {batch_gt_masks.shape}")
    
    print("--- 4. Visualizing ---")
    display_batch(batch_images, batch_gt_masks, batch_gt_boxes, batch_gt_class_ids)

if __name__ == "__main__":
    test_generator()
