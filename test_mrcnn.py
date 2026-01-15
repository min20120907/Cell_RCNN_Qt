import os
import sys
import random
import math
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg') # 確保不依賴視窗介面
import matplotlib.pyplot as plt
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# ------------------------------------------------------------------
# 1. GPU & 環境檢查
# ------------------------------------------------------------------
print("="*50)
print("🔍 環境診斷開始 (Environment Diagnostics)")
print("="*50)
print(f"✅ TensorFlow Version: {tf.__version__}")
print(f"✅ Keras Version: {tf.keras.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detect: {len(gpus)} device(s) found.")
        print(f"   Name: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ WARNING: No GPU found. Running on CPU (will be slow).")

# ------------------------------------------------------------------
# 2. 定義簡單的 Shapes Config
# ------------------------------------------------------------------
class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset."""
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 15
    NUM_CLASSES = 1 + 3  # Background + square, circle, triangle
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 20  # 隨便設個小數字，跑得完就好
    VALIDATION_STEPS = 5

# ------------------------------------------------------------------
# 3. 定義形狀生成器 (不讀硬碟，直接記憶體生成)
# ------------------------------------------------------------------
class ShapesDataset(utils.Dataset):
    def load_shapes(self, count, height, width):
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def random_image(self, height, width):
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        shapes = []
        for _ in range(random.randint(1, 4)):
            shape = random.choice(["square", "circle", "triangle"])
            color = tuple([random.randint(0, 255) for _ in range(3)])
            dims = (random.randint(height//4, height//2), random.randint(height//4, height//2)) # buffer
            x, y = random.randint(0, width-1), random.randint(0, height-1)
            s = random.randint(20, 40) # size
            shapes.append((shape, color, (x, y, s)))
        return bg_color, shapes

    def image_reference(self, image_id):
        return ""

    def load_image(self, image_id):
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool_), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y-s),
                                (x-s/math.sin(math.radians(60)), y+s),
                                (x+s/math.sin(math.radians(60)), y+s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
        return image

# ------------------------------------------------------------------
# 4. 主執行邏輯
# ------------------------------------------------------------------
if __name__ == "__main__":
    # A. 準備數據
    print("\n🛠️ Generating Random Shapes Dataset...")
    dataset_train = ShapesDataset()
    dataset_train.load_shapes(100, 128, 128)
    dataset_train.prepare()
    print("✅ Dataset generated successfully.")

    # B. 建立模型 (Training)
    print("\n🧠 Initializing Mask R-CNN Model...")
    config = ShapesConfig()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=os.path.join(ROOT_DIR, "logs_debug"))
    
    # C. 開始訓練 (只跑 1 個 Epoch，驗證環境能不能跑)
    print("\n🚀 Starting Training (1 Epoch Test)...")
    try:
        model.train(dataset_train, dataset_train,
                    epochs=10,
                    layers='heads') # 只練 heads 比較快
        print("✅ Training finished without errors.")
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # D. 切換到 Inference 模式進行驗證
    print("\n🔎 Switching to Inference Mode...")
    class InferenceConfig(ShapesConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=os.path.join(ROOT_DIR, "logs_debug"))
    
    # 載入剛剛練好的權重
    model_path = model.find_last()
    print(f"⚖️ Loading weights from {model_path}")
    model.load_weights(model_path, by_name=True)
    
    # E. 隨機測試一張
    print("📸 Running prediction on a random image...")
    image_id = random.choice(dataset_train.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_train, inference_config, image_id)
    
    # 預測
    results = model.detect([image], verbose=1)
    r = results[0]
    
    # 檢查結果
    print(f"\n📊 Diagnostic Results:")
    print(f"   - GT Objects: {len(gt_class_id)}")
    print(f"   - Detected Objects: {len(r['class_ids'])}")
    print(f"   - ROIs: {r['rois'].shape}")
    
    # 繪圖並存檔
    save_path = "debug_result.png"
    
    # 簡單繪圖 (不依賴 mrcnn.visualize 的複雜功能，避免那邊報錯)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(image)
    # 畫框框
    for i in range(len(r['rois'])):
        y1, x1, y2, x2 = r['rois'][i]
        p = matplotlib.patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, alpha=0.7, linestyle="dashed", edgecolor="red", facecolor='none')
        ax[1].add_patch(p)
        ax[1].text(x1, y1, f"{r['scores'][i]:.2f}", color='white', backgroundcolor="red", fontsize=8)
    
    ax[1].set_title(f"Prediction (Found {len(r['rois'])})")
    ax[1].axis('off')
    
    plt.savefig(save_path)
    print(f"\n✅ Diagnostic Image saved to: {os.path.abspath(save_path)}")
    print("請打開這張圖片。如果你看到紅色的框框正確框住了圖形，代表你的環境是 100% 正常的！")
