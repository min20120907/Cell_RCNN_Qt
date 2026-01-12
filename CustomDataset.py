import os
import cv2
import numpy as np
import ray
import json
from mrcnn import utils

# 移除不必要的 imports (如 skimage, matplotlib, tqdm) 以保持乾淨

@ray.remote
def load_annotations(annotation, subset_dir, class_id):
    # 1. 穩健的 JSON 讀取
    try:
        data = json.load(open(os.path.join(subset_dir, annotation)))
    except Exception as e:
        print(f"Error loading json {annotation}: {e}")
        return []

    annotations = list(data.values())
    annotations = [a for a in annotations if a['regions']]

    images = []
    for a in annotations:
        if type(a['regions']) is dict:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]
        else:
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes'] for s in a['regions']]
        
        num_ids = []
        for _ in objects:
            num_ids.append(class_id)

        image_path = os.path.join(subset_dir, a['filename'])
        
        # 2. 優化：改用 cv2 讀取 (比 skimage 快且穩)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Cannot read {image_path}, skipping.")
            continue
            
        height, width = img.shape[:2]

        images.append({
            'image_id': a['filename'],
            'path': image_path,
            'width': width,
            'height': height,
            'polygons': polygons,
            'num_ids': num_ids
        })

    return images

class CustomDataset(utils.Dataset):
    @property
    def image_ids(self):
       return self._image_ids

    @image_ids.setter
    def image_ids(self, value):
       self._image_ids = value

    # --- 修正 1: 參數包含 progress_callback ---
    def load_custom(self, dataset_dir, subset, progress_callback=None):
        self.add_class("cell", 1, "cell")
        self.add_class("cell", 2, "chromosome")
        
        assert subset in ["train", "val", "test"]
        subset_dir = os.path.join(dataset_dir, subset)
        
        # 掃描 json
        annotations = [f for f in os.listdir(subset_dir) if f.startswith("via_region_") and f.endswith(".json")]
        
        # 建立 Ray 任務
        futures = [load_annotations.remote(a, subset_dir, 1) for a in annotations if "data_" in a] + \
                  [load_annotations.remote(a, subset_dir, 2) for a in annotations if "chromosome_" in a]
        
        print(f"Loading {len(futures)} annotation files via Ray...")
        
        # --- 修正 2: 處理 GUI 進度條 ---
        total_tasks = len(futures)
        remaining_futures = list(futures)
        
        while remaining_futures:
            # 每次等待部分任務完成
            done, remaining_futures = ray.wait(remaining_futures)
            
            # 更新 GUI
            if progress_callback:
                completed = total_tasks - len(remaining_futures)
                # 傳入 (目前完成數, 總數) 或是 (百分比)，視你的 GUI 設計而定
                try:
                    progress_callback(completed, total_tasks)
                except TypeError:
                    progress_callback(int(completed / total_tasks * 100))

        # 收集結果
        results = ray.get(futures)
        
        for images in results:
            for image in images:
                self.add_image(
                    'cell',
                    image_id=image['image_id'],
                    path=image['path'],
                    width=image['width'], height=image['height'],
                    polygons=image['polygons'],
                    num_ids=image['num_ids'])

    # --- 修正 3: 確保黑白圖轉 RGB (重要！) ---
    def load_image(self, image_id):
        info = self.image_info[image_id]
        path = info['path']
        image = cv2.imread(path)
        
        if image is None:
            # 為了防止訓練中斷，回傳一個全黑的圖 (或拋出錯誤)
            print(f"Error loading image: {path}")
            return np.zeros((info['height'], info['width'], 3), dtype=np.uint8)

        # 如果是黑白圖 (H, W) 或 (H, W, 1)，強制轉成 (H, W, 3)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # cv2 預設是 BGR，Mask R-CNN 需要 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        return image

    # --- 修正 4: 高速 Mask 生成 (取代 skimage) ---
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        
        for i, p in enumerate(info["polygons"]):
            # 取得座標並轉置
            pts = np.vstack((p['all_points_x'], p['all_points_y'])).astype(np.int32).T
            
            # 預先 Clip 防止越界 (比 try-except 快非常多)
            pts[:, 0] = np.clip(pts[:, 0], 0, info["width"] - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, info["height"] - 1)

            # 使用 OpenCV 填色 (1 代表前景)
            cv2.fillPoly(mask[:, :, i], [pts], color=1)

        return mask.astype(np.bool_), np.array(info['num_ids'], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]
