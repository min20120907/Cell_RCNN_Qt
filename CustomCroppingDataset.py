import os
import cv2
import numpy as np
import ray
import json
from mrcnn import utils

# =================================================================================
# 輔助函數區 (幾何運算與裁切邏輯)
# =================================================================================

def remove_file_extension(filename):
    """移除副檔名，支援多重圓點的情況"""
    return filename.rsplit('.', 1)[0] if '.' in filename else filename

def convert_to_polygon_dict(list_polygon):
    """將 list 轉換為字典格式 {'all_points_x': [], 'all_points_y': []}"""
    return {'all_points_x': [point[0] for point in list_polygon], 
            'all_points_y': [point[1] for point in list_polygon]}

def convert_to_polygons(data):
    """將字典格式轉換回 list [[x, y], ...]"""
    polygons = [[[x, y] for x, y in zip(d['all_points_x'], d['all_points_y'])] for d in data]
    return polygons

def are_polygons_close(polygon1, polygon2, threshold=512):
    """檢查兩個多邊形是否靠近 (用於分組)"""
    p1 = np.asarray(polygon1)
    p2 = np.asarray(polygon2)

    x1_min, x1_max = np.min(p1[:, 0]), np.max(p1[:, 0])
    y1_min, y1_max = np.min(p1[:, 1]), np.max(p1[:, 1])
    x2_min, x2_max = np.min(p2[:, 0]), np.max(p2[:, 0])
    y2_min, y2_max = np.min(p2[:, 1]), np.max(p2[:, 1])

    x_diff = np.min([np.abs(x1_min - x2_min), np.abs(x1_max - x2_max)])
    y_diff = np.min([np.abs(y1_min - y2_min), np.abs(y1_max - y2_max)])

    if x_diff < threshold and y_diff < threshold:
        return True
    return False

def group_polygons(polygons):
    """將靠近的多邊形分為同一組，以便一起裁切"""
    groups = []
    for polygon in polygons:
        polygon_added = False
        for group in groups:
            for polygon2 in group:
                if are_polygons_close(polygon, polygon2):
                    group.append(polygon)
                    polygon_added = True
                    break
            if polygon_added:
                break
        if not polygon_added:
            groups.append([polygon])
    return groups

def crop_by_polygon(image, polygon):
    """根據多邊形裁切圖片，並 Padding 成正方形 (注意：這裡使用 Pad64 邏輯可能會導致尺寸不一)"""
    polygon = np.array(polygon, dtype=np.int32)

    # 建立 Mask 並裁切
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    x, y, w, h = cv2.boundingRect(polygon)
    cropped_image = masked_image[y:y+h, x:x+w]

    # Padding 成正方形 (補黑邊)
    dst_size = max(cropped_image.shape)
    bottom = dst_size - cropped_image.shape[0]
    right = dst_size - cropped_image.shape[1]
    padded_image = cv2.copyMakeBorder(cropped_image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0,0,0])

    return padded_image, polygon

def crop_by_group(image, group, margin=20):
    """裁切一整組多邊形，並更新相對座標 (修正版：置中 + Margin)"""
    # 1. 找出該組的邊界框 (Bounding Box)
    all_points_x = [p[0] for polygon in group for p in polygon]
    all_points_y = [p[1] for polygon in group for p in polygon]
    
    min_x, max_x = np.min(all_points_x), np.max(all_points_x)
    min_y, max_y = np.min(all_points_y), np.max(all_points_y)

    # 2. 加入 Margin (防止貼邊，並確保不超出圖片範圍)
    img_h, img_w = image.shape[:2]
    x1 = int(max(0, min_x - margin))
    y1 = int(max(0, min_y - margin))
    x2 = int(min(img_w, max_x + margin))
    y2 = int(min(img_h, max_y + margin))
    
    # 3. 裁切 (這裡不再做 bitwise_and 遮罩，保留背景 Context)
    cropped_image = image[y1:y2, x1:x2]
    
    # 防呆：如果切出來是空的
    if cropped_image.size == 0: 
        return None, None
    
    # 4. Padding 成正方形 (改為置中對齊)
    h_crop, w_crop = cropped_image.shape[:2]
    target_dim = max(h_crop, w_crop)
    
    delta_w = target_dim - w_crop
    delta_h = target_dim - h_crop
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    
    # 補黑邊
    padded_image = cv2.copyMakeBorder(
        cropped_image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=[0,0,0]
    )

    # 5. 更新多邊形座標
    # 新座標 = 原始座標 - 裁切起點 (x1, y1) + Padding 偏移 (left, top)
    updated_polygons = []
    for polygon in group:
        updated_polygon = []
        for point in polygon:
            new_x = point[0] - x1 + left
            new_y = point[1] - y1 + top
            updated_polygon.append([new_x, new_y])
        updated_polygons.append(convert_to_polygon_dict(updated_polygon))
        
    return padded_image, updated_polygons

# =================================================================================
# Ray Worker: 負責並行讀取、裁切與寫入 SSD
# =================================================================================
@ray.remote
def load_annotations(annotation, subset_dir, class_id, cache_dir):
    TARGET_DIM = 512 

    try:
        data = json.load(open(os.path.join(subset_dir, annotation)))
    except Exception as e:
        print(f"Error loading json {annotation}: {e}")
        return []

    annotations = list(data.values())
    annotations = [a for a in annotations if a['regions']]

    images = []
    
    if cache_dir is None:
        cache_dir = os.path.join(subset_dir, "temp_crops")
    os.makedirs(cache_dir, exist_ok=True)

    for a in annotations:
        if type(a['regions']) is dict:
            raw_polygons = [r['shape_attributes'] for r in a['regions'].values()]
        else:
            raw_polygons = [r['shape_attributes'] for r in a['regions']]
        
        # --- [新增過濾邏輯] 排除寬高過小的標註 ---
        valid_polygons_list = []
        for p in raw_polygons:
            # 取得多邊形的所有 X 和 Y
            all_x = p.get('all_points_x', [])
            all_y = p.get('all_points_y', [])
            
            if len(all_x) < 3: # 至少要三個點才能構成面
                continue
                
            w = max(all_x) - min(all_x)
            h = max(all_y) - min(all_y)
            
            # 只保留寬高都大於 1 像素的標註
            if w > 1 and h > 1:
                valid_polygons_list.append(p)
        
        # 如果這張圖過濾後一個標註都沒有，就跳過
        if not valid_polygons_list:
            continue
        # ---------------------------------------

        image_path = os.path.join(subset_dir, a['filename'])
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        # 使用過濾後的 valid_polygons_list 進行分組
        groups = group_polygons(convert_to_polygons(valid_polygons_list))
        polygon_id = 0
        
        for group in groups:
            cropped_image, updated_polygons = crop_by_group(image, group)
            
            if cropped_image is not None and cropped_image.size > 0:
                h, w = cropped_image.shape[:2]
                
                if h != TARGET_DIM or w != TARGET_DIM:
                    scale = TARGET_DIM / max(h, w)
                    cropped_image = cv2.resize(cropped_image, (TARGET_DIM, TARGET_DIM))
                    
                    new_polygons = []
                    for poly in updated_polygons:
                        new_x = [x * scale for x in poly['all_points_x']]
                        new_y = [y * scale for y in poly['all_points_y']]
                        new_polygons.append({'all_points_x': new_x, 'all_points_y': new_y})
                    updated_polygons = new_polygons

                filename_no_ext = remove_file_extension(a['filename'])
                sub_dir = os.path.join(cache_dir, filename_no_ext)
                os.makedirs(sub_dir, exist_ok=True)

                output_filename = os.path.join(sub_dir, f"{polygon_id}.png")
                polygon_id += 1
                
                try:
                    cv2.imwrite(output_filename, cropped_image)
                except Exception as e:
                    print(f"Failed to write crop: {e}")
                    continue

                images.append({
                    'image_id': output_filename,
                    'path': output_filename,
                    'width': cropped_image.shape[1],
                    'height': cropped_image.shape[0],
                    'polygons': updated_polygons,
                    'num_ids': [class_id] * len(updated_polygons)
                })
                
    return images
# =================================================================================
# 主 Dataset 類別
# =================================================================================

class CustomCroppingDataset(utils.Dataset):
    
    def load_image(self, image_id):
        """讀取圖片並確保轉為 RGB 格式 (修正黑白圖問題)"""
        info = self.image_info[image_id]
        path = info['path']
        image = cv2.imread(path)
        
        if image is None:
            print(f"Error reading image: {path}")
            # 回傳全黑圖防止崩潰
            return np.zeros((info['height'], info['width'], 3), dtype=np.uint8)

        # 處理通道: (H, W) -> (H, W, 3)
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def convert_numpy_to_python(self, obj):
        """JSON 序列化輔助函數"""
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(item) for item in obj]
        else:
            return obj

    @property
    def image_ids(self):
       return self._image_ids

    @image_ids.setter
    def image_ids(self, value):
       self._image_ids = value

    def load_custom(self, dataset_dir, subset, progress_callback=None, cache_dir=None):
        """
        Args:
            cache_dir: 指定 SSD 的路徑，用於存放裁切後的暫存圖
        """
        # 定義類別 ID
        self.add_class("cell", 1, "cell")
        self.add_class("cell", 2, "chromosome")
        # self.add_class("cell", 3, "nuclear")
        
        assert subset in ["train", "val", "test"]
        subset_dir = os.path.join(dataset_dir, subset)

        # 掃描標註檔
        annotations = [f for f in os.listdir(subset_dir) if f.startswith("via_region_") and f.endswith(".json")]
        
        # 啟動 Ray Workers
        futures = [load_annotations.remote(a, subset_dir, 1, cache_dir) for a in annotations if "data_" in a] + \
                  [load_annotations.remote(a, subset_dir, 2, cache_dir) for a in annotations if "chromosome_" in a]
        
        print(f"Loading {len(futures)} annotation files via Ray...")
        if cache_dir:
            print(f"Temp crops path: {cache_dir}")

        # 處理進度條
        total_tasks = len(futures)
        remaining_futures = list(futures)
        
        while remaining_futures:
            done, remaining_futures = ray.wait(remaining_futures)
            
            if progress_callback:
                completed = total_tasks - len(remaining_futures)
                try:
                    progress_callback(completed, total_tasks)
                except TypeError:
                    progress_callback(int(completed / total_tasks * 100))
        
        results = ray.get(futures)

        # 加入圖片至 Dataset
        image_id = 0
        for images in results:
            for image in images:
                image_id += 1
                self.add_image(
                    'cell',
                    image_id=image_id, # 這裡 image_id 為流水號
                    path=image['path'], # path 為 SSD 上的檔案路徑
                    width=image['width'], height=image['height'],
                    polygons=image['polygons'],
                    num_ids=image['num_ids'])

    def save_mask(self, mask, image_id, output_dir):
        """將 Mask 存為圖檔 (除錯用)"""
        filename = os.path.splitext(os.path.basename(self.image_info[image_id]['path']))[0]
        filename = filename + "_mask.png"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, mask)
        # 注意：這裡不需要 return

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        # 建立總 Mask 容器 (H, W, N)
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # 1. 轉換座標
            pts = np.vstack((p['all_points_x'], p['all_points_y'])).astype(np.int32).T
            
            # 2. 座標限制 (Clip) 防止超出邊界
            pts[:, 0] = np.clip(pts[:, 0], 0, info["width"] - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, info["height"] - 1)

            # 3. [修正] 建立一個連續的暫存層來畫圖
            # OpenCV 需要連續的記憶體，不能直接畫在 mask[:, :, i] 這種切片上
            temp_mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
            
            # 畫在暫存層 (這是安全的)
            cv2.fillPoly(temp_mask, [pts], color=1)
            
            # 將畫好的暫存層複製回總 Mask (這也是安全的)
            mask[:, :, i] = temp_mask

        # 回傳 mask (bool) 和 class_ids
        return mask.astype(np.bool_), np.array(info['num_ids'], dtype=np.int32)

    def save_json(self, subset):
        """儲存目前的 Dataset 資訊為 JSON"""
        images = []
        for image_id in self.image_ids:
            info = self.image_info[image_id]
            images.append({
                "id": int(image_id),
                "path": info["path"],
                "width": int(info["width"]),
                "height": int(info["height"]),
                "polygons": [self.convert_numpy_to_python(polygon) for polygon in info["polygons"]],
                "num_ids": [int(num_id) for num_id in info["num_ids"]]
            })
        print("Saving JSON file...")
        with open(f"{subset}.json", "w") as f:
            json.dump(images, f, default=str)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]
