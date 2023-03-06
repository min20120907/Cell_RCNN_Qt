import json
import os
import re
from PIL import Image
import random
import multiprocessing as mp
# 讀取VIA的json格式的資料集
with open('via_region_data.json', 'r') as f:
    dataset = json.load(f)

# 定義變換函數，例如縮放、旋轉、平移、翻轉等
def transform(image, region):
    # 縮放圖像
    scale = random.uniform(0.5, 2.0)
    new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
    image = image.resize(new_size)
    
    # 旋轉圖像
    angle = random.uniform(-45, 45)
    image = image.rotate(angle)
    
    # 平移圖像
    x_offset = random.randint(-50, 50)
    y_offset = random.randint(-50, 50)
    image = image.transform(image.size, Image.AFFINE, (1, 0, x_offset, 0, 1, y_offset))
    
    # 翻轉圖像
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # 縮放region的坐標
    x_scale = image.size[0] / region['size_x']
    y_scale = image.size[1] / region['size_y']
    x_new = [int(x * x_scale) for x in region['shape_attributes']['all_points_x']]
    y_new = [int(y * y_scale) for y in region['shape_attributes']['all_points_y']]
    region['shape_attributes']['all_points_x'] = x_new
    region['shape_attributes']['all_points_y'] = y_new
    
    return image, region
# 定義單個圖像的處理函數
def process_image(filename, image_data):
    image = Image.open(os.path.join(filename.split(".")[0]+".png"))
    new_dataset = {}
    for region_id in image_data['regions']:
        region = image_data['regions'][region_id]
        
        # 省略不變的程式碼
        # 取得檔案名稱和副檔名
        file_name, file_ext = filename.split(".")[0], ".png"
        
        # 獲取圖像的寬度和高度
        image_width, image_height = image.size
        
        # 獲取region的寬度和高度
        if 'size_x' in region and 'size_y' in region:
            region_width, region_height = region['size_x'], region['size_y']
        else:
            # 如果region中沒有寬度和高度資訊，就從圖像中獲取
            try:
                region_width = max(region['shape_attributes']['all_points_x']) - min(region['shape_attributes']['all_points_x'])
                region_height = max(region['shape_attributes']['all_points_y']) - min(region['shape_attributes']['all_points_y'])
            except:
                print(region)
                region_width = max(region['x']) - min(region['x'])
                region_height = max(region['y']) - min(region['y'])

        
        region['size_x'] = region_width
        region['size_y'] = region_height
        new_filename = f"{file_name}_transformed{file_ext}"
        new_region_id = f"{region_id}"
        
        new_image, new_region = transform(image, region)
        
        
        new_file_path = os.path.join(new_filename)
        
        # 儲存新圖片
        new_image.save(new_file_path)
        # 將新的圖像
        if new_filename not in new_dataset:
            new_dataset[new_filename] = {
                "fileref": "",
                "size": os.path.getsize(filename.split(".")[0]+".png"),
                "filename": new_filename,
                "base64_img_data": "",
                "file_attributes": {},
                "regions": {}
            }
        new_dataset[new_filename]['regions'][new_region_id] = new_region
    return new_dataset

if __name__ == '__main__':
    with open('via_region_data.json', 'r') as f:
        dataset = json.load(f)
    
    # 使用 multiprocessing 創建多個進程
    with mp.Pool() as pool:
        results = [pool.apply_async(process_image, args=(filename, image_data)) for filename, image_data in dataset.items()]
        new_datasets = [res.get() for res in results]
    
    # 合併結果
    new_dataset = {}
    for dataset in new_datasets:
        for filename, data in dataset.items():
            if filename not in new_dataset:
                new_dataset[filename] = data
            else:
                new_dataset[filename]['regions'].update(data['regions'])

    with open('via_region_data_transformed.json', 'a') as f:
        json.dump(new_dataset, f)
