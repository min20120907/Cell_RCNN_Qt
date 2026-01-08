import random
from matplotlib import pyplot as plt
import skimage
from tqdm import tqdm
from mrcnn import utils
import json
import os
import cv2
import numpy as np
import ray

def remove_file_extension(filename):
    """Removes the file extension from a filename, even if the filename might have multiple dots."""
    filename_parts = filename.rsplit('.', 1)
    return filename_parts[0] if len(filename_parts) == 2 else filename

def convert_to_polygon_dict(list_polygon):
    """Converts a list of polygon points to a dictionary format with 'all_points_x' and 'all_points_y'."""
    if not list_polygon or not isinstance(list_polygon, list):
        raise ValueError("Invalid input for polygon conversion")
    return {'all_points_x': [point[0] for point in list_polygon], 'all_points_y': [point[1] for point in list_polygon]}

def convert_to_polygons(data):
    """Converts list of dictionaries to list of polygons."""
    return [[[x, y] for x, y in zip(d['all_points_x'], d['all_points_y'])] for d in data]

def are_polygons_close(polygon1, polygon2, x_diff_threshold=256, y_diff_threshold=256):
    """Checks if two polygons are close together."""
    polygon1 = np.asarray(polygon1)
    polygon2 = np.asarray(polygon2)

    x1_min, x1_max = np.min(polygon1[:, 0]), np.max(polygon1[:, 0])
    y1_min, y1_max = np.min(polygon1[:, 1]), np.max(polygon1[:, 1])
    x2_min, x2_max = np.min(polygon2[:, 0]), np.max(polygon2[:, 0])
    y2_min, y2_max = np.min(polygon2[:, 1]), np.max(polygon2[:, 1])

    x_diff = np.min([np.abs(x1_min - x2_min), np.abs(x1_max - x2_max)])
    y_diff = np.min([np.abs(y1_min - y2_min), np.abs(y1_max - y2_max)])
    
    return x_diff < x_diff_threshold and y_diff < y_diff_threshold
def group_polygons(polygons_with_ids):
    """Groups polygons that are close together, maintaining their class IDs."""
    groups = []
    for polygon, num_id in polygons_with_ids:
        polygon_added = False
        for group in groups:
            if any(are_polygons_close(polygon, p) for p, _ in group):
                group.append((polygon, num_id))
                polygon_added = True
                break
        if not polygon_added:
            groups.append([(polygon, num_id)])
    return groups


def crop_by_polygon(image, polygon):
    """Crops an image based on a polygon."""
    polygon = np.array(polygon, dtype=np.int32)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    x, y, w, h = cv2.boundingRect(polygon)
    cropped_image = masked_image[y:y + h, x:x + w]

    dst_size = max(cropped_image.shape)
    bottom = dst_size - cropped_image.shape[0]
    right = dst_size - cropped_image.shape[1]
    padded_image = cv2.copyMakeBorder(cropped_image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image
def crop_by_group(image, group_with_ids, target_size=(1024, 1024)):
    """Crops the image by the borders of the group, pads it to a square, and updates the polygons accordingly.

    Args:
        image (numpy.ndarray): The original image.
        group_with_ids (list): A list of tuples, each containing a polygon and its corresponding class ID.
        target_size (tuple): The desired size (width, height) of the output image.

    Returns:
        tuple: A tuple containing:
            - resized_image (numpy.ndarray): The cropped, padded, and resized image.
            - updated_polygons (list): A list of updated polygons with coordinates adjusted to the new image.
            - num_ids (list): A list of class IDs corresponding to each polygon.
    """
    polygons = [item[0] for item in group_with_ids]
    num_ids = [item[1] for item in group_with_ids]
    
    border_x_min = np.min([np.min([point[0] for point in polygon]) for polygon in polygons])
    border_y_min = np.min([np.min([point[1] for point in polygon]) for polygon in polygons])
    border_x_max = np.max([np.max([point[0] for point in polygon]) for polygon in polygons])
    border_y_max = np.max([np.max([point[1] for point in polygon]) for polygon in polygons])

    height, width = image.shape[:2]


    border_x_min = int(max(0, min(border_x_min, width - 1)))
    border_y_min = int(max(0, min(border_y_min, height - 1)))
    border_x_max = int(max(border_x_min + 1, min(border_x_max, width - 1)))
    border_y_max = int(max(border_y_min + 1, min(border_y_max, height - 1)))

    if border_x_max <= border_x_min or border_y_max <= border_y_min:
        print("Invalid cropping coordinates, skipping this group.")
        return None, None, None

    cropped_image = image[border_y_min:border_y_max, border_x_min:border_x_max]

    if cropped_image is None or cropped_image.size == 0:
        print("Cropped image is empty, skipping this group.")
        return None, None, None

    cropped_height, cropped_width = cropped_image.shape[:2]

    side_length = max(cropped_height, cropped_width)
    delta_w = side_length - cropped_width
    delta_h = side_length - cropped_height
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    color = [0, 0, 0]  
    padded_image = cv2.copyMakeBorder(
        cropped_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_LINEAR)

    scale = target_size[0] / side_length

    updated_polygons = []
    for polygon in polygons:
        updated_polygon = []
        for point in polygon:
            x = (point[0] - border_x_min + left) * scale
            y = (point[1] - border_y_min + top) * scale
            updated_polygon.append([x, y])
        updated_polygons.append(convert_to_polygon_dict(updated_polygon))

    return resized_image, updated_polygons, num_ids

@ray.remote
def load_annotations(annotation, subset_dir, class_id):
    """Loads annotations from a JSON file and processes images and polygons."""
    # 加载注释文件
    annotations = json.load(open(os.path.join(subset_dir, annotation)))
    annotations = [a for a in annotations.values() if a.get('regions')]
    output_dir = "/mnt/1TB-DISK-2/dataset-sq"
    images = []
    class_name_to_id = {"cell": 1, "chromosome": 2
                        , "nuclear": 3
                        }
    for a in annotations:
        # 处理区域信息，获取多边形和对象类别
        if type(a['regions']) is dict:
            regions = a['regions'].values()
        else:
            regions = a['regions']

        polygons = [r['shape_attributes'] for r in regions]
        objects = [s['region_attributes'] for s in regions]

        num_ids = []
        for obj in objects:
            class_name = obj.get('name')
            if class_name in class_name_to_id:
                num_ids.append(class_name_to_id[class_name])
            else:
                num_ids.append(class_id)  # 默认使用传入的 class_id，或处理未知类别

        # 将多边形与类别 ID 关联
        polygons_with_ids = list(zip(convert_to_polygons(polygons), num_ids))

        image_path = os.path.join(subset_dir, a['filename'])
        image = skimage.io.imread(image_path)

        # 将多边形分组
        groups_with_ids = group_polygons(polygons_with_ids)
        cropped_image_dictionaries = []
        polygon_id = 0
        for group_with_ids in groups_with_ids:
            # 调用 crop_by_group 函数，获取裁剪并调整后的图像和更新的多边形
            resized_image, updated_polygons, group_num_ids = crop_by_group(image, group_with_ids)
            if resized_image is not None:
                # 创建输出目录
                output_subdir = os.path.join(output_dir, remove_file_extension(a['filename']))
                os.makedirs(output_subdir, exist_ok=True)
                output_filename = os.path.join(output_subdir, f"{polygon_id}.png")
                polygon_id += 1
                if resized_image.size:
                    # 保存调整后的图像
                    if not os.path.isfile(output_filename):
                        cv2.imwrite(output_filename, resized_image)
                    # 创建图像信息字典
                    cropped_image_dictionary = {
                        'image_id': output_filename,
                        'path': output_filename,
                        'width': resized_image.shape[1],
                        'height': resized_image.shape[0],
                        'polygons': updated_polygons,
                        'num_ids': group_num_ids  # 使用更新后的类别 ID 列表
                    }
                    cropped_image_dictionaries.append(cropped_image_dictionary)
        images.extend(cropped_image_dictionaries)
    return images


@ray.remote
def process_polygon(p, height, width):
    """Processes a single polygon to create a mask."""
    pts = np.vstack((p['all_points_x'], p['all_points_y'])).astype(np.int32).T
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

    draw_mask = np.zeros([height, width, 3], dtype=np.uint8)
    cv2.fillPoly(draw_mask, [pts], color=(255, 255, 255))
    draw_mask = cv2.cvtColor(draw_mask, cv2.COLOR_BGR2GRAY)
    draw_mask = np.where(draw_mask > 0, 1, 0)

    return draw_mask

class CustomCroppingDataset(utils.Dataset):
    """Custom dataset class for handling cropped image data."""
    
    def convert_numpy_to_python(self, obj):
        """Converts numpy types to native Python types."""
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

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the dataset."""
        self.add_class("cell", 1, "cell")
        self.add_class("cell", 2, "chromosome")
        self.add_class("cell", 3, "nuclear")
        assert subset in ["train", "val", "test"]
        subset_dir = os.path.join(dataset_dir, subset)

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])
        
        annotations = [f for f in os.listdir(subset_dir) if f.startswith("via_region_") and f.endswith(".json")]
        futures = [load_annotations.remote(a, subset_dir, 1) for a in annotations if "data_" in a]
        futures += [load_annotations.remote(a, subset_dir, 2) for a in annotations if "chromosome_" in a]
        futures += [load_annotations.remote(a, subset_dir, 3) for a in annotations if "nuclear_" in a]
        for _ in tqdm(to_iterator(futures), total=len(futures)):
            pass
        results = ray.get(futures)

        image_id = 0
        for images in results:
            for image in images:
                image_id += 1
                self.add_image(
                    'cell',
                    image_id=image_id,
                    path=image['path'],
                    width=image['width'],
                    height=image['height'],
                    polygons=image['polygons'],
                    num_ids=image['num_ids']
                )

    def save_mask(self, mask, image_id, output_dir):
        filename = os.path.splitext(os.path.basename(self.image_info[image_id]['path']))[0] + "_mask.png"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, mask)

    def load_mask(self, image_id):
        """Load mask for a given image ID."""
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            pts = np.vstack((p['all_points_x'], p['all_points_y'])).astype(np.int32).T
            pts[:, 0] = np.clip(pts[:, 0], 0, info["width"] - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, info["height"] - 1)

            draw_mask = np.zeros([info["height"], info["width"], 3], dtype=np.uint8)
            cv2.fillPoly(draw_mask, [pts], color=(255, 255, 255))
            draw_mask = cv2.cvtColor(draw_mask, cv2.COLOR_BGR2GRAY)
            draw_mask = np.where(draw_mask > 0, 1, 0)
            mask[:, :, i] = draw_mask

        return mask, np.array(info['num_ids'], dtype=np.int32)

    def save_json(self, subset):
        """Save the dataset to a JSON file."""
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
            json.dump(images, f)
    def split_dataset(self, dataset_dir, train_size=0.7, val_size=0.2, test_size=0.1, random_seed=42):
        """Splits the dataset into train, validation, and test sets.

        Args:
            dataset_dir (str): The root directory of the dataset.
            train_size (float): Proportion of the dataset to include in the train split.
            val_size (float): Proportion of the dataset to include in the validation split.
            test_size (float): Proportion of the dataset to include in the test split.
            random_seed (int): Random seed for reproducibility.

        Returns:
            dict: Dictionary containing lists of filenames for each subset.
        """
        # Ensure the sizes sum to 1
        assert train_size + val_size + test_size == 1, "The sizes should sum up to 1."

        # List all files in the dataset directory
        all_files = [f for f in os.listdir(dataset_dir) if f.startswith("via_region_") and f.endswith(".json")]
        random.seed(random_seed)
        random.shuffle(all_files)

        # Calculate the number of files for each split
        num_files = len(all_files)
        num_train = int(train_size * num_files)
        num_val = int(val_size * num_files)

        # Split the files
        train_files = all_files[:num_train]
        val_files = all_files[num_train:num_train + num_val]
        test_files = all_files[num_train + num_val:]

        return {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

    def load_or_split_dataset(self, dataset, train_percentage, val_percentage, test_percentage, save_dir):
        """Load an existing dataset or split into new train, val, and test sets.

        Args:
            dataset (CustomCroppingDataset): The dataset object to load and split.
            train_percentage (float): Percentage of data for the training set.
            val_percentage (float): Percentage of data for the validation set.
            test_percentage (float): Percentage of data for the testing set.
            save_dir (str): Directory where the split datasets will be saved.

        Returns:
            tuple: Three CustomCroppingDataset objects for train, val, and test sets.
        """
        # Ensure the percentages sum up to less than or equal to 1
        assert train_percentage + val_percentage + test_percentage <= 1, "The percentages should sum up to <= 1."

        # Combine all images from the loaded dataset
        all_image_ids = list(dataset.image_ids)
        random.shuffle(all_image_ids)

        # Determine split sizes
        num_images = len(all_image_ids)
        num_train = int(train_percentage * num_images)
        num_val = int(val_percentage * num_images)

        train_ids = all_image_ids[:num_train]
        val_ids = all_image_ids[num_train:num_train + num_val]
        test_ids = all_image_ids[num_train + num_val:]

        # Create train, val, and test datasets
        train_set = CustomCroppingDataset()
        val_set = CustomCroppingDataset()
        test_set = CustomCroppingDataset()

        # Load each subset
        train_set.load_from_ids(dataset, train_ids)
        val_set.load_from_ids(dataset, val_ids)
        test_set.load_from_ids(dataset, test_ids)

        # Save each subset to the specified directory
        train_set.save_json(os.path.join(save_dir, 'train.json'))
        val_set.save_json(os.path.join(save_dir, 'val.json'))
        test_set.save_json(os.path.join(save_dir, 'test.json'))

        return train_set, val_set, test_set

    def load_from_ids(self, source_dataset, image_ids):
        """Loads a subset of images from another dataset using specified IDs.

        Args:
            source_dataset (CustomCroppingDataset): The source dataset object.
            image_ids (list): List of image IDs to load.
        """
        # Copy class information from the source dataset
        for i in range(len(source_dataset.class_info)):
            self.add_class(source_dataset.class_info[i]["source"], source_dataset.class_info[i]["id"], source_dataset.class_info[i]["name"])

        # Copy images from the source dataset based on the provided IDs
        for image_id in image_ids:
            info = source_dataset.image_info[image_id]
            self.add_image(info['source'], image_id=image_id, path=info['path'],
                           width=info['width'], height=info['height'],
                           polygons=info['polygons'], num_ids=info['num_ids'])

        self.prepare()  # Prepare the dataset for usage

    def save_json(self, file_path):
        """Save the dataset information to a JSON file.

        Args:
            file_path (str): Path where the JSON file will be saved.
        """
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

        # Save to JSON file
        with open(file_path, "w") as f:
            json.dump(images, f)

    def convert_numpy_to_python(self, obj):
        """Converts numpy types to native Python types."""
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self.convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_to_python(item) for item in obj]
        else:
            return obj

    def load_and_split_data(self, dataset_dir):
        """Splits the dataset and loads each subset.

        Args:
            dataset_dir (str): Root directory of the dataset.
        """
        # Split the dataset
        split = self.split_dataset(dataset_dir)

        # Load each subset using the existing load_custom method
        print("Loading training data...")
        self.load_custom(dataset_dir, 'train', split['train'])

        print("Loading validation data...")
        self.load_custom(dataset_dir, 'val', split['val'])

        print("Loading test data...")
        self.load_custom(dataset_dir, 'test', split['test'])

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]
