from matplotlib import pyplot as plt
import skimage
from tqdm import tqdm
from mrcnn import utils
import json
import os
import cv2
import numpy as np
import ray
import os
import skimage.transform

def remove_file_extension(filename):
  """Removes the file extension from a filename, even if the filename might have multiple dots.

  Args:
    filename: The filename to remove the file extension from.

  Returns:
    A string containing the filename without the file extension.
  """

  # Split the filename at the last dot.
  filename_parts = filename.rsplit('.', 1)

  # If the filename has multiple dots, the last part will be the file extension.
  if len(filename_parts) == 2:
    return filename_parts[0]
  else:
    return filename
def convert_to_polygon_dict(list_polygon):
  """Converts a polygon to a dictionary with the keys "all_points_x" and "all_points_y".

  Args:
    polygon: A numpy array of the polygon points.

  Returns:
    A dictionary with the keys "all_points_x" and "all_points_y".
  """
  # Convert list to dictionary
  return {'all_points_x': [point[0] for point in list_polygon], 'all_points_y': [point[1] for point in list_polygon]}


def convert_to_polygons(data):
   polygons = [[[x, y] for x, y in zip(d['all_points_x'], d['all_points_y'])] for d in data]
   return polygons
def are_polygons_close(polygon1, polygon2):
  """Checks if two polygons are close together.

  Args:
    polygon1: A list of the first polygon points.
    polygon2: A list of the second polygon points.

  Returns:
    True if the polygons are close together, False otherwise.
  """
  polygon1 = np.asarray(polygon1)
  polygon2 = np.asarray(polygon2)

  x1_min = np.min(polygon1[:, 0])
  x1_max = np.max(polygon1[:, 0])
  y1_min = np.min(polygon1[:, 1])
  y1_max = np.max(polygon1[:, 1])
  x2_min = np.min(polygon2[:, 0])
  x2_max = np.max(polygon2[:, 0])
  y2_min = np.min(polygon2[:, 1])
  y2_max = np.max(polygon2[:, 1])
  x_diff = np.min([np.abs(x1_min - x2_min), np.abs(x1_max - x2_max)])
  y_diff = np.min([np.abs(y1_min - y2_min), np.abs(y1_max - y2_max)])
  x_diff_threshold = 1000
  y_diff_threshold = 1000
  if x_diff < x_diff_threshold and y_diff < y_diff_threshold:
    return True
  else:
    return False


def group_polygons(polygons):
  # group the polygons that are near together, and crop them together
  groups = []
  # print(polygons)
  for polygon in polygons:
    # print(polygon)
    polygon_added = False
    for group in groups:
      for polygon2 in group:
        # Check if the polygons are close together
        if are_polygons_close(polygon, polygon2):
          # Add the polygon to the group
          group.append(polygon)
          polygon_added = True
          break
      if polygon_added:
        break
    if not polygon_added:
      # Create a new group
      groups.append([polygon])
  return groups

def crop_by_polygon(image, polygon):
  # Convert the polygon to a numpy array
  polygon = np.array(polygon, dtype=np.int32)

  # Crop the image to the polygon
  mask = np.zeros(image.shape[:2], dtype=np.uint8)
  cv2.fillPoly(mask, [polygon], 255)
  masked_image = cv2.bitwise_and(image, image, mask=mask)
  x, y, w, h = cv2.boundingRect(polygon)
  cropped_image = masked_image[y:y+h, x:x+w]

  dst_size = max(cropped_image.shape)
  bottom = dst_size - cropped_image.shape[0]
  right = dst_size - cropped_image.shape[1]
  padded_image = cv2.copyMakeBorder(cropped_image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=[0,0,0])

  return padded_image, polygon #temporarily return cropped_image

def crop_by_group(image, group):
  # Crop the image by the borders of the group, and update the polygons
  # accordingly
  border_x_min = np.min([np.min([point[0] for point in polygon]) for polygon in group])
  border_y_min = np.min([np.min([point[1] for point in polygon]) for polygon in group])
  border_x_max = np.max([np.max([point[0] for point in polygon]) for polygon in group])
  border_y_max = np.max([np.max([point[1] for point in polygon]) for polygon in group])
  border_polygon = np.array([
    [border_x_min, border_y_min],
    [border_x_max, border_y_min],
    [border_x_max, border_y_max],
    [border_x_min, border_y_max]
  ])
  try:
    cropped_image, _ = crop_by_polygon(image, border_polygon)
  except:
    print("Error occured")
    print(image.shape)
    print(border_polygon)
    return None, None
  updated_polygons = []
  for polygon in group:
    updated_polygon = [[point[0] - border_x_min, point[1] - border_y_min] for point in polygon]
    updated_polygons.append(convert_to_polygon_dict(updated_polygon))
  return cropped_image, updated_polygons

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
    output_dir = "/home/e814/Documents/dataset-sq"
    # Add images

    # Group polygons
    groups = group_polygons(convert_to_polygons(polygons))
    cropped_image_dictionaries = []
    polygon_id = 0
    # Crop bounding box and update polygons
    for group in groups:
      cropped_image, updated_polygons = crop_by_group(image, group)  # Fix the function name
      if cropped_image is not None:
        try:
            os.mkdir(os.path.join(
                output_dir, os.path.dirname(a['filename'])))
        except:
            pass
        try:
            os.mkdir(os.path.join(
                output_dir, remove_file_extension(a['filename'])))
        except:
            pass
        output_filename = os.path.join(
            output_dir, f"{remove_file_extension(a['filename'])}/{polygon_id}.png")
        polygon_id += 1
        if cropped_image.size:
            try:
                if not os.path.isfile(output_filename):
                    cv2.imwrite(output_filename, cropped_image)
            except:
                continue
            cropped_image_dictionary = {
                'image_id': output_filename,
                'path': output_filename,
                'width': cropped_image.shape[1],
                'height': cropped_image.shape[0],
                'polygons': updated_polygons,
                'num_ids': [class_id]*len(updated_polygons)
            }
            # print(cropped_image_dictionary)
            cropped_image_dictionaries.append(cropped_image_dictionary)
    images.extend(cropped_image_dictionaries)
  return images
@ray.remote
def process_polygon(p, height, width):
    # Convert the polygon points to the correct shape
    pts = np.vstack((p['all_points_x'], p['all_points_y'])).astype(np.int32).T

    # Validate and clip the coordinates to the valid range
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

    # Draw the polygon on the mask using fillPoly
    draw_mask = np.zeros([height, width, 3], dtype=np.uint8)
    cv2.fillPoly(draw_mask, [pts], color=(255, 255, 255))

    draw_mask = cv2.cvtColor(draw_mask, cv2.COLOR_BGR2GRAY)
    draw_mask = np.where(draw_mask > 0, 1, 0)

    return draw_mask
class CustomCroppingDataset(utils.Dataset):
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
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cell", 1, "cell")
        self.add_class("cell", 2, "chromosome")
        # self.add_class("cell", 3, "nuclear")
        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        subset_dir = os.path.join(dataset_dir, subset)

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])
        
        # Load annotations from all JSON files using Ray multiprocessing
        annotations = [f for f in os.listdir(subset_dir) if f.startswith("via_region_") and f.endswith(".json")]
        futures = [load_annotations.remote(a, subset_dir, 1) for a in annotations if "data_" in a]
        futures += [load_annotations.remote(a, subset_dir, 2) for a in annotations if "chromosome_" in a]
        # futures = [load_annotations.remote(a, subset_dir, 3) for a in annotations if "nuclear_" in a]
        # Showing the progressbar
        for _ in tqdm(to_iterator(futures), total=len(futures)):
            pass
        results = ray.get(futures)
        

        # Add images
        image_id = 0
        for images in results:
            for image in images:
                image_id += 1
                self.add_image(
                    'cell',
                    image_id=image_id,  # use file name as a unique image id
                    path=image['path'],
                    width=image['width'], height=image['height'],
                    polygons=image['polygons'],
                    num_ids=image['num_ids'])
  def save_mask(self, mask, image_id, output_dir):
    filename = os.path.splitext(os.path.basename(self.image_info[image_id]['path']))[0]
    filename = filename + "_mask.png"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, mask)
  def load_mask_old(self, image_id):
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        height = info["height"]
        width = info["width"]
        results = ray.get([process_polygon.remote(p, height, width) for _, p in enumerate(info["polygons"])])

        for i, result in enumerate(results):
            mask[:, :, i] = result
        return mask.astype(np.bool), np.array(info['num_ids'], dtype=np.int32)
  def load_mask(self, image_id):
    info = self.image_info[image_id]
    mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
            dtype=np.uint8)

    for i, p in enumerate(info["polygons"]):
      # Convert the polygon points to the correct shape
      pts = np.vstack((p['all_points_x'], p['all_points_y'])).astype(np.int32).T
      
      # Validate and clip the coordinates to the valid range
      pts[:, 0] = np.clip(pts[:, 0], 0, info["width"] - 1)
      pts[:, 1] = np.clip(pts[:, 1], 0, info["height"] - 1)

      # Draw the polygon on the mask using fillPoly
      # fix Expected Ptr<cv::UMat> for argument 'img'
      draw_mask = np.zeros([info["height"], info["width"], 3], dtype=np.uint8)
      cv2.fillPoly(draw_mask, [pts], color=(255, 255, 255))
      
      draw_mask = cv2.cvtColor(draw_mask, cv2.COLOR_BGR2GRAY)
      draw_mask = np.where(draw_mask > 0, 1, 0)
      mask[:, :, i] = draw_mask
    
    return mask, np.array(info['num_ids'], dtype=np.int32)
  # save the json file according to the list of images in the dataset
  def save_json(self, subset):
    # Create a dictionary with the image information
    images = []
    for image_id in self.image_ids:
        info = self.image_info[image_id]
        images.append({
            "id": int(image_id),  # Convert int64 to int
            "path": info["path"],
            "width": int(info["width"]),  # Convert int64 to int
            "height": int(info["height"]),  # Convert int64 to int
            "polygons": [self.convert_numpy_to_python(polygon) for polygon in info["polygons"]],
            "num_ids": [int(num_id) for num_id in info["num_ids"]]  # Convert int64 to int
        })
    print("Saving JSON file...")
    # Save the dictionary to a JSON file
    with open(f"{subset}.json", "w") as f:
        json.dump(images, f)
  def image_reference(self, image_id):
    """Return the path of the image."""
    info = self.image_info[image_id]
    return info["path"]
