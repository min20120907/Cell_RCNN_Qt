import skimage
from tqdm import tqdm
from mrcnn import utils
import json
import os
import cv2
import numpy as np
import ray
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
  x_diff_threshold = 100
  y_diff_threshold = 100
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

  return padded_image, polygon

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
    output_dir = "/mnt/1TB-DISK-2/dataset-sq"
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

class CustomCroppingDataset(utils.Dataset):
  def load_custom(self, dataset_dir, subset):
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cell", 1, "cell")
        self.add_class("cell", 2, "chromosome")
        self.add_class("cell", 3, "nuclear")
        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        subset_dir = os.path.join(dataset_dir, subset)

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])
        
        # Load annotations from all JSON files using Ray multiprocessing
        annotations = [f for f in os.listdir(subset_dir) if f.startswith("via_region_") and f.endswith(".json")]
        futures = [load_annotations.remote(a, subset_dir, 1) for a in annotations if "data_" in a] + \
                    [load_annotations.remote(a, subset_dir, 2) for a in annotations if "chromosome_" in a] + \
                    [load_annotations.remote(a, subset_dir, 3) for a in annotations if "nuclear_" in a]
        # Showing the progressbar
        for _ in tqdm(to_iterator(futures), total=len(futures)):
            pass
        results = ray.get(futures)
        

        # Add images
        for images in results:
            for image in images:
                self.add_image(
                    'cell',
                    image_id=image['image_id'],  # use file name as a unique image id
                    path=image['path'],
                    width=image['width'], height=image['height'],
                    polygons=image['polygons'],
                    num_ids=image['num_ids'])

  def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # print(f"i={i}, rr={rr}, cc={cc}, len(cc)={len(cc)}")
            try:
                mask[rr, cc, i] = 1
            except:
                rr = np.clip(rr, 0, info["height"] - 1)  # Clip row indices to valid range
                cc = np.clip(cc, 0, info["width"] - 1)   # Clip column indices to valid range
                mask[rr, cc, i] = 1
                # print("Error Occured")
                # print(f"i={i}, rr={rr}, cc={cc}, len(cc)={len(cc)}")
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(info['num_ids'], dtype=np.int32)

  def image_reference(self, image_id):
    """Return the path of the image."""
    info = self.image_info[image_id]
    return info["path"]
