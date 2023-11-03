import skimage
from tqdm import tqdm
from mrcnn import utils
import json
import os
import cv2
import numpy as np
import ray


def convert_to_2d_list(nested_list):
  """Converts a nested list to a list of lists.

  Args:
    nested_list: A nested list to convert.

  Returns:
    A list of lists.
  """

  flat_list = []
  for sublist in nested_list:
    flat_list.extend(sublist)

  # Convert the flat list to a 2D list.
  list_of_lists = np.array(flat_list).reshape((-1, 2))

  return list_of_lists

def convert_to_polygon_dict(polygon):
  """Converts a polygon to a dictionary with the keys "all_points_x" and "all_points_y".

  Args:
    polygon: A numpy array of the polygon points.

  Returns:
    A dictionary with the keys "all_points_x" and "all_points_y".
  """
  
  polygon_dict = {}
  polygon_dict["all_points_x"] = polygon[:, 0]
  polygon_dict["all_points_y"] = polygon[:, 1]

  return polygon_dict

def crop_by_polygon(img, polygon):
    crop_size = (64,64)

    # Adjust the size of the cropped area to cover the entire bounding box
    top_left = (min(list(map(list, zip(*polygon)))[0]), min(list(map(list, zip(*polygon)))[1]))
    
    # Crop the image
    new_img = img[int(top_left[1]):int(top_left[1]+crop_size[1]), int(top_left[0]):int(top_left[0]+crop_size[0])]

    # Update the polygon coordinates
    new_polygon = polygon-top_left
    new_polygon = new_polygon.clip(min=0, max=max(crop_size))
    # Check if the new polygon coordinates are within the new image boundaries
    # assert np.all(new_polygon >= 0) and np.all(new_polygon[:, 0] < crop_size[0]) and np.all(new_polygon[:, 1] < crop_size[1]), "The new polygon coordinates are out of the new image boundaries: "+str(new_polygon)
    
    return new_img, new_polygon

def convert_polygon_to_cv2_format(polygon):
  """Converts a polygon to the format that the cv2.boundingRect() function expects.

  Args:
    polygon: A list of (x, y) coordinates of the polygon vertices.

  Returns:
    A NumPy array of points in the format expected by cv2.boundingRect().
  """

  # Convert the polygon to a NumPy array.
  polygon = np.array(polygon)

  # Reshape the polygon array to have the same dimensions as the input to cv2.boundingRect().
  polygon = polygon.reshape((-1, 1, 2))

  return polygon

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
@ray.remote
def load_annotations(annotation, subset_dir, class_id):
    # Load annotations from JSON file
    annotations = json.load(open(os.path.join(subset_dir, annotation)))
    annotations = list(annotations.values())
    annotations = [a for a in annotations if a['regions']]
    output_dir = "/mnt/1TB-DISK-2/dataset-64"
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
        image = cv2.imread(image_path)

        height, width = image.shape[:2]
        # Crop the image to each polygon and save the cropped images to NumPy arrays.
        cropped_images = []
        for polygon in polygons:
            polygon = np.array([
                [polygon['all_points_x'][i], polygon['all_points_y'][i]]
                for i in range(len(polygon['all_points_x']))
            ])
            # print(polygon)
            cropped_image =None
            
            cropped_image, cropped_polygon = crop_by_polygon(
                image, polygon)
            
            # print(cropped_image.shape)
            if cropped_image is not None:
                cropped_images.append(
                    (cropped_image, cropped_polygon))

        # Add the cropped images to the image dictionary.
        cropped_image_dictionaries = []
        polygon_id = 0
        for cropped_image, cropped_polygon in cropped_images:
            
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

            # continue
            # print([calculate_new_polygon_coordinates(polygon, cropped_image.shape[:2])])
            # print(new_polygons)
            cropped_image_dictionary = {
                'image_id': output_filename,
                'path': output_filename,
                'width': cropped_image.shape[1],
                'height': cropped_image.shape[0],
                'polygons': convert_to_polygon_dict(convert_to_2d_list(cropped_polygon)),
                'num_ids': [class_id]
            }
            # print(cropped_image_dictionary)
            cropped_image_dictionaries.append(cropped_image_dictionary)

        # Add the cropped image dictionaries to the list of images.
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
        mask = np.zeros([info["height"], info["width"], 1],
                        dtype=np.uint8)
        p = info["polygons"]
        polygon = np.array([
    	[p['all_points_x'][i], p['all_points_y'][i]]
    	for i in range(len(p['all_points_x']))
    	])
        # Get indexes of pixels inside the polygon and set them to 1
        # rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'],(info["height"], info["width"]))
        mask[:,:,0] = skimage.draw.polygon2mask((info["height"], info["width"]),polygon)
        return mask.astype(np.bool), np.array(info['num_ids'], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]