import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import skimage
import matplotlib.pyplot as plt
import tensorflow as tf
import ray
import cv2
from mrcnn.config import Config
from mrcnn.utils import compute_ap, Dataset, compute_iou
import mrcnn.model as modellib
from sklearn.metrics import precision_score

import matplotlib.patches as patches
from skimage import measure
# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Function to generate masks on CPU


@ray.remote
def generate_mask_subsea(args):
    height, width, subset = args
    with tf.device("/cpu:0"):
        mask = np.zeros([height, width, len(subset)], dtype=np.uint8)
        for i, j in enumerate(range(subset[0], subset[1])):
            start = subset[j]['all_points'][:-1]
            rr, cc = skimage.draw.polygon(start[:, 1], start[:, 0])
            mask[rr, cc, i] = 1
    return mask


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


def generate_mask_subset(args):
    height, width, subset = args
    mask = np.zeros([height, width, len(subset)], dtype=np.uint8)
    for i, j in enumerate(range(subset[0], subset[1])):
        start = subset[j]['all_points'][:-1]
        rr, cc = skimage.draw.polygon(start[:, 1], start[:, 0])
        mask[rr, cc, i] = 1
    return mask


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
    # Compute the center of the polygon
    center = np.mean(polygon, axis=0)
    crop_size = (256,256)
    center[0] = int(center[0])
    center[1] = int(center[1])
    top_left = ()
    # Compute the distances from the center to the image borders
    # d_right_bottom = (img.shape[0]-center[0], img.shape[1]-center[1])
    # if(d_right_bottom[0]>128 and d_right_bottom[1]>128):
    #     top_left = center - 128
    # elif(d_right_bottom[0]<=128 and d_right_bottom[1]>128):
    #     top_left = (center[0] - 256 + img.shape[0], center[1] - 128)
    # elif(d_right_bottom[0]>128 and d_right_bottom[1]<=128):
    #     top_left = (center[0]-128, center[1]-256+img.shape[1])
    # else:
    #     top_left = (center[0] - 256 + img.shape[0], center[1] - 256 + img.shape[1])
    # Calculate the bounding box of the polygon
    bbox = np.array([polygon[:, 0].min(), polygon[:, 1].min(), polygon[:, 0].max(), polygon[:, 1].max()])
  
    top_left = (min(bbox[0], img.shape[1] - crop_size[0]), min(bbox[1], img.shape[0] - crop_size[1]))
    # Adjust the size of the cropped area to cover the entire bounding box
    
    
    # Crop the image
    new_img = img[int(top_left[1]):int(top_left[1]+crop_size[1]), int(top_left[0]):int(top_left[0]+crop_size[0])]
    
    # Check if the new image is of size 256x256
    assert new_img.shape == (256, 256,3), "The shape of the new image is not 256x256, but "+str(new_img.shape)+" top_left: "+str(top_left)

    # Update the polygon coordinates
    new_polygon = polygon-(min(list(map(list, zip(*polygon)))[0]), min(list(map(list, zip(*polygon)))[1]))
    new_polygon = new_polygon.clip(min=0, max=255)
    # Check if the new polygon coordinates are within the new image boundaries
    assert np.all(new_polygon >= 0) and np.all(new_polygon[:, 0] < 256) and np.all(new_polygon[:, 1] < 256), "The new polygon coordinates are out of the new image boundaries: "+str(new_polygon)
    
    return new_img, new_polygon

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
    output_dir = "/home/e814/Documents/dataset-256"
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
            print(cropped_image.shape)
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


class CustomDataset(Dataset):

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
        annotations = [f for f in os.listdir(subset_dir) if f.startswith(
            "via_region_") and f.endswith(".json")]
        futures = [load_annotations.remote(a, subset_dir, 1) for a in annotations if "data_" in a] + \
            [load_annotations.remote(a, subset_dir, 2) for a in annotations if "chromosome_" in a] + \
            [load_annotations.remote(a, subset_dir, 3)
             for a in annotations if "nuclear_" in a]
        # Showing the progressbar
        for _ in tqdm(to_iterator(futures), total=len(futures)):
            pass
        results = ray.get(futures)

        # Add images
        for images in results:
            for image in images:
                self.add_image(
                    'cell',
                    # use file name as a unique image id
                    image_id=image['image_id'],
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
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        # print(f"i={i}, rr={rr}, cc={cc}, len(cc)={len(cc)}")
        try:
            mask[rr, cc, 0] = 1
        except:
            # Clip row indices to valid range
            rr = np.clip(rr, 0, info["height"] - 1)
            # Clip column indices to valid range
            cc = np.clip(cc, 0, info["width"] - 1)
            mask[rr, cc, 0] = 1
            # print("Error Occured")
            # print(f"i={i}, rr={rr}, cc={cc}, len(cc)={len(cc)}")
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(info['num_ids'], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


class GTConfig(Config):
    NAME = "cell"
    TEST_MODE = "inference"
    IMAGE_RESIZE_MODE = "none"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # Change this to a lower value, e.g., 1
    NUM_CLASSES = 1 + 3
    USE_MINI_MASK = False
    VALIDATION_STEPS = 50
    IMAGE_MAX_DIM = 4096
    IMAGE_MIN_DIM = 1024


class InferenceConfig(Config):
    NAME = "cell"
    TEST_MODE = "inference"
    # IMAGE_RESIZE_MODE = "pad64"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # Change this to a lower value, e.g., 1
    NUM_CLASSES = 1 + 3
    USE_MINI_MASK = False
    VALIDATION_STEPS = 50
    IMAGE_MAX_DIM = 4096
    IMAGE_MIN_DIM = 1024


class EvalImage():
    def __init__(self, dataset, model, cfg, cfg_GT, output_folder):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg
        self.cfg_GT = cfg_GT
        self.output_folder = output_folder

    def convert_pixels_to_inches(self, pixels):
        dpi = 300
        inches = pixels / dpi
        return inches

    def evaluate_model(self, limit):
        precisions = []
        # Existing code
        if limit == -1:
            limit = len(self.dataset.image_ids)
        for image_id in range(limit):
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(self.dataset, self.cfg_GT,
                                       self.dataset.image_ids[image_id])
            if gt_mask.size == 0:
                # precisions.append(1)
                continue
            if np.max(gt_mask) == 0:
                # precisions.append(1)
                continue
            molded_images = np.expand_dims(
                modellib.mold_image(image, self.cfg_GT), 0)
            # results = self.model.detect([image], verbose=0)
            # r = results[0]
            # print(r['masks'].shape)
            # Compute AP
            '''
            AP, P,recall,overlaps =\
                compute_ap(gt_bbox, gt_class_id, gt_mask,\
                            r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=0.5)
            precisions.append(AP)
            print("Precision: ",np.max(P))
            print("overlaps: ", np.max(overlaps))
            '''
            # Get the original image size in pixels
            image_height, image_width = image.shape[:2]

            # Convert the image size from pixels to inches
            image_height_inches = self.convert_pixels_to_inches(image_height)
            image_width_inches = self.convert_pixels_to_inches(image_width)

            # Set the figure size to the original image size in inches
            fig = plt.figure(figsize=[image_width_inches, image_height_inches])
            ax = fig.add_subplot(111)
            # Set the DPI of the plot to 300
            plt.rcParams["figure.dpi"] = 300

            # Plot the image
            ax.imshow(image)

            # Plot the GT mask
            for i in range(gt_mask.shape[-1]):
                mask = gt_mask[..., i]
                for contour in measure.find_contours(mask, 0.5):
                    ax.plot(contour[:, 1], contour[:, 0], '-g', linewidth=2)
            '''
            # Plot the predicted mask
            for i in range(r['masks'].shape[-1]):
                mask = r['masks'][..., i]
                for contour in measure.find_contours(mask, 0.5):
                    ax.plot(contour[:, 1], contour[:, 0], '-r', linewidth=2)
            '''
            # Save the plot to a file
            filename = os.path.join(
                self.output_folder, f'image_{image_id}.png')
            plt.savefig(filename)
            plt.close()


if __name__ == "__main__":
    # ... (Your existing argparse code)

    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--dataset", required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--workdir", required=True,
                        help="Path to the working directory")
    parser.add_argument("--weight_path", required=True,
                        help="Path to the weight file")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Number of images to evaluate (default: all)")
    parser.add_argument("--cpu", action="store_true",
                        help="Run on CPU instead of GPU")
    parser.add_argument("--output_folder",
                        help="Output folder for the evaluation results")
    args = parser.parse_args()
    if args.cpu:
        # Set to empty string to use CPU only
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    DATASET_PATH = args.dataset
    WORK_DIR = args.workdir
    LIMIT = args.limit
    weight_path = args.weight_path
    output_folder = args.output_folder
    dataset_test = CustomDataset()
    dataset_test.load_custom(DATASET_PATH, "test")
    dataset_test.prepare()
    print("Number of Images: ", dataset_test.num_images)
    print("Number of Classes: ", dataset_test.num_classes)

    model = modellib.MaskRCNN(
        mode="inference", config=InferenceConfig(), model_dir=WORK_DIR + "/logs")
    model.load_weights(weight_path, by_name=True)
    eval = EvalImage(dataset_test, model, InferenceConfig(),
                     GTConfig(),  output_folder)
    results = eval.evaluate_model(limit=LIMIT)
