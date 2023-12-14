from multiprocessing import Pool
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
from CustomCroppingDataset import CustomCroppingDataset
from CustomDataset import CustomDataset
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



def generate_mask_subset(args):
    height, width, subset = args
    mask = np.zeros([height, width, len(subset)], dtype=np.uint8)
    for i, j in enumerate(range(subset[0], subset[1])):
        start = subset[j]['all_points'][:-1]
        rr, cc = skimage.draw.polygon(start[:, 1], start[:, 0])
        mask[rr, cc, i] = 1
    return mask




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
    IMAGE_RESIZE_MODE = "pad64"
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
    def evaluate_image(self, image_id):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(self.dataset, self.cfg_GT,
                                   self.dataset.image_ids[image_id])
        if gt_mask.size == 0:
            return 1
        if np.max(gt_mask) == 0:
            return 1
        molded_images = np.expand_dims(
            modellib.mold_image(image, self.cfg_GT), 0)
        results = self.model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, P,recall,overlaps =\
            compute_ap(gt_bbox, gt_class_id, gt_mask,\
                        r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=0.5)
        return AP

    def evaluate_model(self, limit):
        if limit == -1:
            limit = len(self.dataset.image_ids)
        with Pool(4) as p:
            precisions = p.map(self.evaluate_image, range(limit))
        return np.mean(precisions)
    def evaluate_model_old(self, limit):
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
            results = self.model.detect([image], verbose=0)
            r = results[0]
            print(r['masks'].shape)
            # Compute AP

            AP, P,recall,overlaps =\
                compute_ap(gt_bbox, gt_class_id, gt_mask,\
                            r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=0.5)
            precisions.append(AP)
            print("Precision: ",np.max(P)) 

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
# 
            # Plot the predicted mask
            
            for i in range(r['masks'].shape[-1]):
                mask = r['masks'][..., i]
                for contour in measure.find_contours(mask, 0.5):
                    ax.plot(contour[:, 1], contour[:, 0], '-r', linewidth=2)
# 
        #     # Save the plot to a file
            filename = os.path.join(
                self.output_folder, f'image_{image_id}.png')
            plt.savefig(filename)
            plt.close()
        return np.mean(precisions)
        # print(np.mean(precisions))


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
    print("Mean Precision: ", np.mean(results))