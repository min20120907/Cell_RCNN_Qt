import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import cv2
from CustomCroppingDataset import CustomCroppingDataset
from mrcnn.config import Config
from mrcnn.utils import Dataset
import mrcnn.model as modellib
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt  # Add this import statement

import matplotlib.patches as patches
from skimage import measure
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

class EvalImage():
    def __init__(self, dataset, output_folder):
        self.dataset = dataset
        self.output_folder = output_folder
        self.cfg_GT = GTConfig()

    def convert_pixels_to_inches(self, pixels):
        dpi = 300
        inches = pixels / dpi
        return inches

    def evaluate_model(self, limit):
        # Existing code
        precisions = []
        if limit == -1:
            limit = len(self.dataset.image_ids)
        for image_id in range(limit):
            # Load image and ground truth data
            try:
                image, _, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(self.dataset, self.cfg_GT,
                                           self.dataset.image_ids[image_id])
            except IndexError:
                print(f'IndexError: {self.dataset.image_ids}')
                image, _, gt_class_id, gt_bbox, gt_mask =\
                    modellib.load_image_gt(self.dataset, self.cfg_GT,
                                           self.dataset.image_ids[0])
            if gt_mask.size == 0 or np.max(gt_mask) == 0:
                continue

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

            # Save the plot to a file
            filename = os.path.join(
                self.output_folder, f'image_{image_id}.png')
            plt.savefig(filename)
            plt.close()
        return np.mean(precisions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--dataset", required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--output_folder", required=True,
                        help="Path to the output folder")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Number of images to evaluate (default: all)")
    args = parser.parse_args()

    # Load the dataset
    dataset = CustomCroppingDataset()
    dataset.load_custom(args.dataset, "test")
    dataset.prepare()

    # Create the evaluation object
    eval_image = EvalImage(dataset, output_folder=args.output_folder)

    # Evaluate the model
    eval_image.evaluate_model(args.limit)

