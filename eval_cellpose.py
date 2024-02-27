import cellpose.io
import cellpose.models
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Get the list of image files in your current directory
image_files = [f for f in os.listdir("/mnt/1TB-DISK-2/cellpose_dataset/test-1317-1417") if f.endswith(".jpg") or f.endswith(".png")]
iou_results = []
precisions = []
# ... (Previous code for loading Cellpose model and processing images)
iou_thresholds = [0,1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
for threshold in iou_thresholds:
    for image_file in tqdm(image_files):
        # if filename contains _masks, skip
        if "_masks" in image_file:
            continue
        # Load the image and ground truth mask
        image = cv2.imread(os.path.join("/mnt/1TB-DISK-2/cellpose_dataset/test-1317-1417", image_file))
        try:
            mask_gt = cv2.imread(os.path.join("/mnt/1TB-DISK-2/cellpose_dataset/test-1317-1417", os.path.splitext(image_file)[0] + "_masks" + os.path.splitext(image_file)[1]))
        except:
            continue
        if mask_gt is None:
            # print(f"Failed to load image: {image_file}")
            continue  # Skip this iteration of the loop
        # Run Cellpose segmentation
        masks, flows, styles = cellpose.models.CellposeModel(pretrained_model="/home/e814/.cellpose/models/CP_20240222_094942_cropped", gpu=True).eval(
            image,  diameter=100, channels=[0, 0], do_3D=False
        )
        # merge masks and convert to binary
        mask_cellpose = masks

        # Assuming mask_gt and mask_cellpose are already loaded and have the same dimensions

        # Ensure both masks are binary
        mask_gt_binary = np.where(mask_gt >  0,  1,  0)
        # transfer from 3 channels to one channel
        mask_gt_binary = mask_gt_binary[:,:,0]
        mask_cellpose_binary = np.where(mask_cellpose > 0, 1, 0)

        try:
            # Calculate intersection using logical_and
            intersection = np.sum(np.logical_and(mask_gt_binary, mask_cellpose_binary))
        except:
            continue
        # Calculate union
        union = np.sum(np.logical_or(mask_gt_binary, mask_cellpose_binary))

        # Calculate IoU
        iou = intersection / union
        # append iou results into list
        iou_results.append(iou)
        # Print the results
        # print(f"Intersection: {intersection}, Union: {union}, IoU: {iou}")
    # print map at 0.5 threshold
    print(f"mAP at {threshold} threshold: {np.mean(iou_results)}")
    # append precision results into list
    precisions.append(np.mean(iou_results))

# plot the results precision and iou
plt.plot(iou_thresholds, precisions)
plt.xlabel("IoU Threshold")
plt.ylabel("Precision")
plt.title("Precision vs IoU Threshold")
plt.show()

