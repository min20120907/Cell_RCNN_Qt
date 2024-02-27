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
iou_thresholds = [0.1,  0.3,  0.5,  0.6,  0.7,  0.8,  0.9]
precisions = []


    
for image_file in tqdm(image_files):
    if "_masks" in image_file:
        continue
    image = cv2.imread(os.path.join("/mnt/1TB-DISK-2/cellpose_dataset/test-1317-1417", image_file))
    try:
        mask_gt = cv2.imread(os.path.join("/mnt/1TB-DISK-2/cellpose_dataset/test-1317-1417", os.path.splitext(image_file)[0] + "_masks" + os.path.splitext(image_file)[1]))
    except:
        continue
    if mask_gt is None:
        continue
    masks, flows, styles = cellpose.models.CellposeModel(pretrained_model="/home/e814/.cellpose/models/CP_20240220_094549", gpu=True).eval(
        image, diameter=100, channels=[0,  0], do_3D=False
    )
    mask_cellpose = masks
    mask_gt_binary = np.where(mask_gt >  0,  1,  0)[:,:,0]
    mask_cellpose_binary = np.where(mask_cellpose >  0,  1,  0)
    # split into multiple individual masks and sort it by middle point coordinates
    
    # calculate IoU for each pair of masks
   
    try:
        intersection = np.sum(np.logical_and(mask_gt_binary, mask_cellpose_binary))
    except:
        continue
    union = np.sum(np.logical_or(mask_gt_binary, mask_cellpose_binary))
    iou = intersection / union
    iou_results.append(iou)

        
for threshold in iou_thresholds:
    iou_results = [iou if iou > threshold else  0 for iou in iou_results]
    # Calculate precision for the current threshold
    if len(iou_results) >  0:
        precision = np.mean(iou_results)
    else:
        precision =  0  # Or handle as you see fit if no IoU meets the threshold

    print(f"mAP at {threshold} threshold: {precision}")
    precisions.append(precision)

# Plot the results precision and iou
plt.plot(iou_thresholds, precisions)
plt.xlabel("IoU Threshold")
plt.ylabel("Precision")
plt.title("Precision vs IoU Threshold (Uncropped)")
plt.show()

