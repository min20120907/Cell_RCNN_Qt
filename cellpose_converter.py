import os
import cv2
import numpy as np
from tqdm import tqdm
from CustomDataset import CustomDataset
def generate_unique_colors(num_colors):
    colors = np.zeros((num_colors,  3), dtype=np.uint8)
    for i in range(num_colors):
        r = int(255 * np.cos(i / num_colors *  2 * np.pi))
        g = int(255 * np.sin(i / num_colors *  2 * np.pi))
        b = int(255 * np.sin(i / num_colors *  2 * np.pi +  2 * np.pi /  3))
        colors[i] = (r, g, b)
    return colors
def save_images_and_masks(dataset, image_folder, mask_folder):
    """
    Save images and masks from a dataset to specified folders.
    
    Args:
        dataset (CustomDataset): The dataset containing the images and masks.
        image_folder (str): Path to the folder where images will be saved.
        mask_folder (str): Path to the folder where masks will be saved.
    """
    # Ensure folders exist
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)

    # Iterate over the dataset
    for idx in tqdm(range(len(dataset.image_ids))):
        # Load image and mask
        image_id = dataset.image_ids[idx]
        image = dataset.load_image(image_id)
        mask,_ = dataset.load_mask(image_id)
        
        # Save image
        image_path = os.path.join(image_folder, f"{image_id}.png")
        cv2.imwrite(image_path, image)

        # Save mask
        mask_path = os.path.join(mask_folder, f"{image_id}_mask.png")
        
        # Define a custom color map for each ROI
        num_rois = mask.shape[-1]
        color_map = np.array(generate_unique_colors(num_rois))

        # Merge all the masks together, and each mask is in a different color
        # Initialize an empty array for the final colored mask
        colored_mask = np.zeros((*mask.shape[:2],   3), dtype=np.uint8)
        # Loop through each mask and apply a color from the color map
        for i in tqdm(range(num_rois)):
            # Create a binary mask for the current ROI
            roi_mask = (mask[..., i] >  0).astype(np.uint8)
            # Assign the color from the color map to the ROI
            colored_mask = colored_mask + np.expand_dims(roi_mask, axis=-1) * color_map[i]

        # Save the merged colored mask
        cv2.imwrite(mask_path, colored_mask)

# Example usage:
dataset = CustomDataset()
dataset.load_custom('/home/e814/Documents/dataset-png', "train")
dataset.prepare()

# Specify where to save the images and masks
image_folder = "/mnt/1TB-DISK-2/cellpose_dataset/images/"
mask_folder = "/mnt/1TB-DISK-2/cellpose_dataset/masks/"

# Call the function to save images and masks
save_images_and_masks(dataset, image_folder, mask_folder)