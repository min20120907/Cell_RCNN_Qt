import numpy as np
import os
import argparse
import json
from tqdm import tqdm
import skimage
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import ray
from mrcnn.config import Config
from mrcnn.utils import compute_ap, Dataset
import mrcnn.model as modellib
from concurrent.futures import ThreadPoolExecutor
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
def generate_mask_subset(args):
    height, width, subset = args
    mask = np.zeros([height, width, len(subset)], dtype=np.uint8)
    for i, j in enumerate(range(subset[0], subset[1])):
        start = subset[j]['all_points'][:-1]
        rr, cc = skimage.draw.polygon(start[:, 1], start[:, 0])
        mask[rr, cc, i] = 1
    return mask

def plot_iou_precision_recall(iou_thresholds, precisions_list, recalls_list, title="IoU/Precision/Recall"):
    plt.figure()
    plt.title(title)
    
    # Plot precision and recall curves for each IoU threshold
    for iou_threshold, precisions, recalls in zip(iou_thresholds, precisions_list, recalls_list):
        plt.plot(recalls, precisions, label="IoU Threshold: {}".format(iou_threshold))

        # Add dots to the curve
        plt.scatter(recalls, precisions, marker='o')

    # Set limits and labels
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0, 1])
    plt.xlim([0, 1])

    # Add legend and show plot
    plt.legend()
    plt.show()

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

        images.append({
            'image_id': a['filename'],  # use file name as a unique image id
            'path': image_path,
            'width': width,
            'height': height,
            'polygons': polygons,
            'num_ids': num_ids
        })

    return images
def parallel_inference(image_ids, model, num_sessions=3):
    def inference_worker(image_id):
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, cfg, image_id)
        molded_images = np.expand_dims(modellib.mold_image(image, cfg), 0)
        results = model.detect([image], verbose=0)
        return results[0]

    results = []
    with ThreadPoolExecutor(max_workers=num_sessions) as executor:
        futures = [executor.submit(inference_worker, image_id) for image_id in image_ids]
        for future in tqdm(futures, total=len(futures), desc="Inference"):
            results.append(future.result())
    return results
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

class InferenceConfig(Config):
    NAME = "cell"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # Change this to a lower value, e.g., 1
    NUM_CLASSES = 1 + 3
    USE_MINI_MASK = False
    VALIDATION_STEPS = 50

def calculate_iou(pred_boxes, gt_boxes):
    # Calculate the coordinates of the intersection area
    x1 = np.maximum(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = np.maximum(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = np.minimum(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = np.minimum(pred_boxes[:, 3], gt_boxes[:, 3])

    # Calculate the intersection area
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate the area of the predicted and ground truth boxes
    pred_box_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Calculate the union area
    union_area = pred_box_area + gt_box_area - intersection_area

    # Calculate IoU for each pair of boxes
    iou = intersection_area / union_area

    return iou

def plot_map_recall_curve(mAP_values, recall_values, save_path=None):
    """
    Plot Mean Average Precision (mAP) and Recall curve.

    Args:
    mAP_values (list): List of mAP values at different points.
    recall_values (list): List of Recall values at corresponding points.
    save_path (str, optional): Path to save the plot as an image file. If not provided, the plot will be displayed.

    Returns:
    None
    """
    # Create a figure
    plt.figure(figsize=(8, 6))

    # Plot the mAP/Recall curve
    plt.plot(recall_values, mAP_values, marker='o', linestyle='-')
    plt.xlabel('Recall')
    plt.ylabel('Mean Average Precision (mAP)')
    plt.title('mAP vs. Recall Curve')

    # Annotate points with mAP values
    for i, mAP in enumerate(mAP_values):
        plt.annotate(f'{mAP:.2f}', (recall_values[i], mAP_values[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Show or save the plot
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
class EvalImage():
    def __init__(self, dataset, model, cfg):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg

    def evaluate_model(self, limit):
        APs = []
        recalls = []
        if limit==-1:
            limit = len(dataset_val.image_ids)
        for image_id in range(limit):
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, self.cfg,
                                       dataset_val.image_ids[image_id])
            molded_images = np.expand_dims(modellib.mold_image(image, self.cfg), 0)
            # make prediction
            results = self.model.detect([image], verbose=0)
            r = results[0]
            # 計算 IoU/Precision 圖
            iou_thresholds = [0, 0.01, 0.5, 1.0]
            # Compute AP
            for iou_threshold in iou_thresholds:
                AP, _,recall,_ =\
                    compute_ap(gt_bbox, gt_class_id, gt_mask,\
                                r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=iou_threshold)
            print("AP:", AP)
                # print("IoU:", calculate_iou(overlaps[0],overlaps[1]))
                # print("Precision:", precisions)

            # print("Recall:", r)
            recalls.append(np.mean(recall))
            # print("Overlap:", overlaps)
            APs.append(AP)      

        plot_iou_precision_recall(iou_thresholds, APs, recalls)

        mAP = np.mean(APs)
        return mAP

if __name__ == "__main__":
    # ... (Your existing argparse code)
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--dataset", required=True, help="Path to the dataset directory")
    parser.add_argument("--workdir", required=True, help="Path to the working directory")
    parser.add_argument("--weight_path", required=True, help="Path to the weight file")
    parser.add_argument("--limit", type=int, default=-1, help="Number of images to evaluate (default: all)")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of GPU")    
    args = parser.parse_args()
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set to empty string to use CPU only
    DATASET_PATH = args.dataset
    WORK_DIR = args.workdir
    LIMIT = args.limit
    weight_path = args.weight_path

    dataset_val = CustomDataset()
    dataset_val.load_custom(DATASET_PATH, "val")
    dataset_val.prepare()
    print("Number of Images: ", dataset_val.num_images)
    print("Number of Classes: ", dataset_val.num_classes)
    image_ids = dataset_val.image_ids[:LIMIT]  # Limit the number of images to evaluate
    class InferenceConfig(Config):
        NAME = "cell"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 3
        USE_MINI_MASK = False
        VALIDATION_STEPS = 50
    model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir=WORK_DIR + "/logs")
    model.load_weights(weight_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    eval = EvalImage(dataset_val, model, InferenceConfig())

    mAP = eval.evaluate_model(limit=LIMIT)
    print("Mean Average Precision (mAP):", mAP)

    # plot_map_recall_curve(results[1], results[2], save_path="mAP_Recall_Curve.png")
    # print("recall curve finished")

