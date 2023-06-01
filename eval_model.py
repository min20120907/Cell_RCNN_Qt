import numpy as np
from mrcnn.config import Config
from mrcnn.utils import compute_ap, Dataset
import mrcnn.model as modellib
import ray
import os
import argparse
import skimage
import json
from tqdm import tqdm
def generate_mask_subset(args):
    height, width, subset = args
    mask = np.zeros([height, width, len(subset)], dtype=np.uint8)
    for i, j in enumerate(range(subset[0], subset[1])):
        start = subset[j]['all_points'][:-1]
        rr, cc = skimage.draw.polygon(start[:, 1], start[:, 0])
        mask[rr, cc, i] = 1
    return mask

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
        assert subset in ["train", "val"]
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
class EvalImage():
    def __init__(self, dataset, model, cfg):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg

    def evaluate_model(self, limit):
        APs = []
        if limit==-1:
            limit = len(dataset_val.image_ids)
        for image_id in range(limit):
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =\
                modellib.load_image_gt(dataset_val, self.cfg,
                                       dataset_val.images_ids[image_id])
            molded_images = np.expand_dims(modellib.mold_image(image, self.cfg), 0)
            # make prediction
            results = self.model.detect([image], verbose=0)
            r = results[0]
            # Compute AP
            AP, _,_,_ =\
                compute_ap(gt_bbox, gt_class_id, gt_mask,\
                            r["rois"], r["class_ids"], r["scores"], r['masks'],iou_threshold=0)

            if AP==0:
                AP=1
            print("AP:", AP)
            # print("Precision:", precisions)
            # print("Recall:", recalls)
            # print("Overlap:", overlaps)
            APs.append(AP)      

        mAP = np.mean(APs)
        return mAP
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--dataset", required=True, help="Path to the dataset directory")
    parser.add_argument("--workdir", required=True, help="Path to the working directory")
    parser.add_argument("--weight_path", required=True, help="Path to the weight file")
    parser.add_argument("--limit", type=int, default=-1, help="Number of images to evaluate (default: all)")
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    WORK_DIR = args.workdir
    LIMIT = args.limit
    weight_path = args.weight_path

    dataset_val = CustomDataset()
    dataset_val.load_custom(DATASET_PATH, "val")
    dataset_val.prepare()
    print("Number of Images: ", dataset_val.num_images)
    print("Number of Classes: ", dataset_val.num_classes)
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

