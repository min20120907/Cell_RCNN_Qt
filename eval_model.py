import numpy as np
from mrcnn.config import Config
from mrcnn.utils import compute_ap
import mrcnn.model as modellib
from trainingThread import CustomDataset, trainingThread
import argparse

class EvalImage():
    def __init__(self, dataset, model, cfg):
        self.dataset = dataset
        self.model = model
        self.cfg = cfg

    def evaluate_model(self, limit=50):
        APs = []
        precisions_dict = {}
        recall_dict = {}

        for index, image_id in enumerate(self.dataset.image_ids):
            if index >= limit:
                break

            image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt_ray(self.dataset, self.cfg, image_id)
            sample = np.expand_dims(image, 0)
            yhat = self.model.detect(sample, verbose=1)
            r = yhat[0]

            
            AP, precisions, recalls, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            precisions_dict[image_id] = np.mean(precisions)
            recall_dict[image_id] = np.mean(recalls)
            APs.append(AP)

        mAP = np.mean(APs)
        return mAP, precisions_dict, recall_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--dataset", required=True, help="Path to the dataset directory")
    parser.add_argument("--workdir", required=True, help="Path to the working directory")
    parser.add_argument("--weight_path", required=True, help="Path to the weight file")
    parser.add_argument("--limit", type=int, default=50, help="Number of images to evaluate (default: 50)")
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
        IMAGE_RESIZE_MODE = "none"
        VALIDATION_STEPS = 50

    model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(), model_dir=WORK_DIR + "/logs")
    model.load_weights(weight_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    eval = EvalImage(dataset_val, model, InferenceConfig())

    mAP, precisions_dict, recall_dict = eval.evaluate_model(limit=LIMIT)
    print("Mean Average Precision (mAP):", mAP)
    print("Precisions:", precisions_dict)
    print("Recalls:", recall_dict)
