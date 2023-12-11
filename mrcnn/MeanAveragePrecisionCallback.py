import traceback
from tensorflow.keras.callbacks import Callback
from mrcnn.model import MaskRCNN, Dataset, mold_image, load_image_gt_remote, load_image_gt_ray, load_image_gt
from mrcnn.utils import compute_matches, compute_ap
import numba as nb
import numpy as np
import ray
from tqdm import tqdm
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

@nb.jit(nopython=True, parallel=True)
def compute_ap_jit(gt_boxes, gt_class_ids, gt_masks,
                   pred_boxes, pred_class_ids, pred_scores, pred_masks,
                   iou_threshold=0.5):
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    # Compute precision and recall at each prediction box step
    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase
    for i in nb.prange(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps
@ray.remote
def process_image(image, inference_model):
    molded_images = np.expand_dims(mold_image(image, inference_model.config), 0)
    results = inference_model.detect(molded_images, verbose=1)
    r = results[0]
    return r
class MeanAveragePrecisionCallback(Callback):
    def __init__(self, train_model: MaskRCNN, inference_model: MaskRCNN, dataset: Dataset,
                 calculate_map_at_every_X_epoch=5, dataset_limit=None,
                 verbose=1):
        super().__init__()
        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.calculate_map_at_every_X_epoch = calculate_map_at_every_X_epoch
        self.dataset_limit = len(self.dataset.image_ids)
        if dataset_limit is not None:
            self.dataset_limit = dataset_limit
        self.dataset_image_ids = self.dataset.image_ids.copy()

        self._verbose_print = print if verbose > 0 else lambda *a, **k: None

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 2 and (epoch+1) % self.calculate_map_at_every_X_epoch == 0:
            self._verbose_print("Calculating mAP...")
            self._verbose_print("Dataset Limit: ", self.dataset_limit)
            self._load_weights_for_model()
            try:
                mAPs = self._calculate_mean_average_precision()
                mAP = np.mean(mAPs)
                if logs is not None:
                    logs["val_mean_average_precision"] = mAP
                self._verbose_print("mAP at epoch {0} is: {1}".format(epoch + 1, mAP))
            except Exception as e:
                print(e)
        super().on_epoch_end(epoch, logs)

    def _load_weights_for_model(self):
        last_weights_path = self.train_model.find_last()
        self._verbose_print("Loaded weights for the inference model (last checkpoint of the train model): {0}".format(
            last_weights_path))
        self.inference_model.load_weights(last_weights_path, by_name=True)

    def _calculate_mean_average_precision(self):
        tf.compat.v1.global_variables_initializer()
        mAPs = []

        np.random.shuffle(self.dataset_image_ids)
        futures = []
        for image_id in tqdm(self.dataset_image_ids[:self.dataset_limit]):
            future = load_image_gt_remote.remote(self.dataset, self.inference_model.config, image_id)
            futures.append(future)
        results = ray.get(futures)
        for image, image_meta, gt_class_id, gt_bbox, gt_mask in results:
            molded_images = np.expand_dims(mold_image(image, self.inference_model.config), 0)

            detection_results = self.inference_model.detect(molded_images, verbose=1)
            r = detection_results[0]
            print("Score: ", r["scores"])
            mAP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                        r["class_ids"], r["scores"], r['masks'])
            mAPs.append(mAP)

        return np.array(mAPs)

    # def on_train_batch_end(self, batch, logs=None):
    #     if batch > 0 and batch % self.calculate_map_at_every_X_epoch == 0:
    #         self._load_weights_for_model()
# 
    #         # Get the input data of the current batch
# 
    #         try:
    #             # Process the input data to obtain the current batch image
    #             current_batch_image,_,gt_bbox, gt_class_id, gt_mask = load_image_gt(self.dataset, self.inference_model.config, self.dataset_image_ids[batch*self.inference_model.config.BATCH_SIZE])
# 
    #             # Perform further operations on the current batch image as needed
    #             molded_images = np.expand_dims(mold_image(current_batch_image, self.inference_model.config), 0)
    #             results = self.inference_model.detect(molded_images, verbose=1)
    #             r = results[0]
# 
    #             # Compute the accuracy for the current batch image
    #             AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
    #                                      r["class_ids"], r["scores"], r['masks'])
# 
    #             # Assign the accuracy value to logs if required
    #             if logs is not None:
    #                 logs["val_accuracy"] = AP
# 
    #             self._verbose_print("Accuracy at batch {0} is: {1}".format(batch + 1, AP))
    #         except Exception as e:
    #             print(e)
    #             traceback.print_exc()
# 
    #     super().on_train_batch_end(batch, logs)

    def _calculate_mean_average_precision_old(self):
        
        mAPs = []
        # Use a random subset of the data when a limit is defined
        np.random.shuffle(self.dataset_image_ids)       
        futures = []
        for image_id in tqdm(self.dataset_image_ids[:self.dataset_limit]):
            future = load_image_gt_remote.remote(self.dataset, self.inference_model.config, image_id)
            futures.append(future)
        results = ray.get(futures)
        with tf.compat.v1.Session() as sess:
            tf.config.run_functions_eagerly(True)
            tf.compat.v1.enable_eager_execution()
            print("Successfully Init Eager Mode")      
            for image, image_meta, gt_class_id, gt_bbox, gt_mask in results:
                future = process_image.remote(image, self.inference_model)
            results = ray.get(future)
            r = results[0]
            # Compute mAP - VOC uses IoU 0.5
            AP, _, _, _ = compute_ap_jit(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                           r["class_ids"], r["scores"], r['masks'])
            mAPs.append(AP)        
        return np.array(mAPs)


