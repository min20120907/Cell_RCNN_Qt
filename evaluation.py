#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py
=============
Academic-grade Instance Segmentation evaluation for the ELANW+MHSA Matterport
Mask R-CNN model in this repo, scoped to the classes ``cell`` and
``chromosome`` (the scheme used by trainingThread / CustomCroppingDataset).

Design notes
------------
* The **pure metric functions** (Boundary F1-score, Hausdorff Distance, COCO
  precision extraction) depend ONLY on ``numpy`` / ``scipy``.  Heavy
  dependencies (``tensorflow``, ``mrcnn``, ``pycocotools``, ``cv2``,
  ``matplotlib``) are imported lazily inside the functions that need them.
  This lets ``test_evaluation.py`` import and unit-test the metric logic on a
  machine without a GPU / TensorFlow.
* Hardware target: RTX 4090 (24 GB VRAM), i7 (20 cores), 64 GB RAM.  See
  :class:`EvalConfig` and :func:`configure_hardware`.

Author: Senior Research Engineer
"""

import os
import sys
import json
import time
import argparse

import numpy as np
from scipy import ndimage

# ----------------------------------------------------------------------------
# Target classes (STRICTLY these two), matching the trained ELANW+MHSA model:
#   cell = 1, chromosome = 2   (same scheme as trainingThread.CustomConfig and
#   CustomCroppingDataset.load_custom).  Evaluation is strictly scoped to these
#   via COCOeval.params.catIds.
# ----------------------------------------------------------------------------
CLASS_NAMES = {1: "cell", 2: "chromosome"}
CAT_IDS = sorted(CLASS_NAMES.keys())            # [1, 2]
EVAL_SEEDS = [42, 1024, 2026]                   # 3-times random evaluation

# Optional soft import of mrcnn so that the *module* still imports on a host
# without TensorFlow (the metric tests only need numpy / scipy).
try:
    from mrcnn.config import Config as _Config
    from mrcnn import utils as _mrcnn_utils
    _HAS_MRCNN = True
except Exception:                                # pragma: no cover - env dependent
    _Config = object
    _mrcnn_utils = None
    _HAS_MRCNN = False


# ============================================================================
# SECTION 1 — Pure, unit-testable metric functions  (numpy / scipy only)
# ============================================================================
def _as_bool_mask(mask):
    """Validate and coerce a 2-D array into a boolean mask."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {arr.shape}")
    return arr.astype(bool)


def _check_same_shape(a, b):
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")


def extract_boundary(mask, connectivity=1):
    """Return the 1-pixel-wide inner boundary of a binary mask.

    boundary = mask AND NOT erode(mask)

    Parameters
    ----------
    mask : 2-D array (bool / 0-1)
    connectivity : 1 -> 4-neighbourhood, 2 -> 8-neighbourhood

    Returns
    -------
    boundary : 2-D bool array, same shape as ``mask``.
    """
    m = _as_bool_mask(mask)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    structure = ndimage.generate_binary_structure(2, connectivity)
    eroded = ndimage.binary_erosion(m, structure=structure, border_value=0)
    return m & ~eroded


def _surface_distances(gt_mask, pred_mask, connectivity=1):
    """Symmetric boundary-to-boundary Euclidean distances.

    Returns
    -------
    (d_pred_to_gt, d_gt_to_pred) : two 1-D arrays of per-boundary-pixel
        nearest-neighbour distances, or ``None`` if either boundary is empty.
    """
    gt_b = extract_boundary(gt_mask, connectivity)
    pred_b = extract_boundary(pred_mask, connectivity)
    if not gt_b.any() or not pred_b.any():
        return None
    # distance_transform_edt gives, for every pixel, the Euclidean distance to
    # the nearest *background* (False) pixel.  Feeding ``~boundary`` turns the
    # boundary into the background, so the result is "distance to the nearest
    # boundary pixel" everywhere.
    dt_to_gt = ndimage.distance_transform_edt(~gt_b)
    dt_to_pred = ndimage.distance_transform_edt(~pred_b)
    d_pred_to_gt = dt_to_gt[pred_b]     # each predicted boundary px -> nearest GT boundary
    d_gt_to_pred = dt_to_pred[gt_b]     # each GT boundary px        -> nearest pred boundary
    return d_pred_to_gt, d_gt_to_pred


def boundary_f1_score(gt_mask, pred_mask, tolerance=2.0, connectivity=1):
    """Boundary F1-score (a.k.a. BF score / contour F-measure).

    A predicted boundary pixel counts as matched if it lies within
    ``tolerance`` pixels of any GT boundary pixel (and vice-versa).

        precision = matched predicted-boundary px / all predicted-boundary px
        recall    = matched GT-boundary px        / all GT-boundary px
        BF1       = 2 * P * R / (P + R)

    Edge cases
    ----------
    * both masks empty        -> 1.0 (they agree there is no object)
    * exactly one mask empty  -> 0.0
    * perfectly overlapping   -> 1.0
    * disjoint & far apart    -> 0.0
    """
    gt = _as_bool_mask(gt_mask)
    pred = _as_bool_mask(pred_mask)
    _check_same_shape(gt, pred)

    gt_b = extract_boundary(gt, connectivity)
    pred_b = extract_boundary(pred, connectivity)
    gt_n, pred_n = int(gt_b.sum()), int(pred_b.sum())

    if gt_n == 0 and pred_n == 0:
        return 1.0
    if gt_n == 0 or pred_n == 0:
        return 0.0

    d_pred_to_gt, d_gt_to_pred = _surface_distances(gt, pred, connectivity)
    precision = float(np.mean(d_pred_to_gt <= tolerance))
    recall = float(np.mean(d_gt_to_pred <= tolerance))

    if (precision + recall) == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def hausdorff_distance(gt_mask, pred_mask, percentile=None, connectivity=1):
    """(Optionally percentile) Hausdorff Distance between mask boundaries.

    ``percentile=None`` -> classic (maximum) Hausdorff Distance.
    ``percentile=95``   -> robust HD95 (common in microscopy / medical papers).

    Edge cases
    ----------
    * perfectly overlapping masks -> 0.0
    * both masks empty            -> 0.0
    * exactly one mask empty      -> +inf (undefined / worst case)
    * disjoint masks              -> the largest boundary separation (finite)
    """
    gt = _as_bool_mask(gt_mask)
    pred = _as_bool_mask(pred_mask)
    _check_same_shape(gt, pred)

    gt_has = extract_boundary(gt, connectivity).any()
    pred_has = extract_boundary(pred, connectivity).any()
    if not gt_has and not pred_has:
        return 0.0
    if not gt_has or not pred_has:
        return float("inf")

    d_pred_to_gt, d_gt_to_pred = _surface_distances(gt, pred, connectivity)
    if percentile is None:
        return float(max(d_pred_to_gt.max(), d_gt_to_pred.max()))
    return float(max(np.percentile(d_pred_to_gt, percentile),
                     np.percentile(d_gt_to_pred, percentile)))


def mask_iou(a, b):
    """Intersection-over-Union of two binary masks (pure numpy)."""
    a = _as_bool_mask(a)
    b = _as_bool_mask(b)
    _check_same_shape(a, b)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0          # both empty -> identical
    return float(inter) / float(union)


def extract_map_at_iou(coco_eval, iou_thr, area_idx=0, maxdet_idx=-1):
    """Extract mAP at a specific IoU threshold from a COCOeval result.

    COCOeval stores ``eval['precision']`` with shape
    ``[T, R, K, A, M]`` (IoU thresholds, Recall thresholds, Categories,
    Area ranges, MaxDets).  COCO's default ``iouThrs`` already contains 0.75
    and 0.90, so we simply slice the matching threshold and average the valid
    precision values over recall thresholds and categories.

    This helper is pure-numpy and therefore unit-testable with a tiny fake
    ``coco_eval`` object.
    """
    iou_thrs = np.asarray(coco_eval.params.iouThrs)
    precision = np.asarray(coco_eval.eval["precision"])
    t = int(np.argmin(np.abs(iou_thrs - iou_thr)))
    # [R, K] slice at the chosen IoU / area / maxdet
    s = precision[t, :, :, area_idx, maxdet_idx]
    valid = s[s > -1]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


# ============================================================================
# SECTION 2 — COCO mAP / AR (lazy pycocotools)
# ============================================================================
def _binary_mask_to_rle(binary_mask):
    """Encode an HxW binary mask to a JSON-friendly COCO RLE dict."""
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _rle_area_bbox(rle):
    from pycocotools import mask as mask_utils
    enc = dict(rle)
    if isinstance(enc["counts"], str):
        enc = {"size": enc["size"], "counts": enc["counts"].encode("ascii")}
    area = float(mask_utils.area(enc))
    bbox = mask_utils.toBbox(enc).tolist()      # [x, y, w, h]
    return area, bbox


def build_coco_groundtruth(images_meta, gt_annotations):
    """Assemble a COCO-format ground-truth dict (in memory)."""
    return {
        "info": {"description": "cell/chromosome instance-seg evaluation"},
        "licenses": [],
        "images": images_meta,
        "annotations": gt_annotations,
        "categories": [{"id": cid, "name": CLASS_NAMES[cid], "supercategory": "organelle"}
                       for cid in CAT_IDS],
    }


def run_coco_eval(gt_dict, dt_list, cat_ids=CAT_IDS, img_ids=None):
    """Run pycocotools COCOeval (segm) and return a metrics dict.

    Strictly restricts evaluation to ``cat_ids`` via ``coco_eval.params.catIds``
    and (optionally) to a sampled subset via ``coco_eval.params.imgIds``.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    if len(dt_list) == 0:
        # No detections at all -> everything is zero.
        return {k: 0.0 for k in
                ("mAP", "mAP50", "mAP75", "mAP90", "mAP_small", "mAP_medium",
                 "mAP_large", "AR1", "AR10", "AR100")}

    coco_dt = coco_gt.loadRes(dt_list)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.catIds = list(cat_ids)              # <-- strict 2-class scope
    if img_ids is not None:
        coco_eval.params.imgIds = list(img_ids)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    s = coco_eval.stats   # standard 12-vector for segm
    return {
        "mAP":        float(s[0]),    # AP @[.50:.95]
        "mAP50":      float(s[1]),    # AP @.50
        "mAP75":      float(s[2]),    # AP @.75
        "mAP90":      extract_map_at_iou(coco_eval, 0.90),   # boundary stability
        "mAP_small":  float(s[3]),
        "mAP_medium": float(s[4]),
        "mAP_large":  float(s[5]),
        "AR1":        float(s[6]),
        "AR10":       float(s[7]),
        "AR100":      float(s[8]),
    }


# ============================================================================
# SECTION 3 — Hardware-optimised inference config (RTX 4090 / 20c / 64GB)
# ============================================================================
class EvalConfig(_Config):
    """Inference config tuned for a single RTX 4090 (24 GB).

    Every architecture-affecting field is kept IDENTICAL to
    ``trainingThread.CustomConfig`` so the trained weights — including the
    Stage-4/5 ELANW + MHSA layers baked into ``resnet_graph`` for
    ``BACKBONE='resnet101'`` — load by name and reconstruct exactly.
    """
    NAME = "cell_mhsa_elanw_eval"
    GPU_COUNT = 1
    # RTX 4090 24 GB easily fits a large inference batch at 512x512.  Batching
    # the detector keeps the GPU saturated; 4 is a safe high-throughput value
    # (raise to 8 if VRAM headroom allows at your resolution).
    IMAGES_PER_GPU = 4

    NUM_CLASSES = 1 + 2                 # background + cell + chromosome

    # --- IDENTICAL to training so weights (incl. ELANW/MHSA) load by name ---
    BACKBONE = "resnet101"             # -> resnet_graph adds ELANW + MHSA
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_RATIOS = [0.75, 1, 1.33, 1.6]
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    MASK_POOL_SIZE = 14
    RPN_NMS_THRESHOLD = 0.7

    # Sharp full-resolution masks (mini-mask only affects training targets,
    # but we pin it off so boundary metrics use crisp masks).
    USE_MINI_MASK = False

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_RESIZE_MODE = "square"

    # For a faithful COCO PR-curve we keep a LOW score floor; raising this
    # only throws away low-score-but-correct detections and depresses mAP.
    DETECTION_MIN_CONFIDENCE = 0.05
    DETECTION_MAX_INSTANCES = 256       # matches training MAX_GT_INSTANCES
    DETECTION_NMS_THRESHOLD = 0.3


def configure_hardware(num_cores=20):
    """Maximise CPU/GPU utilisation; enable GPU memory growth.

    Returns the imported ``tensorflow`` module (lazy import).
    """
    os.environ.setdefault("OMP_NUM_THREADS", str(num_cores))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(num_cores))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "4")

    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(4)
    except Exception:
        pass
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    print(f"[HW] GPUs visible: {len(gpus)} | intra-op threads: {num_cores}")
    return tf


def compute_model_complexity(model):
    """Return (#params, FLOPs).  FLOPs via keras-flops when available."""
    params = int(model.keras_model.count_params())
    flops = None
    try:
        from keras_flops import get_flops
        flops = int(get_flops(model.keras_model, batch_size=1))
    except Exception as e:                       # pragma: no cover
        print(f"[FLOPs] keras-flops unavailable ({e}); reporting params only.")
    return params, flops


# ============================================================================
# SECTION 4 — Dataset (the SAME CustomCroppingDataset used for training) +
#             inference
# ============================================================================
def _make_subset(full_dataset, sub_ids):
    """Build a CustomCroppingDataset containing only ``sub_ids`` (re-indexed).

    Mirrors ``trainingThread.split_dataset.make_subset`` and crucially carries
    ``image_bytes`` so the in-RAM crops survive the split (by reference -> no
    memory blow-up).
    """
    from CustomCroppingDataset import CustomCroppingDataset
    sub = CustomCroppingDataset()
    sub.class_info = list(full_dataset.class_info)
    sub.source_class_ids = dict(full_dataset.source_class_ids)
    sub.num_classes = full_dataset.num_classes
    for new_id, old_id in enumerate(sub_ids):
        info = full_dataset.image_info[old_id]
        sub.add_image(
            info.get("source", "cell"),
            image_id=new_id,
            path=info.get("path"),
            image_bytes=info.get("image_bytes"),
            crop_key=info.get("crop_key"),
            width=info["width"],
            height=info["height"],
            polygons=info.get("polygons", []),
            num_ids=info.get("num_ids", []),
        )
    sub.prepare()
    return sub


def load_eval_dataset(dataset_dir, which_split="test",
                      fractions=(0.7, 0.15, 0.15), split_seed=42,
                      split_ids_path=None):
    """Load the ``train`` folder and split it 0.7/0.15/0.15 (train/val/test).

    Runs evaluation on the held-out portion the model never trained on, through
    the identical ``CustomCroppingDataset`` crop-group -> 512x512 pipeline.

    Two modes (preferred first):

    1. ``split_ids_path`` -> a ``dataset_split.json`` written by
       ``trainingThread.split_dataset``.  Selection is by stable ``crop_key``, so
       the eval set is **byte-for-byte the same held-out crops** used in training
       (zero leakage).  This is the recommended path.
    2. Fallback: seeded ``RandomState(split_seed)`` shuffle reproducing the same
       logic.  Deterministic, but only matches training if both used the same
       seed AND the same crop load order.

    Parameters
    ----------
    which_split    : 'train' | 'val' | 'test'  (which portion to return)
    fractions      : (train, val, test), must sum to ~1.0  (mode 2 only)
    split_seed     : RNG seed for the reproducible split  (mode 2 only)
    split_ids_path : path to the saved split JSON          (mode 1)
    """
    assert which_split in ("train", "val", "test")
    assert abs(sum(fractions) - 1.0) < 1e-6, "fractions must sum to 1.0"

    import ray
    from CustomCroppingDataset import CustomCroppingDataset

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=20)

    full = CustomCroppingDataset()
    # Always load the 'train' folder, then split it ourselves.
    full.load_custom(dataset_dir, "train", cache_dir=None)
    full.prepare()

    # ---- Mode 1: replay the EXACT saved split via stable crop_key ----
    if split_ids_path:
        with open(split_ids_path) as f:
            saved = json.load(f)
        wanted = set(saved[which_split])
        key_to_id = {}
        for iid in full.image_ids:
            k = str(full.image_info[iid].get("crop_key")
                    or full.image_info[iid].get("path")
                    or full.image_info[iid].get("id"))
            key_to_id[k] = iid
        sel = [key_to_id[k] for k in saved[which_split] if k in key_to_id]
        missing = len(wanted) - len(sel)
        print(f"[Split] replayed saved split '{which_split}': matched {len(sel)}"
              f"/{len(wanted)} crops by crop_key"
              + (f" ({missing} not found in current load)" if missing else ""))
        if missing:
            print("[Split] WARNING: some saved crop_keys were not found — "
                  "is the dataset/crop logic identical to training time?")
        return _make_subset(full, sel)

    # ---- Mode 2: seeded reproducible split (fallback) ----
    all_ids = np.array(full.image_ids)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(all_ids)

    n = len(all_ids)
    train_n = int(n * fractions[0])
    val_n = int(n * fractions[1])
    split_ids = {
        "train": all_ids[:train_n],
        "val":   all_ids[train_n:train_n + val_n],
        "test":  all_ids[train_n + val_n:],
    }
    print(f"[Split] seeded split | train folder -> {n} crops | "
          f"train={len(split_ids['train'])} val={len(split_ids['val'])} "
          f"test={len(split_ids['test'])} (seed={split_seed})")
    return _make_subset(full, split_ids[which_split])


def _draw_failure_figure(image, gt_mask, pred_mask, save_path, title):
    """Save a GT-vs-Prediction comparison image for a failure case."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage import measure

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, masks, name, color in (
            (axes[0], gt_mask, "Ground Truth", "lime"),
            (axes[1], pred_mask, "Prediction", "red")):
        ax.imshow(image)
        if masks is not None and masks.shape[-1] > 0:
            for k in range(masks.shape[-1]):
                for c in measure.find_contours(masks[..., k].astype(float), 0.5):
                    ax.plot(c[:, 1], c[:, 0], color=color, linewidth=1.2)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(save_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def run_inference_once(model, dataset, config, output_dir,
                       failure_iou=0.3, max_failure_saves=60):
    """Single GPU pass over the whole dataset.

    Returns a list of per-image records (scalar metrics + COCO entries) plus the
    COCO ground-truth scaffolding.  Inference is run ONCE; the seed loop later
    resamples these cached records, so the (expensive) GPU work is not repeated.
    """
    import mrcnn.model as modellib
    from mrcnn import utils

    ids = list(dataset.image_ids)
    bs = config.BATCH_SIZE

    images_meta, gt_annotations, dt_list, per_image = [], [], [], []
    ann_id = 1
    failure_dir = os.path.join(output_dir, "failure_cases")
    os.makedirs(failure_dir, exist_ok=True)
    n_failures = 0

    # GPU warm-up (exclude from timing).
    warm, *_ = modellib.load_image_gt(dataset, config, ids[0])
    model.detect([warm] * bs, verbose=0)

    i = 0
    while i < len(ids):
        batch_ids = ids[i:i + bs]
        images, gts = [], []
        for iid in batch_ids:
            image, _, gcid, gbox, gmask = modellib.load_image_gt(dataset, config, iid)
            images.append(image)
            gts.append((gcid, gbox, gmask))

        pad = bs - len(images)
        if pad > 0:
            images = images + [images[-1]] * pad

        t0 = time.perf_counter()
        results = model.detect(images, verbose=0)
        dt = time.perf_counter() - t0
        per_img_time = dt / max(1, len(batch_ids))

        for k, iid in enumerate(batch_ids):
            coco_id = i + k
            r = results[k]
            gcid, gbox, gmask = gts[k]
            image = images[k]
            h, w = image.shape[:2]
            images_meta.append({"id": coco_id, "width": int(w), "height": int(h)})

            # ---- COCO ground-truth annotations ----
            for j in range(gmask.shape[-1]):
                rle = _binary_mask_to_rle(gmask[..., j])
                area, bbox = _rle_area_bbox(rle)
                gt_annotations.append({
                    "id": ann_id, "image_id": coco_id,
                    "category_id": int(gcid[j]), "segmentation": rle,
                    "area": area, "bbox": bbox, "iscrowd": 0})
                ann_id += 1

            # ---- COCO detections ----
            for j in range(r["masks"].shape[-1]):
                rle = _binary_mask_to_rle(r["masks"][..., j])
                _, bbox = _rle_area_bbox(rle)
                dt_list.append({
                    "image_id": coco_id, "category_id": int(r["class_ids"][j]),
                    "segmentation": rle, "score": float(r["scores"][j]),
                    "bbox": bbox})

            # ---- per-image AP@0.5 (Matterport, for bootstrap stability) ----
            if gmask.shape[-1] > 0:
                ap05, _, _, overlaps = utils.compute_ap(
                    gbox, gcid, gmask,
                    r["rois"], r["class_ids"], r["scores"], r["masks"],
                    iou_threshold=0.5)
            else:
                ap05, overlaps = float("nan"), np.zeros((r["masks"].shape[-1], 0))

            # ---- matched instance-level Boundary-F1 & Hausdorff ----
            bf1s, hds = [], []
            if r["masks"].shape[-1] > 0 and gmask.shape[-1] > 0:
                gt_match, pred_match, _ = utils.compute_matches(
                    gbox, gcid, gmask,
                    r["rois"], r["class_ids"], r["scores"], r["masks"],
                    iou_threshold=0.5)
                for p_idx, g_idx in enumerate(pred_match):
                    if g_idx > -1:
                        g_idx = int(g_idx)
                        bf1s.append(boundary_f1_score(gmask[..., g_idx], r["masks"][..., p_idx]))
                        hds.append(hausdorff_distance(gmask[..., g_idx], r["masks"][..., p_idx],
                                                      percentile=95))

            per_image.append({
                "coco_id": coco_id,
                "ap05": ap05,
                "bf1": float(np.mean(bf1s)) if bf1s else np.nan,
                "hd95": float(np.mean(hds)) if hds else np.nan,
                "infer_time": per_img_time,
                "n_gt": int(gmask.shape[-1]),
            })

            # ---- failure cases: any GT or pred whose best IoU < failure_iou ----
            if n_failures < max_failure_saves and overlaps.size > 0:
                worst_fn = overlaps.max(axis=0).min() if overlaps.shape[0] else 1.0
                worst_fp = overlaps.max(axis=1).min() if overlaps.shape[1] else 1.0
                if min(worst_fn, worst_fp) < failure_iou:
                    save_path = os.path.join(failure_dir, f"fail_{coco_id:04d}.png")
                    try:
                        _draw_failure_figure(image, gmask, r["masks"], save_path,
                                             f"img {coco_id} | worstIoU={min(worst_fn, worst_fp):.2f}")
                        n_failures += 1
                    except Exception as e:
                        print(f"[failure-viz] {e}")

        i += bs
        print(f"  inferred {min(i, len(ids))}/{len(ids)} images", end="\r")

    print(f"\n[Inference] done. {len(per_image)} images, "
          f"{len(dt_list)} detections, {n_failures} failure figures saved.")
    gt_dict = build_coco_groundtruth(images_meta, gt_annotations)
    return per_image, gt_dict, dt_list


# ============================================================================
# SECTION 5 — 3-times random evaluation + aggregation
# ============================================================================
def _nanmean(values):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def evaluate_for_seed(per_image, gt_dict, dt_list, seed):
    """Bootstrap-resample the cached per-image results for one seed.

    Variation across seeds comes from bootstrap resampling of the test images
    (a standard way to report metric stability / confidence in papers).  COCO
    mAP is recomputed on the unique sampled image ids.
    """
    rng = np.random.RandomState(seed)
    n = len(per_image)
    sample_idx = rng.randint(0, n, size=n)             # bootstrap (with replacement)

    bf1 = _nanmean([per_image[i]["bf1"] for i in sample_idx])
    hd95 = _nanmean([per_image[i]["hd95"] for i in sample_idx])
    ap05 = _nanmean([per_image[i]["ap05"] for i in sample_idx])
    times = np.array([per_image[i]["infer_time"] for i in sample_idx], dtype=float)
    fps = float(1.0 / times.mean()) if times.mean() > 0 else float("nan")

    sampled_coco_ids = sorted({per_image[i]["coco_id"] for i in sample_idx})
    coco = run_coco_eval(gt_dict, dt_list, cat_ids=CAT_IDS, img_ids=sampled_coco_ids)

    result = {"seed": seed, "FPS": fps, "BoundaryF1": bf1,
              "Hausdorff95": hd95, "AP50_matterport": ap05}
    result.update(coco)
    return result


def aggregate_mean_std(seed_results):
    """Mean ± Std across the seeds for every numeric metric."""
    keys = [k for k in seed_results[0] if k != "seed"]
    summary = {}
    for k in keys:
        vals = np.array([r[k] for r in seed_results], dtype=float)
        vals = vals[~np.isnan(vals)]
        summary[k] = {
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "std": float(vals.std(ddof=0)) if vals.size else float("nan"),
        }
    return summary


def _print_report(summary, complexity):
    params, flops = complexity
    print("\n" + "=" * 64)
    print(" 3-times Random Evaluation  (cell + chromosome)")
    print("=" * 64)
    order = ["mAP", "mAP50", "mAP75", "mAP90", "mAP_small", "mAP_medium",
             "mAP_large", "AR1", "AR10", "AR100", "BoundaryF1", "Hausdorff95",
             "AP50_matterport", "FPS"]
    for k in order:
        if k in summary:
            print(f"  {k:18s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
    print("-" * 64)
    print(f"  Params            : {params:,}")
    print(f"  FLOPs             : {('%.3f GFLOPs' % (flops/1e9)) if flops else 'N/A'}")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description="cell/chromosome instance-seg evaluation")
    parser.add_argument("--dataset", required=True,
                        help="dataset root (the 'train' folder is loaded and split)")
    parser.add_argument("--weights", required=True, help="path to trained .h5 weights")
    parser.add_argument("--workdir", default="./logs", help="mrcnn model_dir")
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test"],
                        help="which 0.7/0.15/0.15 split of the train folder to evaluate")
    parser.add_argument("--split-ids", default=None,
                        help="path to dataset_split.json saved by training "
                             "(replays the EXACT held-out split via crop_key; "
                             "recommended). If omitted, falls back to --split-seed.")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="seed for the reproducible 0.7/0.15/0.15 split "
                             "(used only when --split-ids is not given)")
    parser.add_argument("--output", default="./eval_results", help="output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=EVAL_SEEDS)
    args = parser.parse_args()

    if not _HAS_MRCNN:
        sys.exit("ERROR: mrcnn/TensorFlow not importable in this environment.")

    os.makedirs(args.output, exist_ok=True)
    tf = configure_hardware(num_cores=20)         # noqa: F841 (side effects)

    import mrcnn.model as modellib
    config = EvalConfig()
    config.display()

    # Load the 'train' folder and split 0.7/0.15/0.15 (same pipeline + split as
    # trainingThread); evaluate the held-out '--eval-split' (default: test).
    # If --split-ids is given, replay the EXACT saved split (no leakage).
    dataset = load_eval_dataset(args.dataset, which_split=args.eval_split,
                                fractions=(0.7, 0.15, 0.15),
                                split_seed=args.split_seed,
                                split_ids_path=args.split_ids)
    print(f"[Dataset] eval split='{args.eval_split}' | "
          f"{dataset.num_images} crops | {dataset.num_classes} classes")

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.workdir)
    model.load_weights(args.weights, by_name=True)
    complexity = compute_model_complexity(model)

    # One GPU pass; the seed loop resamples the cached results.
    per_image, gt_dict, dt_list = run_inference_once(model, dataset, config, args.output)

    seed_results = []
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        res = evaluate_for_seed(per_image, gt_dict, dt_list, seed)
        for k, v in res.items():
            if k != "seed":
                print(f"    {k:18s}: {v:.4f}")
        seed_results.append(res)

    summary = aggregate_mean_std(seed_results)
    _print_report(summary, complexity)

    out = {
        "classes": CLASS_NAMES,
        "seeds": args.seeds,
        "per_seed": seed_results,
        "summary_mean_std": summary,
        "params": complexity[0],
        "flops": complexity[1],
    }
    out_path = os.path.join(args.output, "evaluation_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
