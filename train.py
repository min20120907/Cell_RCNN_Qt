#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
========
Headless, reproducible trainer for the ELANW+MHSA Mask R-CNN in this repo,
tuned for **RTX 4090 (24 GB) / i7 20-core / 64 GB RAM** and kept
architecture-compatible with ``evaluation.py`` (same backbone / anchors /
classes), so a trained checkpoint loads by name for evaluation and the saved
0.7/0.15/0.15 split is replayable (no train/test leakage).

Classes: ``cell`` (1) + ``chromosome`` (2)  — the repo/checkpoint scheme.

Why a new script (vs trainingThread.py)?
  * no PyQt/Qt-signal coupling -> runs over SSH / in tmux,
  * a single, well-documented, tuned hyper-parameter set,
  * fixed-seed split saved to ``<logs>/dataset_split.json`` for evaluation.

Example
-------
    python train.py --dataset /data/cells --weights coco --logs ./logs
"""

import os
import sys
import json
import time
import argparse

import numpy as np

# --- repo modules (heavy: TF / Ray pulled in here) ---
import tensorflow as tf
import imgaug.augmenters as iaa

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from CustomCroppingDataset import CustomCroppingDataset

# Reproducible split seed — MUST match what evaluation.py expects.
SPLIT_SEED = 42


# ============================================================================
# Hyper-parameters  (the "perfect params")
# ============================================================================
class TrainConfig(Config):
    """Training config. Every *architecture-affecting* field is identical to
    ``evaluation.EvalConfig`` so weights are interchangeable.
    """
    NAME = "cell_mhsa_elanw"
    GPU_COUNT = 1

    # RTX 4090 24 GB: batch 2 at 512² with ResNet101 + backprop is the safe,
    # stable choice (BN is frozen, so a small batch is fine). Use
    # --grad-accum to grow the *effective* batch without more VRAM.
    IMAGES_PER_GPU = 2

    NUM_CLASSES = 1 + 2                 # background + cell + chromosome

    # ---- backbone / anchors (ELANW+MHSA is baked into resnet_graph) ----
    BACKBONE = "resnet101"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.75, 1, 1.33, 1.6]   # 4 anchors/loc -> must match eval
    RPN_ANCHOR_STRIDE = 1
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # ---- masks ----
    USE_MINI_MASK = True               # OOM-safe training targets (56x56)
    MINI_MASK_SHAPE = (56, 56)
    MASK_POOL_SIZE = 14                # matches eval

    # ---- image geometry (CustomCroppingDataset already emits 512² crops) ----
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_RESIZE_MODE = "square"

    # ---- proposals / ROIs (sized for crowded organelle crops) ----
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    PRE_NMS_LIMIT = 6000
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    RPN_NMS_THRESHOLD = 0.7
    TRAIN_ROIS_PER_IMAGE = 256
    ROI_POSITIVE_RATIO = 0.33
    MAX_GT_INSTANCES = 256
    DETECTION_MAX_INSTANCES = 256
    DETECTION_MIN_CONFIDENCE = 0.7     # (inference-only; eval overrides to 0.05)
    DETECTION_NMS_THRESHOLD = 0.3

    # ---- optimisation ----
    LEARNING_RATE = 1e-3               # overridden per stage below
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4               # gamma/beta/layerscale excluded in compile()
    GRADIENT_CLIP_NORM = 5.0
    # Slightly up-weight the mask loss to sharpen boundaries (helps Boundary-F1
    # / Hausdorff at evaluation) without destabilising the box branch.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0, "rpn_bbox_loss": 1.5,
        "mrcnn_class_loss": 1.0, "mrcnn_bbox_loss": 1.5, "mrcnn_mask_loss": 1.2,
    }
    VALIDATION_STEPS = 50
    STEPS_PER_EPOCH = 200              # informational; model.train derives its own


# Default 3-stage transfer-learning schedule (epoch caps are upper bounds;
# EarlyStopping(patience=15) + ReduceLROnPlateau in model.train() govern them).
DEFAULT_SCHEDULE = [
    # (stage name,           layers,  lr,     epoch cap, augmentation key)
    ("Warm-up heads",        "heads", 1e-3,   40,        "light"),
    ("Fine-tune 4+",         "4+",    1e-4,   160,       "full"),
    ("Polish all layers",    "all",   1e-5,   300,       "full"),
]


# ============================================================================
# Augmentation
# ============================================================================
def build_augmenters():
    """Return (light, full) imgaug pipelines suited to microscopy crops."""
    light = iaa.Sequential([
        iaa.Fliplr(0.5), iaa.Flipud(0.5),
        iaa.Sometimes(0.5, iaa.OneOf([
            iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270),
        ])),
    ])
    full = iaa.Sequential([
        iaa.Fliplr(0.5), iaa.Flipud(0.5),
        iaa.SomeOf((0, 3), [
            iaa.OneOf([iaa.Affine(rotate=90), iaa.Affine(rotate=180),
                       iaa.Affine(rotate=270)]),
            iaa.Affine(rotate=(-20, 20)),
            iaa.Affine(scale=(0.8, 1.2)),
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255)),
            iaa.LinearContrast((0.8, 1.2)),
            iaa.Multiply((0.8, 1.2)),          # brightness jitter
        ]),
    ])
    return light, full


# ============================================================================
# Hardware (RTX 4090 / 20 cores / 64 GB)
# ============================================================================
def configure_hardware(num_cores=20):
    os.environ.setdefault("OMP_NUM_THREADS", str(num_cores))
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
    if not gpus:
        print("[HW] WARNING: no GPU detected — training will be extremely slow.")


# ============================================================================
# Dataset: load 'train' folder, seeded 0.7/0.15/0.15 split, persist split ids
# ============================================================================
def _subset_from(full, ids):
    sub = CustomCroppingDataset()
    sub.class_info = list(full.class_info)
    sub.source_class_ids = dict(full.source_class_ids)
    sub.num_classes = full.num_classes
    for new_id, old_id in enumerate(ids):
        info = full.image_info[old_id]
        sub.add_image(
            info.get("source", "cell"),
            image_id=new_id,
            path=info.get("path"),
            image_bytes=info.get("image_bytes"),   # in-RAM crop (by reference)
            crop_key=info.get("crop_key"),
            width=info["width"], height=info["height"],
            polygons=info.get("polygons", []),
            num_ids=info.get("num_ids", []),
        )
    sub.prepare()
    return sub


def load_and_split(dataset_dir, logs_dir, seed=SPLIT_SEED,
                   fractions=(0.7, 0.15, 0.15)):
    """Load the ``train`` folder once, split 0.7/0.15/0.15, save the split.

    The saved ``<logs>/dataset_split.json`` (keyed by stable ``crop_key``) lets
    ``evaluation.py --split-ids`` replay the EXACT held-out test set.
    """
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=20)

    full = CustomCroppingDataset()
    full.load_custom(dataset_dir, "train", cache_dir=None)   # in-memory crops
    full.prepare()

    all_ids = np.array(full.image_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(all_ids)

    n = len(all_ids)
    tr = int(n * fractions[0])
    va = int(n * fractions[1])
    ids = {"train": all_ids[:tr], "val": all_ids[tr:tr + va], "test": all_ids[tr + va:]}

    def keys(arr):
        return [str(full.image_info[i].get("crop_key")
                    or full.image_info[i].get("path")
                    or full.image_info[i].get("id")) for i in arr]

    os.makedirs(logs_dir, exist_ok=True)
    split_path = os.path.join(logs_dir, "dataset_split.json")
    with open(split_path, "w") as f:
        json.dump({
            "seed": int(seed), "fractions": list(fractions),
            "counts": {k: int(len(v)) for k, v in ids.items()},
            "train": keys(ids["train"]), "val": keys(ids["val"]),
            "test": keys(ids["test"]),
        }, f, indent=2, default=str)

    print(f"[Split] {n} crops -> train={len(ids['train'])} "
          f"val={len(ids['val'])} test={len(ids['test'])} (seed={seed})")
    print(f"[Split] saved -> {split_path}")
    return _subset_from(full, ids["train"]), _subset_from(full, ids["val"]), split_path


# ============================================================================
# Lightweight headless callbacks (no Qt)
# ============================================================================
class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        dt = time.time() - getattr(self, "_t0", time.time())
        parts = [f"epoch {epoch + 1}", f"{dt:.0f}s"]
        for k in ("loss", "val_loss", "val_mAP"):
            if k in logs:
                parts.append(f"{k}={logs[k]:.4f}")
        print("  [log] " + " | ".join(parts))


class MeanAPCallback(tf.keras.callbacks.Callback):
    """Headless mAP@0.5 on the val set every ``every`` epochs.

    Loads the most recent checkpoint into a batch-1 inference model and writes
    ``val_mAP`` into ``logs`` (so EpochLogger prints it; does not drive
    checkpointing — ModelCheckpoint still selects on val_loss).
    """
    def __init__(self, train_model, inference_model, dataset, every=5, limit=100):
        super().__init__()
        self.train_model = train_model
        self.inference_model = inference_model
        self.dataset = dataset
        self.every = every
        self.limit = min(limit, len(dataset.image_ids))
        self._ids = list(dataset.image_ids)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every != 0:
            return
        try:
            self.inference_model.load_weights(self.train_model.find_last(), by_name=True)
        except Exception as e:
            print(f"  [mAP] skip (no checkpoint yet): {e}")
            return
        rng = np.random.RandomState(epoch)           # vary sample, deterministically
        ids = rng.choice(self._ids, self.limit, replace=False)
        aps = []
        for image_id in ids:
            image, _, gcid, gbox, gmask = modellib.load_image_gt(
                self.dataset, self.inference_model.config, image_id)
            if gmask.shape[-1] == 0:
                continue
            r = self.inference_model.detect([image], verbose=0)[0]
            ap, _, _, _ = utils.compute_ap(
                gbox, gcid, gmask, r["rois"], r["class_ids"], r["scores"], r["masks"],
                iou_threshold=0.5)
            aps.append(ap)
        mAP = float(np.mean(aps)) if aps else 0.0
        if logs is not None:
            logs["val_mAP"] = mAP
        print(f"  [mAP] epoch {epoch + 1}: val mAP@0.5 = {mAP:.4f} "
              f"(over {len(aps)} imgs)")


# ============================================================================
# Weights
# ============================================================================
HEAD_EXCLUDE = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox",
                "mrcnn_mask", "rpn_model"]


def resolve_weights(model, weights_arg, logs_dir, exclude_heads):
    """Return (path, exclude_list). Supports 'coco' / 'imagenet' / 'last' / <path>."""
    w = weights_arg.lower()
    if w == "coco":
        path = os.path.join(logs_dir, "mask_rcnn_coco.h5")
        if not os.path.exists(path):
            print("[Weights] downloading COCO weights ...")
            utils.download_trained_weights(path)
        return path, HEAD_EXCLUDE                 # always exclude for COCO init
    if w == "imagenet":
        return model.get_imagenet_weights(), HEAD_EXCLUDE
    if w == "last":
        return model.find_last(), []              # resume -> load everything
    # explicit path
    return weights_arg, (HEAD_EXCLUDE if exclude_heads else [])


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description="Headless ELANW+MHSA Mask R-CNN trainer")
    ap.add_argument("--dataset", required=True,
                    help="dataset root (must contain a 'train' folder of via_region_*.json)")
    ap.add_argument("--weights", default="coco",
                    help="'coco' | 'imagenet' | 'last' | /path/to/weights.h5")
    ap.add_argument("--logs", default="./logs", help="model_dir for checkpoints/logs")
    ap.add_argument("--exclude-heads", action="store_true",
                    help="exclude class/bbox/mask/rpn layers (set when init from a "
                         "non-cell checkpoint; ignored for 'coco'/'last')")
    ap.add_argument("--images-per-gpu", type=int, default=None,
                    help="override IMAGES_PER_GPU (default 2 for a 4090)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient accumulation steps (effective batch multiplier)")
    ap.add_argument("--rois", type=int, default=None,
                    help="override TRAIN_ROIS_PER_IMAGE (default 256)")
    ap.add_argument("--map-every", type=int, default=5,
                    help="compute val mAP every N epochs")
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    # optional per-stage epoch overrides (heads / 4+ / all)
    ap.add_argument("--epochs", type=int, nargs=3, default=None,
                    metavar=("HEADS", "FOURPLUS", "ALL"),
                    help="epoch caps per stage (default 40 160 300)")
    args = ap.parse_args()

    configure_hardware(num_cores=20)

    # ---- config (with CLI overrides) ----
    config = TrainConfig()
    if args.images_per_gpu:
        config.IMAGES_PER_GPU = args.images_per_gpu
    if args.rois:
        config.TRAIN_ROIS_PER_IMAGE = args.rois
    config.__init__()                               # recompute derived (BATCH_SIZE, ...)
    config.display()

    # ---- data ----
    dataset_train, dataset_val, split_path = load_and_split(
        args.dataset, args.logs, seed=args.split_seed)

    # ---- models ----
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)

    class InferenceConfig(TrainConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        USE_MINI_MASK = False                       # sharp masks for mAP
    infer_model = modellib.MaskRCNN(mode="inference", config=InferenceConfig(),
                                    model_dir=args.logs)

    weights_path, exclude = resolve_weights(model, args.weights, args.logs,
                                            args.exclude_heads)
    print(f"[Weights] loading {weights_path} (exclude={len(exclude)} layer groups)")
    model.load_weights(weights_path, by_name=True, exclude=exclude)

    # ---- callbacks ----
    callbacks = [
        MeanAPCallback(model, infer_model, dataset_val,
                       every=args.map_every, limit=100),
        EpochLogger(),
    ]

    # ---- schedule ----
    light_aug, full_aug = build_augmenters()
    aug_map = {"light": light_aug, "full": full_aug}
    schedule = list(DEFAULT_SCHEDULE)
    if args.epochs:
        for i, ep in enumerate(args.epochs):
            name, layers, lr, _, augk = schedule[i]
            schedule[i] = (name, layers, lr, ep, augk)

    for stage_no, (name, layers, lr, epochs, augk) in enumerate(schedule, 1):
        print("\n" + "=" * 70)
        print(f" STAGE {stage_no}/{len(schedule)} — {name} "
              f"| layers={layers} | LR={lr} | epoch cap={epochs}")
        print("=" * 70)
        model.config.LEARNING_RATE = lr
        model.train(
            dataset_train, dataset_val,
            epochs=epochs,
            layers=layers,
            augmentation=aug_map[augk],
            custom_callbacks=callbacks,
            gradient_accumulation_steps=args.grad_accum,
            verbose=0,
        )

    print("\n[Done] best checkpoint (by val_loss) is under:", model.log_dir)
    print("[Done] evaluate it with:")
    print(f"    python evaluation.py --dataset {args.dataset} "
          f"--weights <best.h5> --split-ids {split_path} --eval-split test")


if __name__ == "__main__":
    main()
