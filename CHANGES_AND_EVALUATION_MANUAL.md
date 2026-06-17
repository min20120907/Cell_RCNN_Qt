# Cell R-CNN (ELANW+MHSA) — Changes & Evaluation Manual

This document lists every file edited/created in this work session and explains
how to use them. It covers three things:

1. **OOM fix** on a 64 GB-RAM host.
2. **Non-SSD speed fix** for `CustomCroppingDataset`.
3. **mAP tuning** + a new **academic evaluation** script (`evaluation.py`) with a
   `pytest` test suite.

> ⚠️ This host has **no GPU** — training/evaluation must be run on the RTX 4090
> machine. The pure-metric `pytest` suite, however, runs fine on CPU.

---

## 0. Changelog

All entries are from the same work session (**2026-06-17**), newest first.

### [0.5] — Reproducible, leak-free dataset split
- `trainingThread.split_dataset`: fixed seed (`SPLIT_SEED = 42`,
  `np.random.RandomState`) and **persists the split** to
  `<workdir>/logs/dataset_split.json`, keyed by stable `crop_key`.
- `CustomCroppingDataset`: every crop now carries a stable **`crop_key`**
  (source filename + in-image group index), independent of load order/machine.
- `evaluation.py`: new `--split-ids` to **replay the exact saved split** by
  `crop_key` (zero train/test leakage); seeded `--split-seed` kept as fallback.

### [0.4] — Evaluate the held-out split of the `train` folder
- `evaluation.py`: loads the **`train`** folder and splits it **0.7/0.15/0.15**
  (mirroring `trainingThread`), evaluating `--eval-split` (default `test`).

### [0.3] — Align evaluation with the real training setup
- Switched `evaluation.py` from a stand-alone full-image loader to the actual
  **`CustomCroppingDataset`** pipeline; `EvalConfig` mirrors
  `trainingThread.CustomConfig` so the **ELANW+MHSA** weights load by name.
- Class scheme fixed to repo default **`cell`(1) / `chromosome`(2)**.

### [0.2] — Academic evaluation script + tests
- Added **`evaluation.py`** (COCO mAP/AR incl. mAP@.75/.90, Boundary-F1,
  Hausdorff/HD95, FPS/FLOPs, failure cases, 3× random eval over seeds
  `[42, 1024, 2026]`) and **`test_evaluation.py`** (~25 `pytest` metric tests).

### [0.1] — OOM / non-SSD / mAP fixes
- **OOM (64 GB RAM):** `USE_MINI_MASK=True (56×56)`, `IMAGES_PER_GPU 4→2`,
  data-loader workers `32 → min(8, cpu-1)`.
- **Non-SSD speed:** `CustomCroppingDataset` switched to in-memory PNG-byte crops
  (no per-crop tiny-file disk I/O).
- **mAP tuning:** eval `DETECTION_MIN_CONFIDENCE 0.8→0.05`, larger mAP sample,
  light warm-up augmentation, rebalanced epoch caps, EarlyStopping `patience 10→15`.
- **Blocking bug:** fixed a pre-existing `IndentationError` (`def compile` outside
  the `MaskRCNN` class) that prevented `mrcnn.model` from importing at all.

---

## 1. Files changed / created

| File | Status | Purpose |
|------|--------|---------|
| `trainingThread.py` | modified | OOM fix (mini-mask, batch size), mAP tuning (eval confidence, augmentation, epoch schedule), carry RAM crops through the split, **seeded + persisted 0.7/0.15/0.15 split** (`dataset_split.json`) |
| `mrcnn/model.py` | modified | Data-loader worker cap (OOM), EarlyStopping patience, **fixed a pre-existing `IndentationError`** that blocked all training |
| `CustomCroppingDataset.py` | modified | Non-SSD speed fix: in-memory PNG-byte crops instead of thousands of tiny files; added stable **`crop_key`** for reproducible splits |
| `evaluation.py` | **new** | Academic instance-seg evaluation: COCO mAP/AR, Boundary-F1, Hausdorff/HD95, FPS/FLOPs, failure cases, 3× random eval; **replays training's exact split** via `--split-ids` |
| `test_evaluation.py` | **new** | `pytest` unit tests for the pure metric functions |

---

## 2. Detailed changes

### 2.1 `trainingThread.py` — OOM + mAP
- **`USE_MINI_MASK = True` + `MINI_MASK_SHAPE = (56, 56)`** *(primary OOM fix)*.
  With `USE_MINI_MASK = False` and `MAX_GT_INSTANCES = 256`, every batch held a
  `(BATCH, 512, 512, 256)` boolean mask tensor (hundreds of MB), multiplied
  across loader workers → exhausted 64 GB. Mini-masks shrink this ~80×, with
  negligible mAP impact.
- **`IMAGES_PER_GPU` 4 → 2** — extra memory headroom (raise back if VRAM/RAM allow).
- **Eval `DETECTION_MIN_CONFIDENCE` 0.8 → 0.05** — for a faithful PR curve; a high
  score floor discards correct low-score detections and *under-reports* mAP.
- **mAP callback sample 20 → up to 100** val images for a stable metric.
- **Warm-up augmentation `None` → light flips/rotations** (`aug_light`).
- **Epoch caps rebalanced** to `40 / 160 / 300` (heads → 4+ → all); these are upper
  bounds governed by EarlyStopping.
- **`split_dataset` now carries `image_bytes`** so the in-RAM crops survive the
  train/val/test split (by reference — no 3× memory blow-up).
- **Reproducible, leak-free split**: `split_dataset` now uses a **fixed seed**
  (`SPLIT_SEED = 42`, `np.random.RandomState`) and **saves the split** to
  `<workdir>/logs/dataset_split.json` — recorded by each crop's stable `crop_key`
  (source filename + in-image group index), so `evaluation.py` can replay the
  **exact** held-out test set.

### 2.2 `mrcnn/model.py` — OOM + a blocking bug
- **Data-loader workers `32` → `min(8, cpu-1)`** on Linux. 32 worker threads each
  build a full batch into the queue — a major host-RAM driver on 64 GB.
- **EarlyStopping `patience` 10 → 15** for fuller convergence.
- **Fixed a pre-existing `IndentationError`**: `def compile` was at module
  indentation instead of inside the `MaskRCNN` class, which prevented
  `mrcnn.model` from importing **at all** (training could not run anywhere).

### 2.3 `CustomCroppingDataset.py` — non-SSD speed
- The class used to write **thousands of tiny PNG crop files** to disk and read
  them back randomly during training — pathological on HDDs.
- Now the default (`cache_dir=None`) is **in-memory mode**: Ray workers PNG-encode
  each crop to `bytes`; `load_image` decodes straight from RAM (zero disk I/O,
  compact footprint). The old disk-cache path remains as an opt-in (pass
  `cache_dir=...`) for datasets too large for RAM.
- Each crop now carries a stable **`crop_key`** (source filename + in-image group
  index) used to persist/replay the train/val/test split reproducibly.

### 2.4 `evaluation.py` — academic evaluation *(new)*
- **Pure metric core** (numpy/scipy only, unit-testable on CPU): `extract_boundary`,
  `boundary_f1_score`, `hausdorff_distance` (max **and** HD95 via `percentile=`),
  `mask_iou`, `extract_map_at_iou`.
- **COCO mAP/AR** via `pycocotools`, strictly scoped with
  `coco_eval.params.catIds`: mAP, mAP@.5, **mAP@.75, mAP@.90**, small/medium/large
  mAP, AR.
- **Same pipeline AND split as training**: `load_eval_dataset()` loads the **`train`**
  folder via the real `CustomCroppingDataset` and splits it **0.7/0.15/0.15**
  (train/val/test), then evaluates the held-out portion (`--eval-split`, default
  `test`). With **`--split-ids logs/dataset_split.json`** it replays training's
  **exact** saved split by stable `crop_key` (no leakage); otherwise it falls back
  to a seeded split (`--split-seed`). `EvalConfig` mirrors
  `trainingThread.CustomConfig` so the **ELANW+MHSA** weights (`resnet_graph`,
  `BACKBONE="resnet101"`) load by name.
- **Hardware use** (RTX 4090 / 20c / 64 GB): `IMAGES_PER_GPU=4` batched detection,
  GPU memory-growth, 20-core intra-op threads, RAM-cached crops.
- **Efficiency**: FPS from per-image inference timing; params + FLOPs (`keras-flops`).
- **Failure cases**: auto-saves GT-vs-pred figures where best IoU < 0.3.
- **3× random evaluation** over seeds `[42, 1024, 2026]`: one GPU pass is cached,
  then bootstrap-resampled per seed → reports **Mean ± Std** and writes
  `evaluation_summary.json`.

> Classes are `cell` (1) and `chromosome` (2) — the repo/checkpoint scheme.

### 2.5 `test_evaluation.py` — TDD *(new)*
- ~25 `pytest` assertions: BF1 identical=1 / disjoint=0 / symmetric / monotonic;
  HD identical=0 / one-empty=inf / **cross-checked vs `scipy.spatial.directed_hausdorff`**
  / known translation value / HD95 ≤ HD; IoU sanity; shape-mismatch raises;
  `extract_map_at_iou` verified at 0.75 / 0.90 with a fake `COCOeval`.

---

## 3. How to run

### 3.1 Dependencies (RTX 4090 host, project env)
```bash
pip install pytest keras-flops      # scipy / cv2 / pycocotools already in requirements.txt
```

### 3.2 Verify metric logic first (CPU only — no GPU/TF needed)
```bash
pytest -v test_evaluation.py
# expect: all PASSED  -> Boundary-F1 / Hausdorff / mAP-extraction logic is correct
```

### 3.3 Train (on the GPU host)
Training is launched through the GUI/`trainingThread`. The OOM and non-SSD fixes
are automatic. Stages: heads → 4+ → all, governed by EarlyStopping.

### 3.4 Run the 3× evaluation (on the GPU host)
```bash
python evaluation.py \
    --dataset /path/to/dataset \              # contains a train/ folder with via_region_*.json
    --weights /path/to/trained.h5 \
    --workdir ./logs \
    --split-ids ./logs/dataset_split.json \   # RECOMMENDED: replay the EXACT saved split
    --eval-split test \                       # which split to evaluate (default: test)
    --output ./eval_results \
    --seeds 42 1024 2026
```
The script loads the **`train`** folder and splits it **0.7/0.15/0.15**, then
evaluates the held-out `--eval-split` portion — mirroring `trainingThread`.

- **`--split-ids ./logs/dataset_split.json`** (recommended): replays the **exact**
  split training saved, matched by stable `crop_key` → zero train/test leakage.
- If `--split-ids` is omitted, it falls back to a seeded split (`--split-seed`,
  default 42) that reproduces the same logic.
Outputs:
- console **Mean ± Std** table (mAP / mAP@.5 / .75 / .90 / S,M,L / AR / BoundaryF1 / HD95 / FPS),
- `eval_results/evaluation_summary.json`,
- `eval_results/failure_cases/*.png` (IoU < 0.3 FP/FN).

---

## 4. Notes & caveats
- **No GPU on this host**: all model code is verified statically only; confirm OOM
  ceiling and mAP gains with a real run on the 4090.
- **Split reproducibility (resolved)**: training now seeds the split
  (`SPLIT_SEED = 42`) and saves `<workdir>/logs/dataset_split.json` keyed by stable
  `crop_key`. Evaluating with `--split-ids <that file>` replays the **exact** held-out
  test set — no train/test leakage, robust to load-order / machine differences. The
  seeded `--split-seed` path remains only as a fallback when the JSON isn't available.
  If you change the cropping logic or dataset, regenerate the split file by retraining
  (or re-running the split) so the `crop_key`s stay consistent.
- **COCO object scales** are measured in **512² crop space** (eval mirrors the
  cropping pipeline), so the small/medium/large mAP split is relative to crop-space
  sizes — the correct frame for this model.
- **`IMAGES_PER_GPU`**: training uses 2 (memory-safe), evaluation uses 4
  (throughput). Adjust to your VRAM.
- **Disk-cache fallback**: if a dataset is too big for RAM, pass `cache_dir=...` to
  `CustomCroppingDataset.load_custom` to restore the on-disk crop cache.
