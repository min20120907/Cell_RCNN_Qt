# Cell RCNN Qt - Repository Structure

This document outlines the reorganized directory structure for the Cell_RCNN_Qt project.

## Directory Tree

```
Cell_RCNN_Qt/
├── src/                           # Main source package
│   ├── __init__.py
│   ├── core/                      # Training and model files
│   │   ├── __init__.py
│   │   ├── Cell_Trainer.py
│   │   ├── Cell_Trainer_headless.py
│   │   ├── cell_trainer_win.py
│   │   ├── model_cell_gpu.py
│   │   ├── eval_model.py
│   │   ├── eval_model_gpu.py
│   │   ├── eval_model_gpu_cell.py
│   │   └── eval_cellpose.py
│   │
│   ├── datasets/                  # Dataset classes
│   │   ├── __init__.py
│   │   ├── CustomCroppingDataset copy.py  # ← USE THIS VERSION (delete original)
│   │   ├── CustomDataset.py
│   │   ├── LiveCellCroppingDataset.py
│   │   ├── LiveCellDataset.py
│   │   └── SmallDataset.py
│   │
│   ├── threads/                   # Multi-threading modules
│   │   ├── __init__.py
│   │   ├── trainingThread.py
│   │   ├── detectingThread.py
│   │   ├── detectingThread_ray.py
│   │   ├── batchDetectThread.py
│   │   ├── batchDetectThreadResize4x.py
│   │   ├── batch_cocoThread.py
│   │   ├── batch_cocoShrinkThread.py
│   │   ├── cocoThread.py
│   │   ├── anotThread.py
│   │   ├── BWThread.py
│   │   ├── imgseq_thread.py
│   │   └── auto_contour.py
│   │
│   ├── utils/                     # Utility functions
│   │   ├── __init__.py
│   │   ├── imagej_roi_decoder.py
│   │   ├── cellpose_converter.py
│   │   ├── convert_YOLO.py
│   │   ├── coco_convert_gpt.py
│   │   ├── roi2coco_line.py
│   │   ├── expand_mp.py
│   │   ├── shrink.py
│   │   └── customcroppingdataset_count.py
│   │
│   └── ui/                        # User interface
│       ├── __init__.py
│       ├── main_ui.py
│       └── MaskRCNN.ui
│
├── mrcnn/                         # Mask R-CNN library
│
├── scripts/                       # Standalone scripts (NEW)
│   ├── BW-generating.py
│   ├── win-coco.py
│   ├── win-coco-gpt.py
│   ├── win-coco-chromosome.py
│   ├── coco_convert_gpt.py
│   ├── gen_txt.py
│   ├── plot_GT.py
│   ├── plot_cells.py
│   ├── test.py
│   ├── test-mp.py
│   ├── test-mp2.py
│   ├── test-mp3.py
│   ├── test-mc.py
│   ├── test_data_generator.py
│   ├── test_dataset.py
│   ├── test_mrcnn.py
│   ├── example_polygon.py
│   ├── count_dataset.sh
│   ├── run.sh
│   ├── lin2winCOCO.sh
│   └── win2linCOCO.sh
│
├── tools/                         # ImageJ macros and diagrams (NEW)
│   ├── Macro_temp_color_code.ijm
│   ├── RGB-colorization.ijm
│   ├── splitChannel.ijm
│   ├── stack2hyperstack.ijm
│   ├── stack2image.ijm
│   ├── 分析軟體.ijm
│   └── Multiprocessing.drawio
│
├── data/                          # Data files (NEW)
│   ├── cell-feature-classify-db.csv
│   ├── cell-feature-classify-db-converted.csv
│   ├── res.txt
│   ├── merged_file.json
│   ├── profile.json
│   ├── roi_filepath.roi
│   └── merge.py
│
├── livecell/                      # Live cell dataset
├── PymageJ/                       # PyImageJ integration
├── __pycache__/                   # Python cache (gitignored)
│
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules (NEW)
└── STRUCTURE.md                   # This file
```

## Import Changes

### Before (Old Style)
```python
from CustomCroppingDataset import CustomCroppingDataset
from Cell_Trainer import cell_trainer
from detectingThread import DetectingThread
import imagej_roi_decoder
```

### After (New Style)
```python
from src.datasets import CustomCroppingDataset
from src.core import Cell_Trainer
from src.threads import detectingThread
from src.utils import imagej_roi_decoder
```

## Files to DELETE

Delete these files from the root after merging:
- ❌ `CustomCroppingDataset.py` (original - use the copy version instead)
- ❌ `Thumbs.db` (Windows cache file)
- ❌ `batchDetectThread.pyc` (compiled Python file)
- ❌ `new.bat` (unnecessary batch file)
- ❌ `new1.bat` (unnecessary batch file)
- ❌ `__pycache__/` directory (Python cache)

## Files to KEEP (Copy Version)

✅ Keep: `CustomCroppingDataset copy.py`
- Rename to: `CustomCroppingDataset.py` inside `src/datasets/`
- Delete the original `CustomCroppingDataset.py` from root

## Migration Checklist

- [ ] Review this structure and confirm it matches your project needs
- [ ] Merge the `refactor/reorganize-structure` branch to `master`
- [ ] Delete files listed in "Files to DELETE" section above
- [ ] Move Python files to their respective `src/` subdirectories:
  - [ ] Move dataset files to `src/datasets/`
  - [ ] Move trainer files to `src/core/`
  - [ ] Move thread files to `src/threads/`
  - [ ] Move utility files to `src/utils/`
  - [ ] Move UI files to `src/ui/`
- [ ] Create `scripts/` directory and move standalone scripts
- [ ] Create `tools/` directory and move ImageJ macros
- [ ] Create `data/` directory and move data files
- [ ] Update all imports throughout the project to use new `src.X` paths
- [ ] Run tests to verify everything still works
- [ ] Commit final changes

## Benefits of This Structure

✅ **Cleaner Root**: Only essential files at the top level
✅ **Better Organization**: Logical grouping of related files
✅ **Easier Maintenance**: Clear separation of concerns
✅ **Professional Standard**: Follows Python packaging best practices
✅ **Scalability**: Easy to add new modules to existing packages
✅ **IDE Support**: IDEs better understand package structure
✅ **Documentation**: Self-documenting through directory names

## Notes

- The `.gitignore` file has been created to exclude cache files and unnecessary files
- All `__init__.py` files have been created to make directories proper Python packages
- Update your entry point (main.py or similar) to import from `src.ui.main_ui`
- If you have CI/CD pipelines, update import paths there as well
