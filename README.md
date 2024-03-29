# Cell_RCNN_Qt
---
## Cell RCNN Segmentation

### Pre-requirements
- Install XCB
`sudo apt install xcb`
- tensorflow and ImageJ packages:
```bash
 pip install -r requirements.txt
```
- MaskRCNN-like repository

### Troubleshooting
- When you found out the error like following below:
```
QObject::moveToThread: Current thread (0x5578e7485030) is not the object's thread (0x5579015f8b40).
Cannot move to target thread (0x5578e7485030)

qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/min20120907/anaconda3/envs/cell/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
```

Please remove the qt library from `opencv-python` by:
```
rm -rf $PYTHON_PATH/lib/python3.10/site-packages/cv2/qt/
```

If you run this within 4K or higher resolution, you can type this environment variables:
```
QT_FONT_DPI=96 QT_SCALE_FACTOR=2
```

If Segmentation fault occurs, please also type this environment variable:
```
LD_PRELOAD="/usr/lib-linux-gnu/libtcmalloc_minimal.so.4"
```

If the cause is memory fragmentation maybe the environment variable 'TF_GPU_ALLOCATOR=cuda_malloc_async' will improve the situation. 

The usage of new "coco_convert_gpt.py":
```
python coco_convert_gpt.py --append_mode --coco_path '/SSD-1TB-GEN4/dataset-png/train/FIB-SEM of a dividing cell at 3.9 min after anaphase' --zips_path '/SSD-1TB-GEN4/dataset-png/train/FIB-SEM of a dividing cell at 3.9 min after anaphase/single-2000-2200-RoiSet.zip' --mode single  --txt .tif
```

The usage of evaluation:
```
python eval_model.py --dataset /SSD-1TB-GEN4/dataset-png/ --workdir . --weight_path logs/cell20230527T1017/mask_rcnn_cell_0024.h5
```
