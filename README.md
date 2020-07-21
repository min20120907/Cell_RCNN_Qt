# Cell_RCNN_Qt
---
## Cell RCNN Segmentation

### Pre-requirements
- tensorflow and ImageJ packages:
```bash
 pip install -r requirements.txt
```
- MaskRCNN-like repository
- set more files opening in the system settings
```
ulimit -n 20000
```
### Troubleshooting
- The moment when Windows COCO json not working on Linux, do this command
```
sed 's+\/+\\\\+g' 
```
- Relatively you can do this to Linux
```
sed 's+\\\\+\/+g' 
