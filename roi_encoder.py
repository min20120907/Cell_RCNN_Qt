from PymageJ.roi import ROIEncoder, ROIRect

roi_obj = ROIRect(20, 30, 40, 50) # Make ROIRect object specifing top, left, bottom, right
with ROIEncoder('roi_filepath.roi', roi_obj) as roi:
    roi.write()