import glob
from read_roi import *
from PymageJ.roi import ROIEncoder, ROIRect, ROIPolygon
from zipfile import ZipFile
import os
def shrink_roi(zipfile):
    # Declare the ROI File
    roi = read_roi_zip(zipfile)
    # Convert the roi list
    roi_list = list(roi.values())
    print("Shrinking the size quarter...")
    for a in roi_list:
        for b in range(len(a['x'])):
            a['x'][b] = int(a['x'][b] / 4)
        for c in range(len(a['y'])):
            a['y'][c] = int(a['y'][c] / 4)
    os.remove(zipfile)
    print("Export the data...")
    print(roi_list)
    for a in roi_list:
        roi_obj = ROIPolygon(a['x'],a['y'])
        with ROIEncoder(a['name']+".roi",roi_obj) as roi:
            roi.write()
        with ZipFile(zipfile,'a') as myzip:
            myzip.write(a['name']+".roi")
            os.remove(a['name']+".roi") 

for a in glob.glob("./*_small/*.zip"):
    shrink_roi(a)

