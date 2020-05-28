import subprocess
import os
import struct
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io
import codecs
from zipfile import ZipFile
from PymageJ.roi import ROIEncoder, ROIRect, ROIPolygon
import glob
import numpy
from PIL import Image
import skimage
from skimage import feature
import cv2
import mlrose
import progressbar
import time
import logging
logging.getLogger('tensorflow').disabled = True
#PyQt5 Dependencies
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, QThread
#time
from datetime import datetime
import json
import read_roi
import io
from os.path import dirname
import json

class cocoThread(QtCore.QThread):
    def __init__(self, parent=None, coco_path='', txt = '.png'):
        super(cocoThread, self).__init__(parent)
        self.coco_path=coco_path
        self.txt = txt
    append_coco = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)
    def run(self):
            
        try:
            os.chdir(self.coco_path)
            #self.append_coco.emit("Current Path: "+self.coco_path)
            path ="."

            # ROI arrays
            filenames = []
            zips = []
            dirs =[]
            # scanning

            for d in os.walk(path):
                for dir in d:
                    for r,d,f in os.walk(str(dir)):
                        for file in f:
                            if self.txt in file:
                                filenames.append(os.path.join(r, file))
                            elif ".zip" in file:
                                zips.append(os.path.join(r, file))
                        # Sorting
                        zips.sort()
                        filenames.sort()
                        # looping and decoding...
                        #self.append_coco.emit(zips)
                        self.progressBar_setMaximum.emit(len(filenames))
                        for j in range(len(zips)):
                            for i in range(len(filenames)):
                                self.progressBar.emit(i+1)
                                # declare ROI file
                                roi = read_roi.read_roi_zip(zips[j])
    
                                roi_list = list(roi.values())
                                # ROI related file informations
                                filename = filenames[i].replace("./", "")
                                im = cv2.imread("./"+filename)
                                h, w, c = im.shape
                                size = os.path.getsize("./"+filename)
                                try:
                                    f = open("via_region_data.json")
                                    original = json.loads(f.read())
                                    self.append_coco.emit("Writing..."+str(zips[j]))
                                    # Do something with the file
                                except FileNotFoundError:
                                    self.append_coco.emit("File not exisited, creating new file...")
                                    original = {}
                                data = {
                                    filename
                                    + str(size): {
                                        "fileref": "",
                                        "size": size,
                                        "filename": filename,
                                        "base64_img_data": "",
                                        "file_attributes": {},
                                        "regions": {},
                                    }
                                }
                                # write json
                                length = len(list(roi.values()))
                                
                                for a in range(length):
                                    
                                    filename2 = filename.replace(self.txt, "").replace("-", " ").split(" ")
                                    roi_name = roi_list[a]["name"].replace("-", " ").split(" ")
                                    filenum = ""
                                    if int(filename2[-1]) > 10 and int(filename2[-1]) < 100:
                                        filenum = "00" + str(int(filename2[-1]))
                                    elif int(filename2[-1]) > 100 and int(filename2[-1]) < 1000:
                                        filenum = "0" + str(int(filename2[-1]))
                                    elif int(filename2[-1]) > 1 and int(filename2[-1]) < 10:
                                        filenum = "000" + str(int(filename2[-1]))
                                    elif int(filename2[-1]) > 1000 and int(filename2[-1]) < 10000:
                                        filenum = str(int(filename2[-1]))
                                    #self.append_coco.emit("roi_name: ", roi_name[0], "filename: ", filenum)
                                    if filenum == roi_name[0]:
                                        
                                        x_list = roi_list[a]["x"]
                                        y_list = roi_list[a]["y"]
                                        for l in range(len(x_list)):
                                            if x_list[l] >= w:
                                                x_list[l] = w
                                            #  self.append_coco.emit(x_list[j])
                                        for k in range(len(y_list)):
                                            if y_list[k] >=h:
                                                y_list[k] = h
                                        #                self.append_coco.emit(y_list[k])
                                        # parameters
                                        x_list.append(roi_list[a]["x"][0])
                                        y_list.append(roi_list[a]["y"][0])
                                        regions = {
                                            str(a): {
                                                "shape_attributes": {
                                                    "name": "polygon",
                                                    "all_points_x": x_list,
                                                    "all_points_y": y_list,
                                                },
                                                "region_attributes": {"name": dirname(dir).replace("-ROI", " ")+"-"+str(j)},
                                            }
                                        }
                                        data[filename + str(size)]["regions"].update(regions)
                                        original.update(data)
                                with io.open("via_region_data.json", "w", encoding="utf-8") as f:
                                    f.write(json.dumps(original, ensure_ascii=False))
        except:
            self.append_coco.emit("Convert Finished!")