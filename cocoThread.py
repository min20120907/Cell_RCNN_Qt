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
from sympy import Symbol
from math import sqrt
from sympy.solvers import solve

class cocoThread(QtCore.QThread):
    def __init__(self, parent=None, coco_path='', txt = '.png'):
        super(cocoThread, self).__init__(parent)
        self.coco_path=coco_path
        self.txt = txt
    append_coco = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)
    def run(self):
            
        
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
                            
                            for a in roi_list:
                                try:
                                    filename2 = filename.replace(self.txt, "").replace("-", " ").split(" ")
                                    #print("roi_name: ", roi_name[0], "filename: ", filenum)
                                    if int(filename[-1]) == a['position']:
                                        print(a)
                                        x_list = a["x"]
                                        y_list = a["y"]
                                        for l in range(len(x_list)):
                                            if x_list[l] >= w:
                                                x_list[l] = w
                                            #  self.append_coco.emit(x_list[j])
                                        for k in range(len(y_list)):
                                            if y_list[k] >=h:
                                                y_list[k] = h
                                        #self.append_coco.emit(y_list[k])
                                        # parameters
                                        x_list.append(a["x"][0])
                                        y_list.append(a["y"][0])
                                        regions = {
                                            str(a): {
                                                "shape_attributes": {
                                                    "name": "polygon",
                                                    "all_points_x": x_list,
                                                    "all_points_y": y_list,
                                                },
                                                "region_attributes": {"name": dirname(dir)},
                                            }
                                        }
                                        
                                except:
                                    #Line Exception
                                    if "x1" in a:
                                        x1 = a['x1']
                                        x2 = a['x2']
                                        y1 = a['y1']
                                        y2 = a['y2']
                                        width = 600
                                        new_x_list =[]
                                        new_y_list = []

                                        if (x1-x2)==0:
                                            slope = 0
                                        else:
                                            slope = (y1-y2)/(x1-x2)
                                            slope_a = (-1)/slope
                                        midpoint=[(x1+x2)/2, (y1+y2)/2]
                                        #print("斜率: ",slope)
                                        #print(slope_a)
                                        x = Symbol('x')
                                        weight = solve(x**2+(x*slope_a)**2- (width/2)**2, x)
                                        new_x_list.append(int(x1-(weight[1])))
                                        new_x_list.append(int(x1+(weight[1])))
                                        new_x_list.append(int(x2-(weight[0])))
                                        new_x_list.append(int(x2+(weight[0])))
                                        new_x_list.append(int(x1-(weight[1])))
                                        new_y_list.append(int(y1-(weight[1]*slope_a)))
                                        new_y_list.append(int(y1+(weight[1]*slope_a)))
                                        new_y_list.append(int(y2-(weight[0]*slope_a)))
                                        new_y_list.append(int(y2+(weight[0]*slope_a)))
                                        new_y_list.append(int(y1-(weight[1]*slope_a)))
                                        #print("x坐標", new_x_list)
                                        #print("y坐標", new_y_list)
                                        #fix index out of bound exception
                                        for j in range(len(new_x_list)):
                                            if(new_x_list[j]>=2160):
                                                new_x_list[j] = 2159
                                                #print(new_x_list[j])
                                        for k in range(len(new_y_list)):
                                            if(new_y_list[k]>=2160):
                                                new_y_list[k] = 2159
                                                #print(new_y_list[k])
                                        regions = {
                                            str(a): {
                                                "shape_attributes": {
                                                    "name":  "polygon",
                                                    "all_points_x": new_x_list,
                                                    "all_points_y": new_y_list
                                                },
                                                "region_attributes": {
                                                    "name": dirname(dir)
                                                }
                                            } 
                                        }
                                    else:
                                        pass
                                data[filename + str(size)]["regions"].update(regions)
                                original.update(data)
                            with io.open("via_region_data.json", "w", encoding="utf-8") as f:
                                f.write(json.dumps(original, ensure_ascii=False))
        