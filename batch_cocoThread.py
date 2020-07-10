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
from shutil import copyfile
from sympy import Symbol
from math import sqrt
from sympy.solvers import solve
from multiprocessing import Process
import time
class batch_cocoThread(QtCore.QThread):
    def __init__(self, parent=None, coco_path='', txt = '.png'):
        super(batch_cocoThread, self).__init__(parent)
        self.coco_path=coco_path
        self.txt = txt
    append_coco = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)
    EOF_STATUS = False
    
    def run_func(self,zips,filenames,json_name,folder):
        
        count2 = 0
                   
        # Sorting
        zips.sort()
        filenames.sort()
        # looping and decoding...
        #self.append_coco.emit(zips)

        for j in range(len(zips)):
            for i in range(len(filenames)):
                
                count2+=1
                # declare ROI file
                roi = read_roi.read_roi_zip(self.coco_path+"/"+zips[j])
                roi_list = list(roi.values())  
                # ROI related file informations
                filename = filenames[i].replace("./", "")
                im = cv2.imread(self.coco_path+'/'+filename)
                h, w, c = im.shape
                size = os.path.getsize(self.coco_path+'/'+filename)
                try:
                    f = open(json_name)
                    original = json.loads(f.read())
                    f.close()
                    #self.append_coco.emit("Writing..."+str(zips[j]))
                    # Do something with the file
                except ValueError:  # includes simplejson.decoder.JSONDecodeError
                    print('Decoding JSON has failed')
                except FileNotFoundError:
                    #self.append_coco.emit("File not exisited, creating new file...")
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
                for a in roi_list:
                
                    try:
                        filename2 = filename.replace(self.txt, "").replace("-", " ").split(" ")
                        roi_name = a["name"].replace("-", " ").split(" ")
                        roi_num =int(roi_name[0])
                        file_num = int(filename2[-1])
                        has_zero = False
                        if has_zero:
                            roi_num-=1
                        if file_num == roi_num:
                            #print(has_zero)
                            #print(int(filename2[-1]), " ", roi_num)
                            #print(a)
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
                                    "region_attributes": {"name": dirname(folder)},
                                }
                            }
                            data[filename + str(size)]["regions"].update(regions)
                            original.update(data)
                    except KeyError:
                        #Line Exception
                        if a['type']=="line":
                            x1 = a['x1']
                            x2 = a['x2']
                            y1 = a['y1']
                            y2 = a['y2']
                            width = a['width']
                            new_x_list =[]
                            new_y_list =[]
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
                                        "name": dirname(folder)
                                    }
                                } 
                            }
                            data[filename + str(size)]["regions"].update(regions)
                            original.update(data)
                        elif a['type']=="oval":
                            TWO_PI=np.pi*2
                            angles = 128
                            angle_shift = TWO_PI/ angles
                            phi = 0
                            center_x = (2*(a['left']) + a['width'])/2
                            center_y = (2 * a['top'] + a['height'])/2
                            x_list = []
                            y_list = []
                            for i in range(angles):
                                phi+=angle_shift
                                x_list.append(int(center_x + (a['width'] * np.cos(phi)/2)))
                                y_list.append(int(center_y + (a['height'] * np.sin(phi)/2)))
                            regions = {
                                str(a): {
                                    "shape_attributes": {
                                        "name": "polygon",
                                        "all_points_x": x_list,
                                        "all_points_y": y_list,
                                    },
                                    "region_attributes": {"name": dirname(folder)},
                                }
                            
                            }
                            data[filename + str(size)]["regions"].update(regions)
                            original.update(data)
                    except IndexError or FileNotFoundError:
                        self.append_coco.emit("[ERROR] Can't find any type specific files! (Maybe check the file type)")        
                with io.open(json_name, "w", encoding="utf-8") as f:
                    f.write(json.dumps(original, ensure_ascii=False))
                    f.close()
        print("Converted File: ", json_name)
        
            
        
    def run(self):
        
        os.chdir(self.coco_path)
        #self.append_coco.emit("Current Path: "+self.coco_path)
        path ="."
        # ROI arrays
        dirs =[]    
        #scanning
        count =0
        for d in os.walk(path):
            for folder in d[1]:
                for r,d,f in os.walk(str(folder)):
                    for file in f:
                        if self.txt in file:
                            count+=1
                        elif ".zip" in file:
                            count+=1
        
        #print(count)
        i=1
        j=1
        procs=[]
        for d in os.walk(path):
            for folder in d[1]:
                
                for r,d,f in os.walk(str(folder)):
                    filenames = []
                    zips = []
                    for file in f:
                        
                        if os.path.splitext(file)[-1] == self.txt:
                            filenames.append(os.path.join(r, file))
                        elif os.path.splitext(file)[-1] == ".zip":
                            zips.append(os.path.join(r, file))
                    
                    p = Process(target=self.run_func, args=(zips,filenames,"via_region_data_part_"+str(i)+".json",folder))                    
                    procs.append(p)
                    i+=1
        print("Scanning completed! i=",i)
        self.progressBar_setMaximum.emit(i)
        start_time = time.time()
        k=0
        for p in procs:
            p.start()
            j+=1
            k+=1
            self.progressBar.emit(k)
            if j > 8:
                for i in range(8):
                    p.join()
                j=1
        result = {}
        self.append_coco.emit("Combining...")
        for f in glob.glob("*.json"):
            with open(f, "r") as infile:
                result.update(json.load(infile))
                infile.close()


        with open("merged_file.json", "w") as outfile:
             json.dump(result, outfile)
             outfile.close()
        self.append_coco.emit("---CONVERT ENDED----")
        self.append_coco.emit("---" + str(time.time() - start_time)+"----")
                
                
        
                    
        
        
                        
        
                        
        

