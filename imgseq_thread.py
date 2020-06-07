from skimage import io
import glob
import cv2 as cv
import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMainWindow, QApplication, QListView, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
import os
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from skimage import img_as_ubyte

class imgseq_thread(QtCore.QThread):
    def __init__(self, parent=None, txt='.tif',is_path=''):
        super(imgseq_thread, self).__init__(parent)
        self.is_path=is_path
        self.txt=txt
    append = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)
    def autoAdjustments(self,img):
        # create new image with the same size and type as the original image
        new_img = np.zeros(img.shape, img.dtype)

        # calculate stats
        alow = img.min()
        ahigh = img.max()
        amax = 255
        amin = 0

        # access each pixel, and auto adjust
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                a = img[x, y]
                new_img[x, y] = amin + (a - alow) * ((amax - amin) / (ahigh - alow))

        return new_img
    def autoAdjustments_with_convertScaleAbs(self,img):
        alow = img.min()
        ahigh = img.max()
        amax = 255
        amin = 0
        # calculate alpha, beta
        alpha = ((amax - amin) / (ahigh - alow))
        beta = amin - alow * alpha
        # perform the operation g(x,y)= α * f(x,y)+ β
        new_img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
        return [new_img, alpha, beta]
    def run(self):
        for f in glob.glob(self.is_path+"/*"+self.txt):
            im = io.imread(f)
            self.progressBar_setMaximum.emit(im.shape[0])
            progress=0
            for i in range(len(im)):
                self.progressBar.emit(progress)
                progress+=1
                r = self.autoAdjustments_with_convertScaleAbs(img_as_ubyte(im[i][0]))[0]
                g = self.autoAdjustments_with_convertScaleAbs(img_as_ubyte(im[i][1]))[0]
                b = img_as_ubyte(np.zeros((g.shape[0],g.shape[1]),dtype=g.dtype))
                #print(r)
                self.append.emit(str(b.shape))
                rg_channels = cv.merge([r,g,b])
                io.imsave(self.is_path+"/"+os.path.splitext(os.path.basename(f))[0]+"-"+str(i)+self.txt, rg_channels)
                self.append.emit("Conversion completed: "+self.is_path+"/"+str(f)+"-"+str(i)+self.txt)