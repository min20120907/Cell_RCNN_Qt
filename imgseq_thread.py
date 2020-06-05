from skimage import io
import glob
import cv2 as cv
import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtWidgets import QMainWindow, QApplication, QListView, QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
class imgSeqThread(QtCore.QThread):
    def __init__(self, parent=None, txt='.tif',is_path=''):
        super(imgSeqThread, self).__init__(parent)
        self.is_path=is_path
        self.txt=txt
    append = QtCore.pyqtSignal(str)
    progressBar = QtCore.pyqtSignal(int)
    progressBar_setMaximum = QtCore.pyqtSignal(int)

    def run(self):
        for f in glob.glob(self.is_path+"/*"+self.txt):
            im = io.imread(f)
            for i in range(im[0]):
                io.imsave(im[0][i],self.is_path+"/"+f+"-"+str(i)+"-"+self.txt)
                self.append("Conversion completed: "+self.is_path+"/"+str(f)+"-"+str(i)+"-"+self.txt)