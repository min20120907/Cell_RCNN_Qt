#!/usr/bin/env python  
import os  
import time  
from multiprocessing import Pool  
  
def getFile(path) :  
  # Obtain the file list under given folder  
  fileList = []  
  for root, dirs, files in list(os.walk(path)) :  
    for i in files :  
      if i.endswith('.py'):  
        fileList.append(root + "/" + i)  
  return fileList  
  
def operFile(filePath) :  
  # Calculate the line, word count of given file path  
  filePath = filePath  
  fp = open(filePath)  
  content = fp.readlines()  
  fp.close()  
  lines = len(content)  
  alphaNum = 0  
  for i in content :  
    alphaNum += len(i.strip('\n'))  
  return lines,alphaNum,filePath  
  
def out(list1, writeFilePath) :  
  # Output the statistic result into given file  
  fileLines = 0  
  charNum = 0  
  fp = open(writeFilePath,'a')  
  for i in list1 :  
    fp.write(i[2] + " Line: "+ str(i[0]) + "Word: "+str(i[1]) + "\n")  
    fileLines += i[0]  
    charNum += i[1]  
  fp.close()  
  print fileLines, charNum  
  
if __name__ == "__main__":  
  # Create multiple process to handle the request  
  startTime = time.time()  
  filePath = "/mnt/2b619254-a77a-4bbc-9e10-052f1ff6a3cc/Cell_RCNN_Qt"  
  fileList = getFile(filePath)  
  pool = Pool(16)  
  resultList =pool.map(operFile, fileList)  
  pool.close()  
  pool.join()  
  
  writeFilePath = "res.txt"  
  print('Total {} py file(s) being processed!'.format(len(resultList)))  
  out(resultList, writeFilePath)  
  endTime = time.time()  
  print "used time is ", endTime - startTime 
