import read_roi
import json
import io
import os
from os import walk
from math import sqrt
from sympy.solvers import solve
from sympy import Symbol

path = "."

#ROI arrays
filenames = [

]
zips = [

]
#scanning
for r, d, f in os.walk(path):
    for file in f:
        if '.png' in file:
            filenames.append(os.path.join(r, file))
        elif '.zip' in file:
            zips.append(os.path.join(r, file))
#Sorting
zips.sort()
filenames.sort()
#looping and decoding...
for i in range(len(zips)):
    # declare ROI file
    roi = read_roi.read_roi_zip(zips[i])
    roi_list = list(roi.values())
    # ROI related file informations
    filename = filenames[i].replace("./","")
    size = os.path.getsize(filename)
    try:
        f = open("via_region_data.json")
        original = json.loads(f.read())
        print("Writing",filename,"...")
        # Do something with the file
    except FileNotFoundError:
        print("File not exisited, creating new file...")
        original = {}

    data = {
        filename+str(size): {
            "fileref":  "",
            "size":  size,
            "filename": filename,
            "base64_img_data": "",
            "file_attributes": {},
            "regions": {
                }
            }
        }

    # write json

    length = len(list(roi.values()))
    for a in range(length):
        # parameters
        #x_list = roi_list[a]['x']
        #y_list = roi_list[a]['y']
        #x_list.append(roi_list[a]['x'][0])
        #y_list.append(roi_list[a]['y'][0])
        x1 = roi_list[a]['x1']
        x2 = roi_list[a]['x2']
        y1 = roi_list[a]['y1']
        y2 = roi_list[a]['y2']
        width = 600

        new_x_list =[]
        new_y_list = []
        
        if (x1-x2)==0:
            slope = 0
        else:
            slope = (y1-y2)/(x1-x2)
            slope_a = (-1)/slope
        midpoint=[(x1+x2)/2, (y1+y2)/2]
        print("斜率: ",slope)
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
                print(new_x_list[j])
        for k in range(len(new_y_list)):
            if(new_y_list[k]>=2160):
                new_y_list[k] = 2159
                print(new_y_list[k])
        regions = {
            str(a): {
                "shape_attributes": {
                    "name":  "polygon",
                    "all_points_x": new_x_list,
                    "all_points_y": new_y_list
                },
                "region_attributes": {
                    "name": "cell"
                }
            } 
        }
        data[filename+str(size)]["regions"].update(regions)
    original.update(data)
    with io.open('via_region_data.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(original, ensure_ascii=False))

