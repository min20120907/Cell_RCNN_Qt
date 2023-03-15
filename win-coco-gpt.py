import os
import json
import sys
import cv2
import numpy as np
from zipfile import ZipFile
import glob
from multiprocessing import Pool
import read_roi
import time
from tqdm import tqdm
from os.path import dirname
coco_path = sys.argv[1]
txt = sys.argv[2]

def process_file(args):
    zip_file, filename, json_name = args
    original={}
    roi = read_roi.read_roi_zip(coco_path + "/" + zip_file)
    roi_list = list(roi.values())

    im = cv2.imread(coco_path + '/' + filename)
    h, w, c = im.shape
    size = os.path.getsize(coco_path + '/' + filename)

    try:
        with open(json_name) as f:
            original = json.loads(f.read())
    except ValueError:
        print('Decoding JSON has failed')
    except FileNotFoundError:
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
            filename2 = filename.replace(txt, "").replace("-", " ").split(" ")
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
                    #  print(x_list[j])
                for k in range(len(y_list)):
                    if y_list[k] >=h:
                        y_list[k] = h
                #print(y_list[k])
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
            print("[ERROR] Can't find any type specific files! (Maybe check the file type)") 
    with open(json_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(original, ensure_ascii=False))

    return json_name

if __name__ == "__main__":
    os.chdir(coco_path)
    path = "."

    args1 = []
    for d in os.walk(path):
        for folder in d[1]:
            for r, d, f in os.walk(str(folder)):
                filenames = []
                zips = []
                for file in f:
                    if os.path.splitext(file)[-1] == txt:
                        filenames.append(os.path.join(r, file))
                    elif os.path.splitext(file)[-1] == ".zip":
                        zips.append(os.path.join(r, file))

                for filename in filenames:
                    for zip_file in zips:
                        json_name = "via_region_data_part_" + str(len(args1) + 1) + ".json"
                        args1.append((zip_file, filename, json_name))

    print("Scanning completed! i=" + str(len(args1)))

    start_time = time.time()

    with Pool(processes=os.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, args1), total=len(args1)))

    result = {}
    print("Combining...")

    for f in glob.glob("*.json"):
        with open(f, "r") as infile:
            try:
                result.update(json.load(infile))
            except:
                pass

    with open("via_region_data.json", "w") as outfile:
        try:
            json.dump(result, outfile)
        except:
            pass
    print("---CONVERT ENDED----")
    print("---" + str(time.time() - start_time) + "secs ----")

