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

coco_path = sys.argv[1]
txt = sys.argv[2]

def process_file(args):
    zip_file, filename, json_name = args

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

    # ... (rest of the code remains unchanged)

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

