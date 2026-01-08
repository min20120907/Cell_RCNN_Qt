from matplotlib import pyplot as plt
import skimage
# from tqdm import tqdm <-- Removed
from mrcnn import utils
import json
import os
import cv2
import numpy as np
import ray

@ray.remote
def load_annotations(annotation, subset_dir, class_id):
    annotations = json.load(open(os.path.join(subset_dir, annotation)))
    annotations = list(annotations.values()) 
    annotations = [a for a in annotations if a['regions']]

    images = []
    for a in annotations:
        if type(a['regions']) is dict:
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]
        else:
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes'] for s in a['regions']]
        num_ids = []
        for _ in objects:
            try:
                num_ids.append(class_id)
            except:
                pass
        image_path = os.path.join(subset_dir, a['filename'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        images.append({
            'image_id': a['filename'],
            'path': image_path,
            'width': width,
            'height': height,
            'polygons': polygons,
            'num_ids': num_ids
        })

    return images

@ray.remote
def process_polygon(p, height, width):
    pts = np.vstack((p['all_points_x'], p['all_points_y'])).astype(np.int32).T
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    draw_mask = np.zeros([height, width, 3], dtype=np.uint8)
    cv2.fillPoly(draw_mask, [pts], color=(255, 255, 255))
    draw_mask = cv2.cvtColor(draw_mask, cv2.COLOR_BGR2GRAY)
    draw_mask = np.where(draw_mask > 0, 1, 0)
    return draw_mask

class CustomDataset(utils.Dataset):
    @property
    def image_ids(self):
       return self._image_ids

    @image_ids.setter
    def image_ids(self, value):
       self._image_ids = value
    def load_custom(self, dataset_dir, subset):
        self.add_class("cell", 1, "cell")
        self.add_class("cell", 2, "chromosome")
        assert subset in ["train", "val", "test"]
        subset_dir = os.path.join(dataset_dir, subset)

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])
        
        annotations = [f for f in os.listdir(subset_dir) if f.startswith("via_region_") and f.endswith(".json")]
        futures = [load_annotations.remote(a, subset_dir, 1) for a in annotations if "data_" in a] + \
                    [load_annotations.remote(a, subset_dir, 2) for a in annotations if "chromosome_" in a]
        
        # FIXED: Removed tqdm
        print(f"Loading {len(futures)} annotation files via Ray...")
        for _ in to_iterator(futures):
            pass
        results = ray.get(futures)
        
        for images in results:
            for image in images:
                self.add_image(
                    'cell',
                    image_id=image['image_id'],
                    path=image['path'],
                    width=image['width'], height=image['height'],
                    polygons=image['polygons'],
                    num_ids=image['num_ids'])

    def load_mask_old(self, image_id):
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        height = info["height"]
        width = info["width"]
        results = ray.get([process_polygon.remote(p, height, width) for _, p in enumerate(info["polygons"])])
        for i, result in enumerate(results):
            mask[:, :, i] = result
        return mask.astype(np.bool_), np.array(info['num_ids'], dtype=np.int32)

    def save_json(self, subset):
        images = []
        for image_id in self.image_ids:
            info = self.image_info[image_id]
            images.append({
                "id": int(image_id),
                "path": info["path"],
                "width": info["width"],
                "height": info["height"],
                "polygons": info["polygons"],
                "num_ids": [int(num_id) for num_id in info["num_ids"]]
            })
        with open(f"{subset}.json", "w") as f:
            json.dump(images, f)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            try:
                mask[rr, cc, i] = 1
            except:
                rr = np.clip(rr, 0, info["height"] - 1)
                cc = np.clip(cc, 0, info["width"] - 1)
                mask[rr, cc, i] = 1
        return mask.astype(np.bool), np.array(info['num_ids'], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]
