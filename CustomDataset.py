from matplotlib import pyplot as plt
import skimage
from tqdm import tqdm
from mrcnn import utils
import json
import os
import cv2
import numpy as np
import ray
@ray.remote
def load_annotations(annotation, subset_dir, class_id):
    # Load annotations from JSON file
    annotations = json.load(open(os.path.join(subset_dir, annotation)))
    annotations = list(annotations.values()) 
    annotations = [a for a in annotations if a['regions']]

    # Add images
    images = []
    for a in annotations:
        # Get the x, y coordinates of points of the polygons that make up
        # the outline of each object instance. These are stored in the
        # shape_attributes (see JSON format above)
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
        # load_mask() needs the image size to convert polygons to masks.
        # Unfortunately, VIA doesn't include it in JSON, so we must read
        # the image. This is only manageable since the dataset is tiny.
        image_path = os.path.join(subset_dir, a['filename'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        images.append({
            'image_id': a['filename'],  # use file name as a unique image id
            'path': image_path,
            'width': width,
            'height': height,
            'polygons': polygons,
            'num_ids': num_ids
        })

    return images
@ray.remote
def process_polygon(p, height, width):
    # Convert the polygon points to the correct shape
    pts = np.vstack((p['all_points_x'], p['all_points_y'])).astype(np.int32).T

    # Validate and clip the coordinates to the valid range
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

    # Draw the polygon on the mask using fillPoly
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
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cell", 1, "cell")
        self.add_class("cell", 2, "chromosome")
        # self.add_class("cell", 3, "nuclear")
        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        subset_dir = os.path.join(dataset_dir, subset)

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])
        
        # Load annotations from all JSON files using Ray multiprocessing
        annotations = [f for f in os.listdir(subset_dir) if f.startswith("via_region_") and f.endswith(".json")]
        futures = [load_annotations.remote(a, subset_dir, 1) for a in annotations if "data_" in a] + \
                    [load_annotations.remote(a, subset_dir, 2) for a in annotations if "chromosome_" in a]
        #             [load_annotations.remote(a, subset_dir, 3) for a in annotations if "nuclear_" in a]
        # futures =   [load_annotations.remote(a, subset_dir, 3) for a in annotations if "nuclear_" in a]
        # Showing the progressbar
        for _ in tqdm(to_iterator(futures), total=len(futures)):
            pass
        results = ray.get(futures)
        

        # Add images
        for images in results:
            for image in images:
                self.add_image(
                    'cell',
                    image_id=image['image_id'],  # use file name as a unique image id
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
    # save the json file according to the list of images in the dataset
    def save_json(self, subset):
        # Create a dictionary with the image information
        images = []
        for image_id in self.image_ids:
            info = self.image_info[image_id]
            images.append({
                "id": int(image_id),  # Convert int64 to int
                "path": info["path"],
                "width": info["width"],
                "height": info["height"],
                "polygons": info["polygons"],
                "num_ids": [int(num_id) for num_id in info["num_ids"]]  # Convert int64 to int
            })
        
        # Save the dictionary to a JSON file
        with open(f"{subset}.json", "w") as f:
            json.dump(images, f)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # print(f"i={i}, rr={rr}, cc={cc}, len(cc)={len(cc)}")
            try:
                mask[rr, cc, i] = 1
            except:
                rr = np.clip(rr, 0, info["height"] - 1)  # Clip row indices to valid range
                cc = np.clip(cc, 0, info["width"] - 1)   # Clip column indices to valid range
                mask[rr, cc, i] = 1
                # print("Error Occured")
                # print(f"i={i}, rr={rr}, cc={cc}, len(cc)={len(cc)}")
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.array(info['num_ids'], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]