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
    a = json.load(open(os.path.join(subset_dir, annotation)))
    # annotations = [a for a in annotations if a['regions']]
    # Add images
    images = []
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
class LiveCellDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the bottle dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # List of all possible classes
        classes = ['A172', 'BT474', 'BV2', 'Huh7', 'MCF7', 'SHSY5Y', 'SkBr3', 'SKOV3'] # All classes from LIVECell

        # Add classes
        for i, class_name in enumerate(classes, start=1):
            self.add_class("cell", i, class_name)
        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        subset_dir = os.path.join(dataset_dir, subset)

        def to_iterator(obj_ids):
            while obj_ids:
                done, obj_ids = ray.wait(obj_ids)
                yield ray.get(done[0])
        
        # Load annotations from all JSON files using Ray multiprocessing
        annotations = [f for f in os.listdir(subset_dir) if f.startswith("via_region_") and f.endswith(".json")]
        futures = []
        for class_name in classes:
            futures += [load_annotations.remote(a, subset_dir, i) for a in annotations if class_name in a]

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