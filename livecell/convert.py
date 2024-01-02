import json

# Read COCO dataset
with open('train/livecell_coco_train.json') as f:
    data = json.load(f)

# Initialize counter
i = 0 

# Iterate over images in COCO dataset
for img in data['images']:
    img_dict = {}
    img_dict['fileref'] = ""
    img_dict['size'] = img['width']*img['height'] # assuming the size is the area of the image
    img_dict['filename'] = img['file_name']
    img_dict['base64_img_data'] = ""
    img_dict['file_attributes'] = {}
    img_dict['regions'] = {}

    # Extract class name from filename
    class_name = img['file_name'].split('_')[0]
    
    # Iterate over annotations
    for ann in data['annotations']:
        if ann['image_id'] == img['id']:
            region = {}
            region['shape_attributes'] = {'name': 'polygon', 'all_points_x': ann['segmentation'][0][::2], 'all_points_y': ann['segmentation'][0][1::2]}
            region['region_attributes'] = {'name': class_name}

            img_dict['regions'][f"region{ann['id']}"] = region

    i+=1
    with open(f"train/via_region_{class_name}_{i}.json",'w') as fi:
        json.dump(img_dict,fi)

