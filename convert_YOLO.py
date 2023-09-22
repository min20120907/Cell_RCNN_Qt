import json
from PIL import Image
import glob
import os

# Define your class mappings (class_name to class_id) here.
class_mapping = {
    "cell": 0,
    "chromosome": 1,
    "nuclear": 2,
    # Add more class mappings as needed
}

# Define the source directory where JSON files are located
source_dir = "/SSD-1TB-GEN4/dataset-png/"

# Define the output directories for train, val, and test sets
output_train_dir = os.path.join(source_dir, "train")
output_val_dir = os.path.join(source_dir, "val")
output_test_dir = os.path.join(source_dir, "test")

# Create the output directories if they don't exist
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)
# Function to process JSON files for a given list of files and output directory
def process_json_files(json_files, output_dir, setName):
    for json_file in json_files:
        # Load the JSON data
        with open(json_file, "r") as json_file:
            data = json.load(json_file)
        # Iterate through each image in the JSON
        for image_filename, image_data in data.items():
            original_filename = image_data["filename"]
            # Path to the image file
            image_path = os.path.join(source_dir, setName, original_filename)
            # Open the image to get its width and height
            image = Image.open(image_path)
            image_width, image_height = image.size
            annotations = []
            for region_id, region_data in image_data["regions"].items():
                # Extract region coordinates
                x_points = region_data["shape_attributes"]["all_points_x"]
                y_points = region_data["shape_attributes"]["all_points_y"]
                # Calculate bounding box coordinates
                x_min = min(x_points)
                x_max = max(x_points)
                y_min = min(y_points)
                y_max = max(y_points)
                # Calculate bounding box center and dimensions
                x_center = (x_min + x_max) / (2 * image_width)
                y_center = (y_min + y_max) / (2 * image_height)
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                # Create YOLO annotation line
                annotation = f"{class_id} {x_center} {y_center} {width} {height}"
                annotations.append(annotation)
            if annotations:
                # Save YOLO annotations to a text file
                output_file_path = os.path.join(output_dir, original_filename.replace('.png', '.txt').replace('.tif', '.txt').replace('.jpg', '.txt'))
                with open(output_file_path, "w") as output_file:
                    output_file.write("\n".join(annotations))
# Iterate through the classes
for class_name, class_id in class_mapping.items():
    # Get a list of JSON files for the current class in train, val, and test directories
    json_pattern_train = os.path.join(source_dir, "train", f"via_region_{class_name}*.json")
    json_pattern_val = os.path.join(source_dir, "val", f"via_region_{class_name}*.json")
    json_pattern_test = os.path.join(source_dir, "test", f"via_region_{class_name}*.json")

    json_files_train = glob.glob(json_pattern_train)
    json_files_val = glob.glob(json_pattern_val)
    json_files_test = glob.glob(json_pattern_test)

    

    # Process JSON files for train, val, and test
    process_json_files(json_files_train, output_train_dir, "train")
    process_json_files(json_files_val, output_val_dir,"val")
    process_json_files(json_files_test, output_test_dir,"test")
