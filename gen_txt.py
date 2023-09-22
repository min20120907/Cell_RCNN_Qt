import glob
import os

# Define the source directory where JSON files are located
source_dir = "/SSD-1TB-GEN4/dataset-png/"

# Define the output directories for train, val, and test sets
output_train_dir = os.path.join(source_dir, "train")
output_val_dir = os.path.join(source_dir, "val")
output_test_dir = os.path.join(source_dir, "test")

# Define the paths for the path list files
train_txt_path = os.path.join(source_dir, "train.txt")
val_txt_path = os.path.join(source_dir, "val.txt")
test_txt_path = os.path.join(source_dir, "test.txt")

# Get a list of image files in each directory
train_image_files = glob.glob(os.path.join(output_train_dir, "*/*.png")) + \
                    glob.glob(os.path.join(output_train_dir, "*/*.jpg")) + \
                    glob.glob(os.path.join(output_train_dir, "*/*.tif"))

val_image_files = glob.glob(os.path.join(output_val_dir, "*/*.png")) + \
                  glob.glob(os.path.join(output_val_dir, "*/*.jpg")) + \
                  glob.glob(os.path.join(output_val_dir, "*/*.tif"))

test_image_files = glob.glob(os.path.join(output_test_dir, "*/*.png")) + \
                   glob.glob(os.path.join(output_test_dir, "*/*.jpg")) + \
                   glob.glob(os.path.join(output_test_dir, "*/*.tif"))

# Write the paths to the respective text files
with open(train_txt_path, "w") as train_txt:
    train_txt.write("\n".join(train_image_files))

with open(val_txt_path, "w") as val_txt:
    val_txt.write("\n".join(val_image_files))

with open(test_txt_path, "w") as test_txt:
    test_txt.write("\n".join(test_image_files))

