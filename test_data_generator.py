import matplotlib.pyplot as plt
from mrcnn import model as modellib
import imgaug.augmenters as iaa
from CustomCroppingDataset import CustomCroppingDataset
from CustomDataset import CustomDataset
from mrcnn.config import Config
import cv2
# testing dataset.
print("Loading testing dataset")
dataset_test = CustomCroppingDataset()
dataset_test.load_custom("/home/e814/Documents/dataset-png","test")
dataset_test.prepare()

class CustomConfig(Config):
    """Configuration for testing on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    MAX_GT_INSTANCES = 1
    IMAGE_RESIZE_MODE = "none"
    # Give the configuration a recognizable name
    NAME = "cell"
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 6
    # IMAGE_CHANNEL_COUNT = 1
    # GPU_COUNT = 2
    USE_MINI_MASK = False
    # Number of classes (including background)
    NUM_CLASSES = 1 + 3 # Background + cell + chromosome
    # NUM_CLASSES = 1 + 1 # Background + cell
    # Number of testing steps per epoch
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_DIM = 64
    # Backbone network architecture
    BACKBONE = "resnet101"
    # Number of validation steps per epoch
    VALIDATION_STEPS = 50

aug = iaa.Sometimes(5/6, iaa.OneOf([
    iaa.Fliplr(1),
    iaa.Flipud(1),
    iaa.Affine(rotate=(-45, 45)),
    iaa.Affine(rotate=(-90, 90)),
    iaa.Affine(scale=(0.5, 1.5)),
    iaa.Fliplr(0.5), # 左右翻轉概率為0.5
    iaa.Flipud(0.5), # 上下翻轉概率為0.5
    iaa.Affine(rotate=(-10, 10)), # 隨機旋轉-10°到10°
    iaa.Affine(scale=(0.8, 1.2)), # 隨機縮放80%-120%
    iaa.Crop(percent=(0, 0.1)), # 隨機裁剪，裁剪比例為0%-10%
    iaa.GaussianBlur(sigma=(0, 0.5)), # 高斯模糊，sigma值在0到0.5之間
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)), # 添加高斯噪聲，噪聲標準差為0到0.05的像素值
    iaa.LinearContrast((0.5, 1.5)), # 對比度調整，調整因子為0.5到1.5
]))

test_generator = modellib.data_generator(
            dataset_test, CustomConfig(), shuffle=True, batch_size=CustomConfig().BATCH_SIZE)
# Get a batch of data from the generator
inputs, outputs = next(test_generator)

# Get the images and masks from the batch
images = inputs[0]
masks = inputs[6]

# Convert images to RGB format
images_rgb = []
for image in images:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images_rgb.append(image_rgb)

# Plot the images and masks
for i in range(len(images_rgb)):
    image_rgb = images_rgb[i]
    mask = masks[i]

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Image')

    # Plot the mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0], cmap='gray')  # Change index as needed to view other masks
    plt.title('Mask')

    plt.show()
