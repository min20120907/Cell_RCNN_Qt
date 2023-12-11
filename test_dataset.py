import unittest
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from CustomCroppingDataset import crop_image_by_polygon

class TestCustomCroppingDataset(unittest.TestCase):
    def test_crop_image_by_polygon(self):
        # Load the image
        img = Image.open('/home/e814/Documents/dataset-png/sample.png')

        # Define a polygon
        polygon = [(25, 25), (75, 75), (25, 75), (75, 25)]

        # Crop the image
        cropped_img = crop_image_by_polygon(img, polygon)

        # Plot the original image and the polygon
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        poly_patch = patches.Polygon(polygon, fill=False)
        ax.add_patch(poly_patch)
        plt.show()

        # Plot the cropped image
        plt.imshow(cropped_img)
        plt.show()

        # Assert the cropped image is as expected (replace with your own condition)
        self.assertEqual(cropped_img.size, (50, 50))

if __name__ == '__main__':
    unittest.main()