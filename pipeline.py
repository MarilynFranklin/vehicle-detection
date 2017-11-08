import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lesson_functions import *
from scipy.ndimage.measurements import label

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

class Pipeline():
    def __init__(self, image, save_image=False):
        self.image = image
        self.save_image = save_image

        self.save('original.jpg', self.image)

    def is_binary(self, img):
        return len(img.shape) == 2

    def save(self, fname, img):
        if self.is_binary(img):
            img = cv2.cvtColor(img*255, cv2.COLOR_GRAY2RGB)

        if self.save_image:
            output_folder = "./output_images/"
            filename = "{output_folder!s}/{fname!s}".format(**locals())
            cv2.imwrite(filename, img)

    def process_image(self, img):
        print("woo")

    def run(self):
        ystart = 400
        ystop = 656
        scale = 1.5

        out_img, img_boxes = find_cars(self.image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

        heat = np.zeros_like(self.image[:,:,0]).astype(np.float)
        heat = add_heat(heat, img_boxes)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat,1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(self.image), labels)

        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(122)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        plt.savefig("./output_images/plot.jpg")
        fig.tight_layout()

        return draw_img

class ProcessImages():
    def run_pipeline(self, image, save_image=False):
        pipeline = Pipeline(image, save_image)
        result = pipeline.run()

        return result
