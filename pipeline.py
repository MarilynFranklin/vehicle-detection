import numpy as np
import cv2
import matplotlib.image as mpimg
import os
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lesson_functions import *
from scipy.ndimage.measurements import label
import collections

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

class Pipeline():
    def __init__(self, image, heatmaps, save_image=False, image_prefix='', scale=1.5,
            heat_threshold=2):
        self.image = image
        self.save_image = save_image
        self.image_prefix = image_prefix
        self.scale = scale
        self.heat_threshold = heat_threshold
        self.heatmaps = heatmaps

        self.save('original.jpg', self.image)

    def is_binary(self, img):
        return len(img.shape) == 2

    def save(self, fname, img):
        if self.save_image:
            output_folder = "output_images/"+self.image_prefix
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            filename = "{output_folder!s}/{fname!s}".format(**locals())
            if (self.is_binary(img)):
                mpimg.imsave(filename, img, cmap='hot')
            else:
                mpimg.imsave(filename, img)

    def run(self):
        settings = [
                { 'ystart': 400, 'ystop': 464, 'scale': .5 },
                { 'ystart': 432, 'ystop': 624, 'scale': 1.5 },
                { 'ystart': 400, 'ystop': 656, 'scale': 2 }
        ]
        boxes = []
        for index, config in enumerate(settings):

            out_img, img_boxes = find_cars(self.image, config['ystart'],
                    config['ystop'], config['scale'], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
            boxes += img_boxes

            fname = "find-cars-%(index)s.jpg." % locals()
            self.save(fname, out_img)

        heat = np.zeros_like(self.image[:,:,0]).astype(np.float)
        current_heatmap = add_heat(heat, boxes)
        self.heatmaps.append(current_heatmap)
        heat = sum(self.heatmaps)

        # Apply threshold to help remove false positives
        thresh = len(self.heatmaps) + 1
        heat = apply_threshold(heat, thresh)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(self.image), labels)
        self.save('heatmap.jpg', heatmap)
        self.save('boxes.jpg', draw_img)

        # fig = plt.figure()
        # plt.subplot(121)
        # plt.imshow(draw_img)
        # plt.title('Car Positions')
        # plt.subplot(122)
        # plt.imshow(heatmap, cmap='hot')
        # plt.title('Heat Map')
        # plt.savefig("./output_images/plot.jpg")
        # fig.tight_layout()

        return draw_img

class ProcessImages():
    def __init__(self, scale=1.5, heat_threshold=2):
      self.scale = scale
      self.heat_threshold = heat_threshold
      self.heatmaps = collections.deque(maxlen=15)

    def run_pipeline(self, image, save_image=False, image_prefix=''):
        pipeline = Pipeline(image, self.heatmaps, save_image, image_prefix, self.scale,
                self.heat_threshold)
        result = pipeline.run()

        return result
