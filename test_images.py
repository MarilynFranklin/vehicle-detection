import numpy as np
import cv2
import glob
import os
import matplotlib.image as mpimg
from pipeline import ProcessImages

process_images1 = ProcessImages(scale=1.5)
process_images2 = ProcessImages(scale=1)
process_images3 = ProcessImages(scale=.5)
scales = [1.5, 1, .5]
heat_thresholds = [1, 2, 3]

def run_pipeline(path, heat_threshold, scale):
    fname = path.split('/')[-1]
    name = fname.split('.')[0]
    prefix = 'scale-{scale!s}/heat-threshold-{heat_threshold!s}/{name!s}/'.format(**locals())
    img = mpimg.imread(path)
    process_images = ProcessImages(scale=scale, heat_threshold=heat_threshold)
    process_images.run_pipeline(img, save_image=(True), image_prefix=prefix)

def processTestImages(input_files='./test_images/*.jpg', output_folder='./'):
    images = glob.glob(input_files)

    for scale in scales:
        for threshold in heat_thresholds:
            for path in images:
                run_pipeline(path, threshold, scale)

processTestImages()
