import numpy as np
import cv2
import glob
import os
import matplotlib.image as mpimg
from pipeline import ProcessImages

process_images = ProcessImages()

def processTestImages(input_files='./test_images/*.jpg', output_folder='./'):
    images = glob.glob(input_files)

    for index, path in enumerate(images):
        fname = path.split('/')[-1]
        name = "test_images/{fname!s}".format(**locals())
        new_name = "test_images_output/{fname!s}".format(**locals())
        img = mpimg.imread(name)
        image = process_images.run_pipeline(img, save_image=(index == 0))
        mpimg.imsave(new_name, image)

processTestImages()