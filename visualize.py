import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from lesson_functions import *
import random

cars = glob.glob('./vehicles/*/*.png')
notcars = glob.glob('./non-vehicles/*/*.png')

random_idx = int(random.uniform(0, len(cars)))

car = cars[random_idx]
notcar = notcars[random_idx]

def visualize(file, type, orient=9, pix_per_cell=8, cell_per_block=2):
    image = mpimg.imread(file)
    mpimg.imsave('./examples/features/'+type+'.jpg', image)
    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    mpimg.imsave('./examples/features/'+type+'-YCrCb.jpg', feature_image)

    for channel in range(feature_image.shape[2]):
        features, hog_image = get_hog_features(feature_image[:,:,channel], orient,
                pix_per_cell, cell_per_block, vis=True, feature_vec=False)
        mpimg.imsave('./examples/features/'+type+'-hog-'+str(channel+1)+'.jpg', hog_image)

visualize(car, 'car')
visualize(notcar, 'not-car')
