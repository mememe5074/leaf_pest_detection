import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
import sys
from skimage import io, color
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from matplotlib.pyplot import figure
import os

path_img_data = 'D:/leaf_pest/PlantVillage-Dataset/raw/segmented/'
entries = os.listdir(path_img_data)

def mean_color(img, labels):
    out = np.zeros_like(img)
    for label in np.unique(labels):
        indices = np.nonzero(labels == label)
        out[indices] = np.mean(img[indices], axis=0)
    return out
def read_image(path):
    """
    Return:
        3D array, row col [LAB]
    """
    rgb = mpimg.imread(path)
    return rgb

def plot_slic_segmentation(image):
    labels = slic(image=image, n_segments=255, compactness=10, sigma=1, enforce_connectivity=True, start_label=1)
    return mean_color(image, labels)

def plot_fz_segmentation(image):
    labels = felzenszwalb(image, scale=255, sigma=1, min_size=10)
    return mean_color(image, labels)

for i in entries:
    entries_2 = [f for f in os.listdir(str(path_img_data)+str(i))]
    for s in entries_2:
        img =read_image(str(path_img_data+str(i)+'/'+str(s)))
        rgbimg = img_as_float(img)

        pic = plot_slic_segmentation(rgbimg)

        #resizing 0 to 50
        plt.figure(figsize=(2,2))
        plt.imshow(plot_slic_segmentation(rgbimg))
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(fname=str(s), bbox_inches='tight', pad_inches=0)

