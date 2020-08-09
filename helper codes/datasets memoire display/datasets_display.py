import matplotlib.pyplot as plt
from numpy import expand_dims
import cv2 as cv
import os 
import numpy as np

dataset_path = 'E:/Master/Master thesis/our memoire/resource/codes for resources/datasets memoire display/georgia tech ds/S1/'

img_names = []
dataset_files = os.listdir(dataset_path)

# getting image names and stocking them in a list
for i in range(len(dataset_files)):
	img_names.append(dataset_path + dataset_files[i])
# print()
# creating a list of the read images
images_list = []
for image in range(10):
	img1 = cv.imread(img_names[image])
	img1 = cv.resize(img1,(150, 200))
	images_list.append(img1)


# concatinating the images 
for image in range(len(dataset_files)):
	final = cv.hconcat(images_list[:])

# final = cv.hconcat([img1,img2,img3,img4,img5, img6,img7,img8, img9, img10])

cv.imwrite('final.png', final)



