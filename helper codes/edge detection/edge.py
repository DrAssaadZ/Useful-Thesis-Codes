import numpy as np

import cv2 
from scipy.ndimage import convolve



kernel1 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
# kernel2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

image = cv2.imread('2.png',0) 
new_img = cv2.filter2D(image, -1, kernel1)
# cv2.imshow('img', new_img)
# cv2.waitKey()
cv2.imwrite('laplacien.jpg', new_img)