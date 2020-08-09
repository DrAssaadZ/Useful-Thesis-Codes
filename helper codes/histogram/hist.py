
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np



img = cv.imread('4.jpg', 0)

# plt.hist(img.ravel(), bins=256, range=(0.0, 255), fc='k', ec='k')
# plt.show()
# plt.savefig('hist1.jpg')

# img2 = cv.equalizeHist(img) 
# # cv.imshow('img', img)
# cv.imwrite('img3equalized.jpg', img2)
# plt.hist(img2.ravel(), bins=256, range=(0.0, 255), fc='k', ec='k')

# plt.savefig('hist3equalised.jpg')

# 110, 
T = 45
# retval, threshold = cv.threshold(img, T, 255, cv.THRESH_BINARY)

retval,threshold = cv.threshold(img,0,255,cv.THRESH_OTSU)
cv.imshow('original',threshold)
# cv.imwrite('threshold' + str(T) + '.jpg' , threshold)
cv.imwrite('otsuthreshold.jpg', threshold)
print(retval)
cv.waitKey()