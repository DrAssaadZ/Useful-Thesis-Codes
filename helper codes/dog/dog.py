import cv2 
import numpy as np

def difference_of_gaussian(img, sigma1=1, sigma2=2):
    # difference of gaussian
    img_blur1 = cv2.GaussianBlur(img, (5, 5), sigma1, borderType=cv2.BORDER_REPLICATE)
    img_blur2 = cv2.GaussianBlur(img, (3, 3), sigma2, borderType=cv2.BORDER_REPLICATE)
    img_dog = (img_blur1 - img_blur2)

    # normalize the pixel values of the DoG images between -1 and 1
    # img_dog = img_dog / np.max(np.abs(img_dog))
    img_dog2 = (255.0 * (0.5*img_dog + 0.5)).clip(0, 255).astype(np.uint8)

    return img_dog2

img = cv2.imread('imgcat.jpg')
img1 = difference_of_gaussian(img)
cv2.imwrite('dog.jpg', img1)