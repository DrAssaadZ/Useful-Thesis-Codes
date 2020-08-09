import matplotlib.pyplot as plt
import cv2 as cv
from skimage.feature import hog
from skimage import data, exposure


image = cv.imread('6.jpg')

fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)

fd1, hog_image1 = hog(image, orientations=16, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=True, multichannel=True)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3,  figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Image originale')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Rescale histogram for better display
hog_image_rescaled1 = exposure.rescale_intensity(hog_image1, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Image HoG \n avec orientations = 9')

ax3.axis('off')
ax3.imshow(hog_image_rescaled1, cmap=plt.cm.gray)
ax3.set_title('Image HoG \n avec orientations = 16')

plt.savefig('hog img.jpg', dpi=300)