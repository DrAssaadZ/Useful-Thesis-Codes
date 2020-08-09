import numpy as np
# from skimage.util import random_noise
import cv2 
# from PIL import Image

# # def add_noise(img, noise):
# #     im_arr = np.asarray(img)

# #     # random_noise() method will convert image in [0, 255] to [0, 1.0],
# #     # inherently it use np.random.normal() to create normal distribution
# #     # and adds the generated noised back to image

# #     noise_img = random_noise(im_arr, mode=noise, var=0.5)
# #     noise_img = (255 * noise_img).astype(np.uint8)

# #     img = Image.fromarray(noise_img)
# #     # return np.array(img)
# #     img.save('speckle_img.jpg')


# # img = cv.imread('4.jpg')

# # b_img = add_noise(img, 'speckle')
# ---------------------------------------------------
# salt and pepper
# import numpy as np
# import random
# import cv2

# def sp_noise(image,prob):
#     '''
#     Add salt and pepper noise to image
#     prob: Probability of the noise
#     '''
#     output = np.zeros(image.shape,np.uint8)
#     thres = 1 - prob 
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             rdn = random.random()
#             if rdn < prob:
#                 output[i][j] = 0
#             elif rdn > thres:
#                 output[i][j] = 255
#             else:
#                 output[i][j] = image[i][j]
#     return output

# image = cv2.imread('4.jpg',0) # Only for grayscale image
# noise_img = sp_noise(image,0.05)
# cv2.imwrite('sp_noise.jpg', noise_img)
# --------------------------------------------------

# def add_gaussian_noise(image_in, noise_sigma):
#     temp_image = np.float64(np.copy(image_in))

#     h = temp_image.shape[0]
#     w = temp_image.shape[1]
#     noise = np.random.randn(h, w) * noise_sigma

#     noisy_image = np.zeros(temp_image.shape, np.float64)
#     if len(temp_image.shape) == 2:
#         noisy_image = temp_image + noise
#     else:
#         noisy_image[:,:,0] = temp_image[:,:,0] + noise
#         noisy_image[:,:,1] = temp_image[:,:,1] + noise
#         noisy_image[:,:,2] = temp_image[:,:,2] + noise

#     """
#     print('min,max = ', np.min(noisy_image), np.max(noisy_image))
#     print('type = ', type(noisy_image[0][0][0]))
#     """

#     return noisy_image

# image = cv2.imread('4.jpg',0) 
# new_img = add_gaussian_noise(image, 35)
# cv2.imwrite('gaussian_noise_image.jpg', new_img)

# -----------------------------------------------------

def noisy(noise_typ,img):
	image 
	if noise_typ == "speckle":
		row,col,ch = image.shape()
		gauss = np.random.randn(row,col,ch)
		gauss = gauss.reshape(row,col,ch)        
		noisy = image + image * gauss
		return noisy

image = cv2.imread('4.jpg',0) 
new_img = noisy("speckle",image)
cv2.imwrite('speckle_noise_image.jpg', new_img)