# import libraries
import cv2
import numpy as np



# get grayscale image
def get_grayscale(orginal_image: str):
    return cv2.cvtColor(orginal_image, cv2.COLOR_BGR2GRAY)

# media noise removal
def remove_noise_medianBlur(orginal_image: str):
    return cv2.medianBlur(orginal_image,5)

# gaussian noise removal
def remove_noise_gaussianBlur(orginal_image: str):
    return cv2.GaussianBlur(orginal_image,(5,5),0)

#thresholding
def thresholding(orginal_image: str):
    THRESH_BINARY_INV = cv2.THRESH_BINARY_INV
    THRESH_OTSU = cv2.THRESH_OTSU
    return cv2.threshold(orginal_image, 0, 255, THRESH_BINARY_INV  + THRESH_OTSU)[1]

#canny edge detection
def canny(orginal_image: str):
    return cv2.Canny(orginal_image, 50, 100)

#dilation
def dilate(orginal_image: str):
    kernel = np.ones((1,1),np.uint8)
    return cv2.dilate(orginal_image, kernel, iterations = 1)
    
#erosion
def erode(orginal_image: str):
    kernel = np.ones((1,1),np.uint8)
    return cv2.erode(orginal_image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(orginal_image: str):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(orginal_image, cv2.MORPH_OPEN, kernel)


