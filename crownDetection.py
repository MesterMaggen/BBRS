import cv2 as cv
import numpy as np

image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/6.jpg", cv.IMREAD_COLOR)
#new_image = np.zeros((500, 500, 3),dtype=np.uint8)

def StretchedBGR(BGR_Image):
    HSV_Image = cv.cvtColor(BGR_Image, cv.COLOR_BGR2HSV)
    V_channel
    img_float = BGR_Image.astype(np.float32)
    min_val = np.min(img_float)
    max_val = np.max(img_float)
    stretched = 255 * (img_float - min_val) / (max_val - min_val)
    return stretched.astype(np.uint8)

sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  10, -1],
                              [-1, -1, -1]])

stretched_image = 
sharpened_image = cv.filter2D(src=image, ddepth=-1, kernel=sharpening_kernel)

lower_bound = np.array([160, 160, 160])
upper_bound = np.array([255, 255, 255])

masked_image = cv.inRange(sharpened_image, lower_bound, upper_bound)

maskedMedian_image = cv.medianBlur(masked_image,3)

cv.imshow('Original Image', image)
cv.imshow("HisStretched Image", stretched_image)
cv.imshow('Sharpened Image', sharpened_image)
cv.imshow('Masked Image', masked_image)
cv.imshow('Masked MedianFilter Image', maskedMedian_image)
cv.waitKey(0)
cv.destroyAllWindows()