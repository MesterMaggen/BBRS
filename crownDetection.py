import cv2 as cv
import numpy as np

image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg", cv.IMREAD_COLOR)
#new_image = np.zeros((500, 500, 3),dtype=np.uint8)

sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  10, -1],
                              [-1, -1, -1]])

sharpened_image = cv.filter2D(src=image, ddepth=-1, kernel=sharpening_kernel)

lower_bound = np.array([160, 160, 160])
upper_bound = np.array([255, 255, 255])

masked_image = cv.inRange(sharpened_image, lower_bound, upper_bound)

cv.imshow('Original Image', image)
#cv.imshow('Sharpened Image', sharpened_image)
cv.imshow('Masked Image', masked_image)
cv.waitKey(0)
cv.destroyAllWindows()