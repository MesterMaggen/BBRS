import cv2 as cv
import numpy as np

image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/2.jpg", cv.IMREAD_COLOR)
new_image = np.zeros((500, 500, 3),dtype=np.uint8)

sharpening_kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
sharpened_image = cv.filter2D(src=image, ddepth=-1, kernel=sharpening_kernel)

gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
gray_sharp_image = cv.cvtColor(sharpened_image,cv.COLOR_BGR2GRAY)

c = 255/(1.05**np.max(gray_image)-1)

for x in range(500):
    for y in range(500):
        new_image[y,x] = c*(1.05**gray_image[y,x]-1)

cv.imshow('Original Image', image)
cv.imshow('Gray Image', gray_image)
cv.imshow('GrayMapping Image', new_image)
cv.waitKey(0)
cv.destroyAllWindows()