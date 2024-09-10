import cv2 as cv
import numpy as np

image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg", cv.IMREAD_COLOR)

image2 = cv.resize(image, (0, 0), fx = 0.1, fy = 0.1)

cv.imshow("Image",image)
cv.imshow("Image2",image2)

cv.waitKey(0)