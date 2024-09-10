import cv2 as cv
import numpy as np

image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/1.jpg", cv.IMREAD_COLOR)

RGBSum = np.zeros((5,5,3))

cv.split

for tileRow in range(5):
    for tileColumn in range(5):
        tempimg = image[tileRow*100:(tileRow+1)*100, tileColumn*100:(tileColumn+1)*100]
        print("First square")
        for row in tempimg:
            for pixel in row:
                #print(pixel)
                RGBSum[tileRow,tileColumn, 0] += pixel[0]
                RGBSum[tileRow,tileColumn, 1] += pixel[1]
                RGBSum[tileRow,tileColumn, 2] += pixel[2]
        
RGBSum = RGBSum/10000
print(RGBSum)

'''
for row in image:
    for pixel in row:
        print(f"Pixel value: {pixel}")
'''


