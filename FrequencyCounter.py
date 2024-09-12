import cv2 as cv
import numpy as np
from collections import Counter

image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/9.jpg", cv.IMREAD_COLOR)
new_image = np.zeros((500, 500, 3),dtype=np.uint8)

for tileRow in range(5):
    for tileColumn in range(5):
        tempimg = image[tileRow*100:(tileRow+1)*100, tileColumn*100:(tileColumn+1)*100]

        for y, row in enumerate(tempimg):
            for x, pixel in enumerate(row):
                tempimg[y,x,0] = (round(np.float64(pixel[0]) / 9) * 9)
                tempimg[y,x,1] = (round(np.float64(pixel[1]) / 9) * 9)
                tempimg[y,x,2] = (round(np.float64(pixel[2]) / 9) * 9)
        
        tempimg = [tuple(tempimg[i][j]) for i in range(100) for j in range(100)]
        counted = Counter(tempimg)
        most_common_object, most_common_count = counted.most_common(2)[0]
        print("Tile[",tileRow,",",tileColumn,"]:",counted.most_common(2))
        new_image[tileRow*100:(tileRow+1)*100, tileColumn*100:(tileColumn+1)*100] = most_common_object

cv.imshow("Original Image", image)
cv.imshow("New RGB", new_image)
#cv.imshow("New HSV", cv.cvtColor(new_image, cv.COLOR_BGR2HSV))
cv.waitKey(0)
cv.destroyAllWindows()