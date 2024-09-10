import cv2 as cv
import numpy as np
from collections import Counter

image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/3.jpg", cv.IMREAD_COLOR)
new_image = np.zeros((500, 500, 3),dtype=np.uint8)

for tileRow in range(5):
    for tileColumn in range(5):
        tempimg = image[tileRow*100:(tileRow+1)*100, tileColumn*100:(tileColumn+1)*100]
        tempimg = [tuple(tempimg[i][j]) for i in range(100) for j in range(100)]
        counted = Counter(tempimg)
        most_common_object, most_common_count = counted.most_common(2)[1]
        print(most_common_object)
        new_image[tileRow*100:(tileRow+1)*100, tileColumn*100:(tileColumn+1)*100] = most_common_object
        #print("(", tileRow, ",", tileColumn, "):", most_common_object)

#cv.imshow("Forest square", new_image[0:100,300:400])
#print("Forest square", new_image[50,350])
#cv.imshow("Mine square", new_image[200:300,0:100])
#print("Mine square", new_image[250,50])

# Save or display the new image
cv.imshow("Original Image", image)
cv.imshow("New Image", new_image)
cv.waitKey(0)
cv.destroyAllWindows()