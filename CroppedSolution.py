import cv2 as cv
import numpy as np

image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/3.jpg", cv.IMREAD_COLOR)

RGBSum = np.zeros((5,5,3))

for tileRow in range(5):
    for tileColumn in range(5):
        tempimg = image[tileRow*100:(tileRow+1)*100, tileColumn*100:(tileColumn+1)*100]
        #print("First square")
        for row in tempimg:
            for pixel in row:
                #print(pixel)
                RGBSum[tileRow,tileColumn, 0] += pixel[0]
                RGBSum[tileRow,tileColumn, 1] += pixel[1]
                RGBSum[tileRow,tileColumn, 2] += pixel[2]


RGBAvg = np.round(RGBSum/10000)
#print(RGBAvg)

HSVAvg = np.zeros((5,5,3), dtype=np.uint8)

def tile_classifier(hue, saturation, value):
    if value < 50:
        return 'mine'
    elif 10 <= hue <= 30 and saturation > 50 and value < 200:
        return 'wasteland'
    elif 30 <= hue < 90:
        return 'desert'
    elif 60 <= hue <= 90 and value >= 170:
        return 'plains'
    elif 60 <= hue <= 90 and value < 170:
        return 'forest'
    elif 150 <= hue < 240:
        return 'ocean'
    
classification_array = np.zeros((5,5), dtype=object)

for tileRow in range(5):
    for tileColumn in range(5):
        rgb_value = np.array([[RGBAvg[tileRow, tileColumn]]], dtype=np.uint8)

        #print(f"BGR value at ({tileRow}, {tileColumn}): {rgb_value}")
        
        
        hsv_tile = cv.cvtColor(rgb_value, cv.COLOR_BGR2HSV)
        hue, saturation, value = hsv_tile[0,0]

        classification = tile_classifier(hue, saturation, value)

        classification_array[tileRow, tileColumn] = classification

        saturation_online = (saturation / 255) * 100
        value_online = (value / 255) * 100

        #print(f"Tile at ({tileRow}, {tileColumn}): OpenCV HSV = {hsv_tile[0][0]}, "
        #      f"Online HSV = (H: {hue}, S: {saturation_online:.2f}, V: {value_online:.2f}), "
        #      f"Classification = {classification}")

print(classification_array)


new_image = np.zeros((500, 500, 3),dtype=np.uint8)


for tileRow in range(5):
    for tileColumn in range(5):
        new_image[tileRow*100:(tileRow+1)*100, tileColumn*100:(tileColumn+1)*100] = RGBAvg[tileRow, tileColumn]

# Save or display the new image
cv.imshow("Original Image", image)
cv.imshow("New Image", new_image)
cv.waitKey(0)
cv.destroyAllWindows()