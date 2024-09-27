import cv2 as cv
import numpy as np
from collections import deque


image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/2.jpg", cv.IMREAD_COLOR)

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
    
    # Light green as plains
    if 28 <= hue <= 65 and value >= 100 and saturation > 160: 
        return 'plains'
    
    # Dark green as forest
    elif 28 <= hue <= 50 and saturation > 100 and value < 80: 
        return 'forest'
    
    # "brown" as wasteland
    elif 18 <= hue <= 30 and saturation < 170 and value > 80:
        return 'wasteland'
    
    # Blue tones as ocean
    elif 90 <= hue < 120 and saturation > 80: 
        return 'ocean'
    
    # yellow tones as desert
    elif 18 <= hue < 30 and saturation > 200: 
        return 'desert'
    
    # black tones as mine
    elif 15 <= hue < 30 and saturation > 100 and value < 80: 
        return 'mine'
    
    
    else:
        return 'start tile'
    
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


new_image = np.zeros((500, 500, 3))

for tileRow in range(5):
    for tileColumn in range(5):
        new_image[tileRow*100:(tileRow+1)*100, tileColumn*100:(tileColumn+1)*100] = RGBAvg[tileRow, tileColumn]

# Save or display the new image
#cv.imshow("Averaged Image", new_image)
#cv.waitKey(0)
#cv.destroyAllWindows()

#property_array = np.array([], dtype=int)
property_array = np.zeros((5,5))
property_count = 1
inProperty = False

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def create_property(array,i,j,tile_type,property_ID):
    rows, cols = array.shape
    queue = deque([(i, j)])
    
    property_array[i,j] = property_ID

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # If the neighboring tile is within bounds and matches the type, assign it the same property ID
            if 0 <= nx < rows and 0 <= ny < cols and array[nx, ny] == tile_type and property_array[nx, ny] == 0:
                property_array[nx,ny] = property_ID
                queue.append((nx, ny))

def property_counter(array):
    global property_array
    global property_count

    rows, cols = array.shape


    for i in range(rows):
        for j in range(cols):
            if property_array[i,j] == 0:
                create_property(array,i,j,array[i,j],property_count)
                property_count += 1

property_counter(classification_array)

print(property_array)           

property_list = np.array([], dtype=int)

for i in range(1, len(np.unique(property_array))+1):
    property_list.append(np.count_nonzero(property_array == i))

print(property_list)
 