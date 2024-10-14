import cv2 as cv
import numpy as np
from collections import deque


#image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/4.jpg", cv.IMREAD_COLOR)

#define RGBsum array to store the sum of RGB values of each tile

def RBGimage(image):

    RGBAvg = np.zeros((5,5,3))

    for tileRow in range(5):
        for tileColumn in range(5):
            tempimg = image[tileRow*100:(tileRow+1)*100, tileColumn*100:(tileColumn+1)*100]
            RGBAvg[tileRow,tileColumn] = np.mean(tempimg, axis=(0,1))

    RGBAvg = np.clip(RGBAvg, 0, 255).astype(np.uint8)

    RGBAvg_scaled = cv.resize(RGBAvg, (500, 500), interpolation=cv.INTER_NEAREST)

    # Display the image
    #cv.imshow("RGBAvg", RGBAvg_scaled)
    #cv.waitKey()
    #cv.destroyAllWindows()            

    # divide the sum of RGB values by 10000 to get the average RGB value of each tile

    return np.round(RGBAvg)

def tile_thresholder(hue, saturation, value):
    
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
    
    # else it is the start tile or no tile at all
    # classified as start tile in both cases for simplification
    
    else:
        return 'start tile'

def tile_classifier(image):
    
    classification_array = np.zeros((5,5), dtype=object)

    RGBAvg = RBGimage(image)

    for tileRow in range(5):
        for tileColumn in range(5):
            rgb_value = np.array([[RGBAvg[tileRow, tileColumn]]], dtype=np.uint8)

            hsv_tile = cv.cvtColor(rgb_value, cv.COLOR_BGR2HSV)
            hue, saturation, value = hsv_tile[0,0]

            classification = tile_thresholder(hue, saturation, value)

            classification_array[tileRow, tileColumn] = classification

    #print(classification_array)

    return classification_array

property_array = np.zeros((5,5))

# Function to find connected tiles of the same type (properties):

def create_property(array,i,j,tile_type,property_ID, property_array):
    rows, cols = array.shape
    queue = deque([(i, j)])    
    property_array[i,j] = property_ID

    #flood fill algorithm:

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # If the neighboring tile is within bounds and matches the type, assign it the same property ID
            if 0 <= nx < rows and 0 <= ny < cols and array[nx, ny] == tile_type and property_array[nx, ny] == 0:
                property_array[nx,ny] = property_ID
                queue.append((nx, ny))

    return property_array

def property_counter(array):
    property_array = np.zeros(array.shape)
    property_count = 1
    rows, cols = array.shape

    for i in range(rows):
        for j in range(cols):
            if property_array[i,j] == 0:
                create_property(array,i,j,array[i,j],property_count, property_array)
                property_count += 1

    property_list = np.zeros([2, len(np.unique(property_array))], dtype=int)

    # add each property and crowns in the property to the property_list array
    for i in range(1, len(np.unique(property_array))):
        property_list[0,i] = np.count_nonzero(property_array == i)

    return property_list

def ScoreCounter(classification_array, property_list):
    score = 0

    # property size * crowns in property for each property is added to the score
    for i in range(1, len(np.unique(property_array))):

        score += property_list[0,i] * property_list[1,i]

    print("tile Score: " + str(score))

    # 10 additional points if the start tile is in the center of the board
    if (classification_array[2,2] == 'start tile'):
        score += 10

        print("Start tile in the center of the board")

    UnknownCount = np.sum(classification_array == 'start tile')
    if UnknownCount == 1:
        score += 5

        print("full board")

    print(Classified_array)

    return score

def Classifier(image):

    Classified_array = tile_classifier(image)

    property_list = property_counter(Classified_array)

    return Classified_array, property_list
