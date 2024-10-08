import cv2 as cv
import numpy as np
from collections import deque


image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/4.jpg", cv.IMREAD_COLOR)

#define RGBsum array to store the sum of RGB values of each tile

RGBSum = np.zeros((5,5,3))

# Find the sum of all RBG values in all 25 100x100 pixel tiles

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

# divide the sum of RGB values by 10000 to get the average RGB value of each tile

RGBAvg = np.round(RGBSum/10000)

# HSVAvg array to store the average HSV values of each tile

#Classification of the tiles with HSV thresholding:

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
    
classification_array = np.zeros((5,5), dtype=object)

# Classifier function to classify each tile

def tile_classifier(RGBAvg):

    for tileRow in range(5):
        for tileColumn in range(5):
            rgb_value = np.array([[RGBAvg[tileRow, tileColumn]]], dtype=np.uint8)

            hsv_tile = cv.cvtColor(rgb_value, cv.COLOR_BGR2HSV)
            hue, saturation, value = hsv_tile[0,0]

            classification = tile_thresholder(hue, saturation, value)

            classification_array[tileRow, tileColumn] = classification

    return classification_array

tile_classifier(RGBAvg)

print(classification_array)

property_array = np.zeros((5,5))
property_count = 1
inProperty = False

# directions for "flood fill" algorithm to find connected tiles of the same type
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Function to find connected tiles of the same type (properties):

def create_property(array,i,j,tile_type,property_ID):
    rows, cols = array.shape
    queue = deque([(i, j)])
    
    property_array[i,j] = property_ID

    #flood fill algorithm:

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

# array with properties and crowns in each property
property_list = np.zeros([2, len(np.unique(property_array))], dtype=int)

# add each property and crowns in the property to the property_list array
for i in range(1, len(np.unique(property_array))):
    property_list[0,i] = np.count_nonzero(property_array == i)

print(property_list)

# function to count the score of the board:

def ScoreCounter(property_list, classification_array):
    score = 0

    # property size * crowns in property for each property is added to the score
    for i in range(1, len(np.unique(property_array))):
        score += property_list[0,i] * property_list[1,i]

    # 10 additional points if the start tile is in the center of the board
    if (classification_array[2,2] == 'start tile'):
        score += 10

    return score

print(ScoreCounter(property_list, classification_array))