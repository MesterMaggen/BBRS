import cv2 as cv
import numpy as np
from collections import deque
import crownDetection as cd

def histogramStrech(channel):
    min_val = np.min(channel)
    max_val = np.max(channel)

    stretched_channel = 255 * ((channel - min_val) / (max_val - min_val))

    stretched_channel = stretched_channel.astype(np.uint8)

    return stretched_channel

def histogramFunction(image):
    
    r,g,b = cv.split(image)

    r_strecthed = histogramStrech(r)
    g_strecthed = histogramStrech(g)
    b_strecthed = histogramStrech(b)

    stretched_image = cv.merge([r_strecthed, g_strecthed, b_strecthed])

    return stretched_image

def colorNormalizer(image):

    imgFloat = image.astype(np.float32)

    sumChannels = np.sum(imgFloat, axis=(2))

    sumChannels[sumChannels == 0] = 0.01

    r_normalized = imgFloat[:,:,0] / sumChannels
    g_normalized = imgFloat[:,:,1] / sumChannels
    b_normalized = imgFloat[:,:,2] / sumChannels

    normalized_image = cv.merge([b_normalized, g_normalized, r_normalized])

    normalized_image = (normalized_image * 255).astype(np.uint8)

    return normalized_image

def gaussianBlur(image, kernel_size):

    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)


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
    
    print(hue, saturation, value)

    # Light green as plains
    if 28 <= hue <= 65 and saturation > 100 and value >= 82: 
        return 'plains'
    
    # Dark green as forest
    elif 25 <= hue <= 80 and saturation > 70 and value < 75: 
        return 'forest'

  # yellow tones as desert
    elif 22 <= hue < 30 and saturation > 200 and value > 105: 
        return 'desert'

    # "brown" as wasteland
    elif 18 <= hue <= 30 and saturation < 200 and value > 82:
        return 'wasteland'
    
    # Blue tones as ocean
    elif 90 <= hue < 120 and saturation > 80: 
        return 'ocean'
    
    # black tones as mine
    elif 15 <= hue < 30 and saturation > 60 and value < 77: 
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
    
    RGBavg_scaled = cv.resize(RGBAvg, (500, 500), interpolation=cv.INTER_NEAREST)

    # cv.imshow("Image", RGBavg_scaled)

    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return classification_array


# Function to find connected tiles of the same type (properties):
property_array = np.zeros((5,5))


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

def property_counter(classified_array, crownArray):
    property_array = np.zeros(classified_array.shape)
    property_count = 1
    rows, cols = classified_array.shape

    combined_array = np.zeros((rows,cols,2), dtype=object)

    for i in range(rows):
        for j in range(cols):
            combined_array[i,j] = [classified_array[i,j], crownArray[i,j]]

    for i in range(rows):
        for j in range(cols):
            if property_array[i,j] == 0:
                create_property(classified_array,i,j,classified_array[i,j],property_count, property_array)
                property_count += 1

    unique_properties = np.unique(property_array)
    unique_properties = unique_properties[unique_properties > 0]

    property_list = np.zeros([2, len(unique_properties)], dtype=int)

    # add each property and crowns in the property to the property_list array
    for i, prop in enumerate(unique_properties):
        property_list[0, i] = np.count_nonzero(property_array == prop)
        property_list[1, i] = np.sum(crownArray[property_array == prop])

    # print(property_array)


    return property_list

def ScoreCounter(classification_array, property_list, imageNr):
    score = 0

    print("Image: " + str(imageNr))

    # property size * crowns in property for each property is added to the score
    for i in range(0, property_list.shape[1]):

        score += (property_list[0,i] * property_list[1,i])

    # print(property_list)
    # print("tile Score: " + str(score))

    # 10 additional points if the start tile is in the center of the board
    if (classification_array[2,2] == 'start tile'):
        score += 10

        # print("Start tile in the center of the board")

    UnknownCount = np.sum(classification_array == 'start tile')
    if UnknownCount == 1:
        score += 5

        # print("full board")

    return score

def Classifier(image):

    # strechedImage = histogramFunction(image)

    # normalizedImage = colorNormalizer(strechedImage)

    # blurredImage = gaussianBlur(strechedImage, 5)

    # RGBimage = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    Classified_array = tile_classifier(image)

    property_list = property_counter(Classified_array, cd.CrownDetection(image))

    # print(property_list)

    return Classified_array, property_list
