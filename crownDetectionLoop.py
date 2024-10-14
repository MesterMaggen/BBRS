import cv2 as cv
import numpy as np
from collections import deque
import time

def StretchedBGR(BGR_Image):
    HSV_Image = cv.cvtColor(BGR_Image, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(HSV_Image)
    
    v_float = v.astype(np.float32)
    v_min = np.min(v_float)
    v_max = np.max(v_float)
    
    v_stretched = 255 * (v_float - v_min) / (v_max - v_min)
    #v_stretched = cv.equalizeHist(v)
    
    v_stretched = v_stretched.astype(np.uint8)
    
    hsv_stretched = cv.merge([h, s, v_stretched])

    return cv.cvtColor(hsv_stretched, cv.COLOR_HSV2BGR)

def WithinYellowBounds(HSV):
    #HSV_lower = np.array([24,150,150],np.uint8)
    HSV_lower = np.array([24,110,110],np.uint8)
    #HSV_upper = np.array([35,195,195],np.uint8)
    HSV_upper = np.array([31,235,235],np.uint8)

    return np.all(HSV_lower <= HSV) and np.all(HSV <= HSV_upper)

def BlobsAfterMorph(after_image):
    Masked_Image = np.zeros((500,500),np.uint8)
    
    num_labels_after, labels_after, stats_after, centroids_after = cv.connectedComponentsWithStats(after_image)

    # Find coordinates of blobs after closing
    Blobs = [np.column_stack(np.where(labels_after == i)) for i in range(1, num_labels_after)]

    for i, Blob in enumerate(Blobs):
        if 150 >= len(Blob) >= 40:
            del Blobs[i]
    
    for Blob in Blobs:
        for coord in Blob:
            Masked_Image[coord[0],coord[1]] = 255

    return Masked_Image

def FindYellowBlobs(BGR_Image):
    HSV_Image = cv.cvtColor(BGR_Image,cv.COLOR_BGR2HSV)
    Masked_Image = np.zeros((500,500),np.uint8)
    Blobs = []
    visited = set()

    for y, row in enumerate(HSV_Image):
        for x, pixel in enumerate(row):
            if WithinYellowBounds(pixel) and (y,x) not in visited:
                coords = deque()
                Blob = []
                coords.append((y,x))
                while(len(coords) > 0):
                    position = coords.pop()
                    visited.add((position[0],position[1]))
                    Masked_Image[position[0],position[1]] = 255
                    Blob.append(position)
                    
                    for i in range(-1, 2, 1):
                        for j in range(-1, 2, 1):
                            neighbor = (position[0] + i, position[1] + j)
                            if i == 0 and j == 0:
                                continue
                            elif not (0 <= neighbor[0] < 500 and 0 <= neighbor[1] < 500):
                                continue
                            elif (not WithinYellowBounds(HSV_Image[position[0]+i,position[1]+j])):
                                continue
                            elif ((neighbor[0],neighbor[1])) in visited:
                                continue
                            else:
                                Masked_Image[neighbor[0],neighbor[1]] = 255
                                coords.append([neighbor[0],neighbor[1]])
                                visited.add((neighbor[0],neighbor[1])) 
                Blobs.append(Blob)

    # cv.imshow('Bush Image', Masked_Image)

    kernel = np.ones((3,3),np.uint8)
    opened_image = cv.morphologyEx(Masked_Image, cv.MORPH_OPEN, kernel)

    opened_filtered_image = BlobsAfterMorph(opened_image)

    for i, Blob in enumerate(Blobs):
        if not (150 >= len(Blob) >= 40):
            for coord in Blob:
                Masked_Image[coord[0],coord[1]] = 0
    

    # cv.imshow('BushFilter Image', Masked_Image)
    # cv.imshow('BushOpened Image', blobbed_image)

    Blobs = np.concatenate(Blobs).tolist()

    return Masked_Image #Masked_Image

def CreateFilterImage():
    image = np.zeros((500, 500), dtype=np.uint8)

    for i in range(5):
        for j in range(5):
            top_left = (100*i, 100*j)  
            bottom_right = (100*(i+1), 100*(j+1))
            cv.rectangle(image, top_left, bottom_right, (255), thickness=55)
    
    return image

for j in range(1,32,1):
    imageText = "King Domino dataset/Cropped and perspective corrected boards/" + str(j) + ".jpg"
    image = cv.imread(imageText, cv.IMREAD_COLOR)
    # print("Image:",j)
    stretched_image = StretchedBGR(image)
 
    sharpening_kernel = np.array([[-1, -1, -1], [-1,  10 , -1], [-1, -1, -1]])
    sharpened_image = cv.filter2D(src=stretched_image, ddepth=-1, kernel=sharpening_kernel)

    lower_bound = np.array([155]*3)
    upper_bound = np.array([255, 255, 255])
    masked_image = cv.inRange(sharpened_image, lower_bound, upper_bound)

    blobbed_image1 = FindYellowBlobs(stretched_image)

    kernel = np.ones((9,9),np.uint8)
    blobbed_image = cv.morphologyEx(blobbed_image1, cv.MORPH_CLOSE, kernel)

    num_labels_before, labels_before, stats_before, centroids_before = cv.connectedComponentsWithStats(blobbed_image1)
    num_labels_after, labels_after, stats_after, centroids_after = cv.connectedComponentsWithStats(blobbed_image)

    blob_coords_before = [np.column_stack(np.where(labels_before == i)) for i in range(1, num_labels_before)]
    blob_coords_after = [np.column_stack(np.where(labels_after == i)) for i in range(1, num_labels_after)]

    min_blobs = min(len(blob_coords_before), len(blob_coords_after))

    # for i in range(min_blobs):
    #     before_coords = blob_coords_before[i]
    #     after_coords = blob_coords_after[i]

    #     # Find newly added coordinates by comparing sets of coordinates
    #     added_coords = np.setdiff1d(after_coords, before_coords, assume_unique=True)
        
    #     if added_coords.size > 0:
    #         print(f"New coordinates added to blob {i}: {added_coords}")

    # # Handle merged blobs: Check if blobs before closing merged into a single blob after
    # for i in range(num_labels_after - 1):  # Skip the background
    #     overlapping_labels = np.unique(labels_before[labels_after == i + 1])
        
    #     if len(overlapping_labels) > 1:
    #         print(f"Blobs {overlapping_labels[1:]} merged into blob {i + 1}")

    template = cv.imread("CrownTemplate.jpg", cv.IMREAD_GRAYSCALE)
    matched_image = stretched_image.copy()
    filter_image = CreateFilterImage()
    #cv.imshow('Filter Image', filter_image)

    filtered_matches = []

    for i in range(4):
        templated_image = cv.matchTemplate(masked_image, template, cv.TM_CCOEFF_NORMED)

        pad_y = blobbed_image.shape[0] - templated_image.shape[0]
        pad_x = blobbed_image.shape[1] - templated_image.shape[1]
        
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top  # Remaining padding goes to the bottom
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left  # Remaining padding goes to the right
 
        # Apply padding using np.pad
        padded_template = np.pad(templated_image, 
                                ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                mode='constant', constant_values=0)

        locations = np.where((padded_template >= 0.20) & (blobbed_image == 255) & (filter_image == 255))
        matches = list(zip(*locations[::-1]))  # This gives [(x1, y1), (x2, y2), ...]

        # Get the corresponding match scores for each location
        scores = padded_template[locations]

        sorted_idxs = np.argsort(scores)[::-1]  # Sort in descending order of score

        for idx in sorted_idxs:
            current_match = matches[idx]
            current_match = (current_match[0] - pad_left, current_match[1] - pad_top)
            keep = True
            for selected_match in filtered_matches:
                # Calculate Euclidean distance between the current match and already selected matches
                distance = np.linalg.norm(np.array(current_match) - np.array(selected_match))
                if distance < template.shape[1]//2:
                    keep = False
                    break
            if keep:
                #filtered_matches.append(current_match)
                filtered_matches.append(current_match)
                # print("Appended",current_match)
                # print("Filtered Matches",filtered_matches)
                h, w = template.shape[:2]
                
                for pt in filtered_matches:
                    cv.rectangle(matched_image, current_match, (current_match[0] + w, current_match[1] + h), (0, 255, 0), 2)
        
        template = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)
    
    crownArray = np.zeros((5,5),np.uint8)

    for match in filtered_matches:
        crownArray[match[1]//100,match[0]//100] += 1

    # cv.imshow('Original Image', image)
    # cv.imshow("Stretched Image", stretched_image)
    # cv.imshow('Sharpened Image', sharpened_image)
    # cv.imshow("Gray Image", gray_image)
    # cv.imshow('Edges', edges)        
    # cv.imshow('Blobbed Image1', blobbed_image1)   
    # cv.imshow('Blobbed Image', blobbed_image)
    # cv.imshow('Masked Image', masked_image)
    # cv.imshow('Matched Image', matched_image)

    cv.waitKey()
    cv.destroyAllWindows()