import cv2 as cv
import numpy as np
from collections import deque

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
    HSV_lower = np.array([24,140,140],np.uint8)
    #HSV_upper = np.array([32,255,255],np.uint8)
    HSV_upper = np.array([35,255,255],np.uint8)

    return np.all(HSV_lower <= HSV) and np.all(HSV <= HSV_upper)

def FindYellowBlobs(BGR_Image):
    HSV_Image = cv.cvtColor(BGR_Image,cv.COLOR_BGR2HSV)
    Masked_Image = np.zeros((500,500),np.uint8)
    Blobs = []

    for y, row in enumerate(HSV_Image):
        for x, pixel in enumerate(row):
            if WithinYellowBounds(pixel):
                #print("Found Yellow")
                coords = deque()
                Blob = []
                coords.append((y,x))
                while(len(coords) > 0):
                    position = coords.pop()
                    HSV_Image[position[0],position[1]] = np.zeros(3,np.uint8)
                    Masked_Image[position[0],position[1]] = 255
                    Blob.append(position)
                    
                    for i in range(-1, 2, 1):
                        for j in range(-1, 2, 1):
                            if i == 0 and j == 0:
                                continue
                            elif position[0]+i < 0 or position[0]+i > 499 or position[1]+j < 0 or position[1]+j > 499:
                                #print("Out Of bounds: position(",position[0]+i,position[1]+j,") i,j(",i,j,")")
                                continue
                            elif (not WithinYellowBounds(HSV_Image[position[0]+i,position[1]+j])):
                                #print("Not Yellow: position(",position[0]+i,position[1]+j,") i,j(",i,j,")")
                                continue
                            elif ([position[0]+i,position[1]+j]) in coords:
                                #print("Found already found Yellow: position(",position[0]+i,position[1]+j,") i,j(",i,j,")")
                                continue
                            else:
                                #print("Found Yellow Again: position(",position[0]+i,position[1]+j,") i,j(",i,j,")")
                                Masked_Image[position[0]+i,position[1]+j] = 255
                                coords.append([position[0]+i,position[1]+j])
                        try:
                            HSV_Image[position[0]+i,position[1]+j] = np.zeros(3,np.uint8) 
                        except:
                            continue
                Blobs.append(Blob)
            else: 
                #print("Not Yellow") 
                HSV_Image[y,x] = np.zeros(3,np.uint8)
    for i, Blob in enumerate(Blobs):
        if 120 >= len(Blob) >= 40:
            print(f"List {i} has {len(Blob)} coordinates.")
        else:
            for coord in Blob:
                Masked_Image[coord[0],coord[1]] = 0
    
    Blobs = np.concatenate(Blobs).tolist()

    return Masked_Image, Blobs

def FindYellowBlobsChat(BGR_Image):
    HSV_Image = cv.cvtColor(BGR_Image, cv.COLOR_BGR2HSV)
    Masked_Image = np.zeros((500, 500), np.uint8)
    Blobs = []
    Yellows = 0

    # Create a mask for yellow colors
    yellow_mask = cv.inRange(HSV_Image, np.array([24, 140, 140]), np.array([35, 255, 255]))
    
    # Finding coordinates where yellow is detected
    yellow_coords = np.column_stack(np.where(yellow_mask > 0))
    
    visited = set()  # Keep track of visited coordinates

    for (y, x) in yellow_coords:
        if (y, x) in visited:
            continue  # Skip already visited coordinates

        coords = deque()
        Blob = []
        coords.append((y, x))
        Yellows += 1  # Count found blobs

        while coords:
            position = coords.pop()
            visited.add(position)  # Mark as visited
            Masked_Image[position[0], position[1]] = 255
            Blob.append(position)

            # Explore neighbors
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    
                    neighbor = (position[0] + i, position[1] + j)

                    # Check bounds and yellow condition
                    if (0 <= neighbor[0] < Masked_Image.shape[0] and
                        0 <= neighbor[1] < Masked_Image.shape[1] and
                        neighbor not in visited and
                        WithinYellowBounds(HSV_Image[neighbor[0], neighbor[1]])):
                        
                        coords.append(neighbor)

        # Only keep blobs within a certain size range
        if 40 <= len(Blob) <= 120:
            Blobs.append(Blob)
            print(f"Found yellow blob with {len(Blob)} coordinates.")
        else:
            # If blob is too small, clear the mask for its coordinates
            for coord in Blob:
                Masked_Image[coord[0], coord[1]] = 0

    print("Found", Yellows, "yellow blobs.")
    
    Blobs = np.concatenate(Blobs).tolist() if Blobs else []

    return Masked_Image, Blobs

for j in range(10,11,1):
    imageText = "King Domino dataset/Cropped and perspective corrected boards/" + str(j) + ".jpg"
    image = cv.imread(imageText, cv.IMREAD_COLOR)
    print("Image:",j)
    stretched_image = StretchedBGR(image)

    sharpening_kernel = np.array([[-1, -1, -1], [-1,  10 , -1], [-1, -1, -1]])
    sharpened_image = cv.filter2D(src=stretched_image, ddepth=-1, kernel=sharpening_kernel)

    lower_bound = np.array([155]*3)
    upper_bound = np.array([255, 255, 255])
    masked_image = cv.inRange(sharpened_image, lower_bound, upper_bound)

    blobbed_image, search = FindYellowBlobs(stretched_image)
    
    template = cv.imread("CrownTemplate.jpg", cv.IMREAD_GRAYSCALE)
    matched_image = stretched_image.copy()
    for i in range(4):
        templated_image = cv.matchTemplate(masked_image, template, cv.TM_CCOEFF_NORMED)
        #locations = np.where(templated_image >= 0.35)
        locations = np.where(templated_image >= 0.26)
        h, w = template.shape[:2]
        
        for pt in zip(*locations[::-1]):
            cv.rectangle(matched_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        template = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)

    # cv.imshow('Original Image', image)
    cv.imshow("Stretched Image", stretched_image)
    # cv.imshow('Sharpened Image', sharpened_image)
    # cv.imshow("Gray Image", gray_image)
    # cv.imshow('Edges', edges)           
    cv.imshow('Blobbed Image', blobbed_image)
    cv.imshow('Masked Image', masked_image)
    cv.imshow('Matched Image', matched_image)

    cv.waitKey()
    cv.destroyAllWindows()