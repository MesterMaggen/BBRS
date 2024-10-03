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

def FindYellowBlobs(BGR_Image):
    HSV_Image = cv.cvtColor(BGR_Image,cv.COLOR_BGR2HSV)
    Masked_Image = np.zeros((500,500),np.uint8)
    
    HSV_lower = np.array([27,127,153],np.uint8)
    HSV_upper = np.array([34,255,255],np.uint8)

    for y, row in enumerate(HSV_Image):
        for x, pixel in enumerate(row):
            if 26 < pixel[0] < 35 and 127 < pixel[1] < 255 and 153 < pixel[1] < 255:
                coords = deque()
                coords.append((y,x))
                while(len(coords) > 0):
                    HSV_Image[y,x] = np.zeros(3,np.uint8)
                    position = coords.pop()
                    
                    for i in range(-1, 1, 1):
                        for j in range(-1, 1, 1):
                            if 0 <= position[0]+i <= 500 and 0 <= position[1]+j <= 500:
                                
                            else:
                                continue

                        

            else: 
                HSV_Image[y,x] = np.zeros(3,np.uint8)
                Masked_Image[y,x] = 0


    # image = cv.cvtColor(image,cv.COLOR_BGR2HSV)
    # print("Color:", image[25,25])
    # image[0:50,0:50] = HSV_upper
    # image = cv.cvtColor(image,cv.COLOR_HSV2BGR)


for j in range(1,2,1):
    imageText = "King Domino dataset/Cropped and perspective corrected boards/" + str(j) + ".jpg"
    image = cv.imread(imageText, cv.IMREAD_COLOR)
    print("Image:",j)
    stretched_image = StretchedBGR(image)

    sharpening_kernel = np.array([[-1, -1, -1], [-1,  10 , -1], [-1, -1, -1]])
    #sharpening_kernel = np.array([[ 0, -1,  0], [-1,   7, -1], [ 0, -1,  0]])
    sharpened_image = cv.filter2D(src=stretched_image, ddepth=-1, kernel=sharpening_kernel)

    lower_bound = np.array([155]*3)
    upper_bound = np.array([255, 255, 255])
    masked_image = cv.inRange(sharpened_image, lower_bound, upper_bound)

    FindYellowBlobs(stretched_image)
    
    # morphology_kernel = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE,ksize=(5,4))
    # morphology_kernel = np.array([[1]*3]*3)
    # print(morphology_kernel)
    # masked_image = cv.dilate(masked_image, morphology_kernel)
    # masked_image = cv.erode(masked_image, morphology_kernel)

    # gray_image = cv.cvtColor(sharpened_image,cv.COLOR_BGR2GRAY)
    # blurred_image = cv.GaussianBlur(gray_image, (3, 3), 1.0)
    # edges = cv.Canny(blurred_image, 50, 200)
    
    template = cv.imread("CrownTemplate.jpg", cv.IMREAD_GRAYSCALE)
    matched_image = stretched_image.copy()
    for i in range(4):  
        templated_image = cv.matchTemplate(masked_image, template, cv.TM_CCOEFF_NORMED)
        locations = np.where(templated_image >= 0.35)
        h, w = template.shape[:2]
        
        for pt in zip(*locations[::-1]):
            cv.rectangle(matched_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        template = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)

    # cv.imshow('Original Image', image)
    # cv.imshow("Stretched Image", stretched_image)
    # cv.imshow('Sharpened Image', sharpened_image)
    # cv.imshow("Gray Image", gray_image)
    # cv.imshow('Edges', edges)           
    cv.imshow('Masked Image', masked_image)
    cv.imshow('Matched Image', matched_image)

    cv.waitKey()
    cv.destroyAllWindows()