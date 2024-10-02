import cv2 as cv
import numpy as np

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

for j in range(1,11,1):
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

    template = cv.imread("CrownTemplate.jpg", cv.IMREAD_GRAYSCALE)
    matched_image = stretched_image.copy()
    for i in range(4):  
        templated_image = cv.matchTemplate(masked_image, template, cv.TM_CCOEFF_NORMED)
        locations = np.where(templated_image >= 0.35)
        h, w = template.shape[:2]
        
        for pt in zip(*locations[::-1]):
            cv.rectangle(matched_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        template = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)

    cv.imshow('Original Image', image)
    cv.imshow("Stretched Image", stretched_image)
    cv.imshow('Sharpened Image', sharpened_image)
    cv.imshow('Masked Image', masked_image)
    cv.imshow('Matched Image', matched_image)

    cv.waitKey()
    cv.destroyAllWindows()