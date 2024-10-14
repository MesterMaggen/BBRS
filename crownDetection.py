import cv2 as cv
import numpy as np
 
image = cv.imread("King Domino dataset/Cropped and perspective corrected boards/2.jpg", cv.IMREAD_COLOR)

def StretchedBGR(BGR_Image):
    HSV_Image = cv.cvtColor(BGR_Image, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(HSV_Image)
    
    v_float = v.astype(np.float32)
    v_min = np.min(v_float)
    v_max = np.max(v_float)
    
    v_stretched = 255 * (v_float - v_min) / (v_max - v_min)
    
    v_stretched = v_stretched.astype(np.uint8)
    
    hsv_stretched = cv.merge([h, s, v_stretched])

    return cv.cvtColor(hsv_stretched, cv.COLOR_HSV2BGR)

def CrownDetection(image):
    stretched_image = StretchedBGR(image)

    sharpening_kernel = np.array([[-1, -1, -1], [-1,  10 , -1], [-1, -1, -1]])
    sharpened_image = cv.filter2D(src=stretched_image, ddepth=-1, kernel=sharpening_kernel)

    lower_bound = np.array([155]*3)
    upper_bound = np.array([255]*3)
    masked_image = cv.inRange(sharpened_image, lower_bound, upper_bound)

    template = cv.imread("CrownTemplate.jpg", cv.IMREAD_GRAYSCALE)
    matched_image = image.copy()
    filtered_matches = []

    for i in range(4):
        templated_image = cv.matchTemplate(masked_image, template, cv.TM_CCOEFF_NORMED)
        locations = np.where(templated_image >= 0.34)
        
        matches = list(zip(*locations[::-1]))  # This gives [(x1, y1), (x2, y2), ...]

        # Get the corresponding match scores for each location
        scores = templated_image[locations]

        sorted_idxs = np.argsort(scores)[::-1]  # Sort in descending order of score

        for idx in sorted_idxs:
            current_match = matches[idx]
            current_match = (current_match[0], current_match[1])
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
                cv.rectangle(matched_image, current_match, (current_match[0] + w, current_match[1] + h), (0, 255, 0), 2)

        template = cv.rotate(template, cv.ROTATE_90_CLOCKWISE)

    crownArray = np.zeros((5,5),np.uint8)

    for match in filtered_matches:
        crownArray[match[1]//100,match[0]//100] += 1

    return crownArray