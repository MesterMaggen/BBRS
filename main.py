import cv2 as cv
import numpy as np
from collections import deque
import classifierAndPropertyCount as cp

def ScoreCount(image):

    cp.tile_classifier(image)



for j in range(1,60,1):
    imageText = "King Domino dataset/Cropped and perspective corrected boards/" + str(j) + ".jpg"

    image = cv.imread(imageText, cv.IMREAD_COLOR)

    Classified_array, property_list = cp.Classifier(image)

    score = cp.ScoreCounter(Classified_array, property_list)

    print(score)


    