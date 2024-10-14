import cv2 as cv
import numpy as np
from collections import deque
import classifierAndPropertyCount as cp
import crownDetectionLoop as cdl

with open("GroundTruth.DAT", 'r') as file:
    dat_values = [line.strip() for line in file]

trainAmount = 1

TestArray = np.zeros(75-trainAmount)

for j in range(trainAmount,75,1):
    imageText = "King Domino dataset/Cropped and perspective corrected boards/" + str(j) + ".jpg"

    image = cv.imread(imageText, cv.IMREAD_COLOR)

    Classified_array, property_list = cp.Classifier(image)

    score = cp.ScoreCounter(Classified_array, property_list, j)

    # print("total board score:", score)
    # print("")

    TestArray[j-trainAmount] = score

print(TestArray)

correctScore = 0

# Share of correctly classified boards:

for i in range(trainAmount,75,1):
    if TestArray[i-20] == dat_values[i-1]:
        correctScore += 1
    
CorrectBoardShare = correctScore / (75-trainAmount) * 100

print(CorrectBoardShare)

# Mean score error:

GroundTruthScoreSum = 0
AlgorithmScoreSum = 0

for i in range(trainAmount,75,1):
    GroundTruthScoreSum += int(dat_values[i-1])

    AlgorithmScoreSum += TestArray[i-trainAmount]

meanScoreError = abs(GroundTruthScoreSum - AlgorithmScoreSum) / (75-trainAmount)

print(meanScoreError)
