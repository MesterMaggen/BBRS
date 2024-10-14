import cv2 as cv
import numpy as np
from collections import deque
import classifierAndPropertyCount as cp
import crownDetection as cd

with open("GroundTruth.DAT", 'r') as file:
    dat_values = [line.strip() for line in file]

trainAmount = 2

TestArray = np.zeros(74-trainAmount)

for j in range(trainAmount,74,1):
    imageText = "King Domino dataset/Cropped and perspective corrected boards/" + str(j) + ".jpg"

    image = cv.imread(imageText, cv.IMREAD_COLOR)

    Classified_array, property_list = cp.Classifier(image)

    # if j == trainAmount:
    #     print(Classified_array)

    score = cp.ScoreCounter(Classified_array, property_list, j)

    # print("total board score:", score)
    # print("")

    TestArray[j-trainAmount] = score

# print(TestArray)

correctScore = 0

# Share of correctly classified boards:

for i in range(trainAmount,74,1):



    if int(TestArray[i-trainAmount]) == int(dat_values[i-1]):
        correctScore += 1

    
CorrectBoardShare = round(correctScore / (74-trainAmount) * 100)

print(f"Board percentage: {CorrectBoardShare}%")

# Mean score error:

GroundTruthScoreSum = 0
AlgorithmScoreSum = 0

for i in range(trainAmount,74,1):
    GroundTruthScoreSum += int(dat_values[i-1])

    AlgorithmScoreSum += TestArray[i-trainAmount]

meanScoreError = round(abs(GroundTruthScoreSum - AlgorithmScoreSum) / (74-trainAmount))

print(f"Mean score error: {meanScoreError}")



