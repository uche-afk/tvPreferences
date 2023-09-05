import numpy as np
import csv
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def trainLR():
    global trainFilename
    global testFilename
    global demographicModifier
    global positiveCount
    global negativeCount
    global numTrainingData
    global numTraits
    global theta

    trainFilename = "netflix-train.csv"
    testFilename = "netflix-test.csv"
    demographicModifier = 1 # set to 1 if the data has a demographic column
    positiveCount = 0
    negativeCount = 0
    numTrainingData = 0
    numTraits = 0
    STEPS = 100
    LEARNINGRATE = 0.00625

    dataArray = np.genfromtxt(trainFilename, delimiter=',', skip_header=1, dtype=np.uint8)
    numTraits = len(dataArray[0]) - demographicModifier - 1
    theta = np.zeros(numTraits + 1)

    LLbefore = 0
    for i in range(len(dataArray)):
        currentData = dataArray[i]
        currentData = np.concatenate((np.array([1]), currentData))
        currentTraits = currentData[:(-1 - demographicModifier)].copy()
        label = currentData[-1]

        LLbefore += (label * math.log(sigmoid(np.dot(theta, currentTraits)))) + ((1 - label) * math.log(1 - sigmoid(np.dot(theta, currentTraits))))

    for i in range(STEPS):
        gradient = np.zeros(numTraits + 1)
        for j in range(len(dataArray)):
            currentData = dataArray[j]
            currentData = np.concatenate((np.array([1]), currentData))
            currentTraits = currentData[:(-1 - demographicModifier)].copy()
            label = currentData[-1]
            for k in range(numTraits + 1):
                gradient[k] += ((label - sigmoid((np.dot(theta, currentTraits)))) * currentTraits[k])

        for j in range(len(theta)):
            theta[j] += LEARNINGRATE * gradient[j]

    LLafter = 0
    for i in range(len(dataArray)):
        currentData = dataArray[i]
        currentData = np.concatenate((np.array([1]), currentData))
        currentTraits = currentData[:(-1 - demographicModifier)].copy()
        label = currentData[-1]

        LLafter += (label * math.log(sigmoid(np.dot(theta, currentTraits)))) + ((1 - label) * math.log(1 - sigmoid(np.dot(theta, currentTraits))))

        print("theta", theta)



def testLR():
    global posTestSuccesses
    global negTestSuccesses
    global numPosTestingData
    global numNegTestingData

    posTestSuccesses = 0
    negTestSuccesses = 0
    numPosTestingData = 0
    numNegTestingData = 0

    dataArray = np.genfromtxt(testFilename, delimiter=',', skip_header=1, dtype=np.uint8)

    for i in range(len(dataArray)):
        currentData = dataArray[i]
        currentData = np.concatenate((np.array([1]), currentData))
        currentTraits = currentData[:(-1 - demographicModifier)].copy()
        if (sigmoid(np.dot(currentTraits, theta))) >= 0.5:
            prediction = 1
        else:
            prediction = 0

        realY = currentData[-1]

        if realY == 1:
            numPosTestingData += 1
            if prediction == realY:
                posTestSuccesses += 1
        if realY == 0:
            numNegTestingData += 1
            if prediction == realY:
                negTestSuccesses += 1
            

def printResults():
    print("Class 0: tested ", numNegTestingData, " correctly classified ", negTestSuccesses)
    print("Class 1: tested ", numPosTestingData, " correctly classified ", posTestSuccesses)
    print("Overall: tested ", numNegTestingData + numPosTestingData, " correctly classified ", negTestSuccesses + posTestSuccesses)
    accuracy = (negTestSuccesses + posTestSuccesses) / (numNegTestingData + numPosTestingData)
    print("Accuracy: ", accuracy)


def main():
    trainLR()
    testLR()
    printResults()

if __name__ == '__main__':
    main()