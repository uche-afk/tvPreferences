from distutils.command.clean import clean
import numpy as np
import csv

def trainNB():
    global lookupDict0
    global lookupDict1
    global trainFilename
    global testFilename
    global positiveCount
    global negativeCount
    global numTrainingData
    global numTraits

    lookupDict0 = {}
    lookupDict1 = {}
    trainFilename = "netflix-train.csv"
    testFilename = "netflix-test.csv"
    demographicModifier = 0 # set to 1 if the data has a demographic column
    positiveCount = 0
    negativeCount = 0
    numTrainingData = 0
    numTraits = 0



    with open(trainFilename) as f:
        lines = f.readlines()
        numTrainingData = len(lines) - 1
        cleanLine = lines[1].strip('\n')
        list = cleanLine.split(',')
        numTraits = len(list) - 1 - demographicModifier
        positiveCount = 0
        negativeCount = 0

        cleanLine = lines[0].strip('\n')
        list = cleanLine.split(',')
        # print("Name of best predicting film: ", list[18])

        for i in range(numTraits):
            lookupDict0[i] = 0
            lookupDict1[i] = 0

        for i in range(1, len(lines)):
            cleanLine = lines[i].strip('\n')
            dataList = cleanLine.split(',')
            if int(dataList[len(dataList) - 1]) == 0:
                currentDict = lookupDict0
                negativeCount += 1
            else:
                currentDict = lookupDict1
                positiveCount += 1
            for j in range(numTraits):
                if int(dataList[j]) == 1:
                    currentDict[j] += 1

        for i in range(numTraits):  # LaPlace Smoothing
            lookupDict0[i] += 1
            lookupDict0[i] /= (negativeCount + 2)
            lookupDict1[i] += 1
            lookupDict1[i] /= (positiveCount + 2)

        print("lookupDict1", lookupDict1)

def testNB():
    positiveProb = positiveCount / numTrainingData
    # print("positiveProb:", positiveProb)
    negativeProb = negativeCount / numTrainingData

    global posTestSuccesses
    global negTestSuccesses
    global numPosTestingData
    global numNegTestingData

    posTestSuccesses = 0
    negTestSuccesses = 0
    numPosTestingData = 0
    numNegTestingData = 0

    demographic0Count = 0
    demographic1Count = 0
    demographic0Affirm = 0
    demographic1Affirm = 0

    with open(testFilename) as f:
        lines = f.readlines()
        cleanLine = lines[1].strip('\n')
        list = cleanLine.split(',')
        for i in range(1, len(lines)):
            cleanLine = lines[i].strip('\n')
            dataList = cleanLine.split(',')

            product0 = 1
            product1 = 1
            for j in range(numTraits):
                if int(dataList[j]) == 1:
                    product0 *= lookupDict0[j]
                    product1 *= lookupDict1[j]
                else:
                    product0 *= 1 - lookupDict0[j]
                    product1 *= 1 - lookupDict1[j]
                
            product0 *= negativeProb
            product1 *= positiveProb

            prediction = 0

            if product1 >= product0:
                prediction = 1
            
            realY = int(dataList[len(dataList) - 1])

        
            if int(dataList[-2]) == 1:
                demographic1Count += 1
                if prediction == 1:
                    demographic1Affirm += 1
            else:
                demographic0Count += 1
                if prediction == 1:
                    demographic0Affirm += 1

            if int(realY) == 1:
                numPosTestingData += 1
                if prediction == int(realY):
                    posTestSuccesses += 1
            if int(realY) == 0:
                numNegTestingData += 1
                if prediction == int(realY):
                    negTestSuccesses += 1

        print("demographic 1 ratio:", demographic1Affirm/demographic1Count)
        print("demographic 0 ratio:", demographic0Affirm/demographic0Count)

    """
    for i in range(numTraits):
            print("ratio" , i, (lookupDict1[i] * positiveProb) / ((1 - lookupDict1[i]) * positiveProb))
    """

def printResults():
    print("Class 0: tested ", numNegTestingData, " correctly classified ", negTestSuccesses)
    print("Class 1: tested ", numPosTestingData, " correctly classified ", posTestSuccesses)
    print("Overall: tested ", numNegTestingData + numPosTestingData, " correctly classified ", negTestSuccesses + posTestSuccesses)
    accuracy = (negTestSuccesses + posTestSuccesses) / (numNegTestingData + numPosTestingData)
    print("Accuracy: ", accuracy)

def main():
    trainNB()
    testNB()
    printResults()

if __name__ == '__main__':
    main()