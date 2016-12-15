'''

Dinesh Singh

dinesh.singh@nyu.edu

kNN Supervised Learning Algorithm

The K-Nearest Neighbor algorithm creates its prediction by first searching through the
training dataset and determining the ’k’ most similar instances. The most similar instances are
evaluated and summarized, and then returned as the prediction for the data being classified. In
my implementation of K-Nearest Neighbor, I used the Euclidean distance as a heuristic. I also had
a dedicated function for file I/O, a function that finds the appropriate neighbors, a function that
makes a prediction based on those neighbors, and a function that determines the accuracy of the
prediction. My algorithm produces an accuracy of about 86.6% with k = 5 neighbors.

'''

import csv
import math
import operator


# Calculates the Euclidean distance
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(1, length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# Opens and loads the file into a dataset
def loadDataset(filename, targetSet=[]):
    with open(filename, 'r') as csvfile:
        next(csvfile)
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(10):
                dataset[x][y] = float(dataset[x][y])
            targetSet.append(dataset[x])


# Finds a set of 'k' neighbors
def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# Evaluates the neighbors
def getResponse(neighbors):
    votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][0]
        if response in votes:
            votes[response] += 1
        else:
            votes[response] = 1
    sortedVotes = sorted(iter(votes.items()), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


# Calculates the accuracy of the algorithm
def getAccuracy(testSet, predictions):
    numCorrect = 0
    for x in range(len(testSet)):
        if testSet[x][0] == predictions[x]:
            numCorrect += 1
    return (numCorrect / float(len(testSet))) * 100.0


def main():

    trainingSet = []
    testSet = []
    loadDataset('votes-train.csv', trainingSet)
    loadDataset('votes-test.csv', testSet)
    print( 'Train set: ' + repr(len(trainingSet)) )
    print( 'Test set: ' + repr(len(testSet)))

    predictions = []
    k = 5
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> Predicted = ' + repr(result) + ', Actual = ' + repr(testSet[x][0]))

    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


main()
