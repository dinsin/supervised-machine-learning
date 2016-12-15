'''

Dinesh Singh

dinesh.singh@nyu.edu

ID3 Decision Tree Supervised Learning Algorithm

The ID3 Decision Tree algorithm works by constructing a tree in a top-down, recursive,
divide-and-conquer method. Initially, the training samples are at the root of the tree. Samples are
then passed through recursively, and partitioned based on a heuristic or predetermined attribute.
The ID3 algorithm stops partitioning when all samples belong tot he same class, there are no
remaining samples, or there are no attributes left to interpret. My implementation of the ID3 algorithm has functions for file I/O, determining accuracy, building the tree, making decisions using
the tree, and making a given node a parent or terminal node. I also created a function that uses
a proportion to calculate the optimal location at which to split datasets. The average accuracy of
my ID3 algorithm is 79%.

'''

import csv


# Opens and loads a dataset into a list
def loadDataset(filename):
    dataSet = list()
    with open(filename, 'r') as csvfile:
        next(csvfile)
        lines = csv.reader(csvfile)
        lineSet = list(lines)
        for x in range(len(lineSet)):
            for y in range(len(lineSet[x])):
                lineSet[x][y] = float(lineSet[x][y])
            dataSet.append(lineSet[x])
    return dataSet


# Calculates the accuracy of the algorithm
def calcAccuracy(actual, predicted):
    numCorrect = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            numCorrect += 1
    return numCorrect / float(len(actual)) * 100.0


# Evaluate an algorithm
def evaluateAlgorithm(testSet, trainSet, algorithm, maxDepth, minSize):
    scores = list()
    predicted = algorithm(testSet, trainSet, maxDepth, minSize)
    actual = [row[0] for row in testSet]
    accuracy = calcAccuracy(actual, predicted)
    scores.append(accuracy)
    return scores


# Splits a dataset based on a chosen attribute and value
def testSplit(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Calculate the index for a split dataset
def calcIndex(groups, class_values):
    index = 0.0
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            proportion = [row[0] for row in group].count(class_value) / float(size)
            index += (proportion * (1.0 - proportion))
    return index


# Select the best split point for a dataset
def getSplit(dataset):
    class_values = list(set(row[0] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(1, len(dataset[0])):
        for row in dataset:
            groups = testSplit(index, row[index], dataset)
            optIndex = calcIndex(groups, class_values)
            if optIndex < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], optIndex, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# Creates a terminal node
def makeTerminal(group):
    outcomes = [row[0] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Makes a node a parent or make it terminal
def split(node, maxDepth, minSize, depth):
    left, right = node['groups']
    del (node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = makeTerminal(left + right)
        return
    # check for max depth
    if depth >= maxDepth:
        node['left'], node['right'] = makeTerminal(left), makeTerminal(right)
        return
    # process left child
    if len(left) <= minSize:
        node['left'] = makeTerminal(left)
    else:
        node['left'] = getSplit(left)
        split(node['left'], maxDepth, minSize, depth + 1)
    # process right child
    if len(right) <= minSize:
        node['right'] = makeTerminal(right)
    else:
        node['right'] = getSplit(right)
        split(node['right'], maxDepth, minSize, depth + 1)


# Constructs a decision tree
def constructTree(testSet, maxDepth, minSize):
    root = getSplit(testSet)
    split(root, maxDepth, minSize, 1)
    return root


# Makes a prediction using the decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# Defines the core tree algorithm
def decisionTree(test, train, maxDepth, minSize):
    tree = constructTree(test, maxDepth, minSize)
    predictions = list()
    for row in train:
        prediction = predict(tree, row)
        print("> Predicted = %d, Actual = %d" % (prediction, row[0]))
        predictions.append(prediction)
    return (predictions)



def main():

    testSet = loadDataset("votes-test.csv")
    trainSet = loadDataset("votes-train.csv")

    maxDepth = 5
    minSize = 10
    scores = evaluateAlgorithm(testSet, trainSet, decisionTree, maxDepth, minSize)
    print('Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

main()