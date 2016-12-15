'''

Dinesh Singh

dinesh.singh@nyu.edu

Perceptron Supervised Learning Algorithm

The Perceptron algorithm is a model of a single neuron. It operates by receiving inputs
from the training data that are then weighted and used in the activation equation. The output
from this equation is put into a transfer function, which generates Perceptron’s prediction. In
my implementation of Peceptron, I included functions for file I/O, for determining accuracy, for
training the weights, and for evaluating the algorithm. My implementation uses a learning rate of
0.01, and runs for 1000 epochs. The average accuracy of my Perceptron algorithm is 86.0%.

'''

import csv
from csv import reader


# Opens and loads a data into a set
def loadDataset(filename):
    dataset = list()
    with open(filename, 'r') as file:
        next(file)
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Converts a column of string to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Converts a column of string to int
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Calculates accuracy of the algorithm
def calcAccuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluates Perceptron using a cross validation split
def evaluate(trainingDataset, testDataset, perceptron, *args):
    predicted = perceptron(trainingDataset, testDataset, *args)
    actual = [row[0] for row in testDataset]
    accuracy = calcAccuracy(actual, predicted)
    return accuracy


# Uses weights to make a prediction
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


# Estimates the weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            prediction = predict(row, weights)
            error = row[0] - prediction
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    return weights


# Definition of the main Perceptron algorithm
def perceptron(train, test, l_rate, n_epoch):
    predictions = list()
    weights = train_weights(train, l_rate, n_epoch)
    for row in test:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return (predictions)


def main():
    # load and prepare data
    trainingDataset = loadDataset("votes-train.csv")
    for i in range(1, 10):
        str_column_to_float(trainingDataset, i)

    testDataset = loadDataset("votes-test.csv")
    for i in range(1, 10):
        str_column_to_float(testDataset, i)

    # convert string class to integers
    str_column_to_int(trainingDataset, 0)
    str_column_to_int(testDataset, 0)

    # evaluate algorithm
    l_rate = 0.01
    n_epoch = 1000
    accuracy = evaluate(trainingDataset, testDataset, perceptron, l_rate, n_epoch)
    print('Accuracy: ' + str(float(accuracy)) + '%')


main()

