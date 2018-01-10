'''
Brian Luu, Siying Li, Chiho Kim
Option C: Implementing KNN and Naive Bayes Classification

Standard dataset: Fisher's Iris

Original dataset: Wisconsin Diagnostic Breast Cancer:
http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
'''

import csv
import operator
import random


def createDataset(filename, trainSet, testSet, trainSize):
    csvfile = open(filename)
    lines = csv.reader(csvfile)
    dataset = list(lines)
    testNum = len(dataset) - 1 - round((len(dataset) - 1) * trainSize)
    attributes = len(dataset[0]) - 1

    # create testSet
    for i in range(testNum):
        index = random.randint(0, len(dataset) - 2)  # ignore the last line
        chosen = dataset.pop(index)
        for j in range(attributes):
            chosen[j] = float(chosen[j])
        testSet.append(chosen)

    # create trainSet
    for x in range(len(dataset) - 1):  # ignore the last line
        for y in range(attributes):
            dataset[x][y] = float(dataset[x][y])
        trainSet.append(dataset[x])


def manhattanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += abs(instance1[x] - instance2[x])
    return distance


def getNeighbors(trainSet, test, k):
    distances = []
    length = len(test) - 1
    for x in range(len(trainSet)):
        dist = manhattanDistance(test, trainSet[x], length)
        distances.append((trainSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResult(neighbors, bag=False):
    if not bag:
        print("Out of the " + str(len(neighbors)) + " nearest neighbors...")
    frequency = {}
    for x in range(len(neighbors)):
        species = neighbors[x][-1]
        if species in frequency:
            frequency[species] += 1
        else:
            frequency[species] = 1
    for key, value in frequency.items():
        if value == len(neighbors):
            if not bag:
                print("All of the neighbors are " + str(key) + ".")
        else:
            if not bag:
                print(str(value) + " of the neighbors are " + str(key) + ".")
    results = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    return results[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def run(trainSet, testSet, k, bag=False):
    predictions = []
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainSet, testSet[x], k)
        if not bag:
            print("\nTest #" + str(x + 1) + ". sample: " + str(testSet[x]) + ":")
        result = getResult(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testSet, predictions)
    if not bag:
        print('\nPrediction accuracy: ' + str(accuracy) + '%')

# run(0.9, 3, 'iris.data')
