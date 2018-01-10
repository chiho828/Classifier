'''
Brian Luu, Siying Li, Chiho Kim
Option C: Implementing KNN and Naive Bayes Classification

Standard dataset: Fisher's Iris

Original dataset: Wisconsin Diagnostic Breast Cancer:
http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
'''

import KNN
import NaiveBayes
import csv
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


def createBags(trainSet, bagSize, bagNum):
    bbag = []
    for i in range(bagNum):
        sbag = []
        for j in range(bagSize):
            index = random.randint(0, len(trainSet) - 1)
            sbag.append(trainSet[index])
        bbag.append(sbag)
    return bbag


def findMajor(candidates):
    map = {}
    major = ('', 0)
    for i in candidates:
        if i in map:
            map[i] += 1
        else:
            map[i] = 1
        if map[i] > major[1]:
            major = (i, map[i])
    return major[0]


def bagging(k, trainSet, testSet, bagSize, bagNum):
    bags = createBags(trainSet, bagSize, bagNum)

    predictions = []
    for x in range(len(testSet)):
        candidates = []
        for y in range(len(bags)):
            bag = bags[y]
            neighbors = KNN.getNeighbors(bag, testSet[x], k)
            result = KNN.getResult(neighbors, True)
            candidates.append(result)
            candidates.append(NaiveBayes.run(bag=True, trainingSet=bag, test=testSet[x]))
        predictions.append(findMajor(candidates))
    accuracy = KNN.getAccuracy(testSet, predictions)
    print('')
    print('Accuracy: ' + str(accuracy) + '%')


def run():
    'Main loop, it gets and processes user input until "bye".'
    print('''Hi there! My name is Mr. Rabbits!                        (\_/)           
Welcome to Mr. Rabbits' Machine Learning Adventure!      (^.^)
Today we will be exploring the difference between       c(> <)
Naive Bayes classification and k-nearest neighbors.
There are two datasets to choose from: Fisher's Iris flower data set or ________.''')
    while True:
        invalid = False
        info = input('''Please let me know which classifier you would like to explore:
(type 'knn' or 'naive bayes' or 'bagging' or 'bye' to exit).\n''')
        if info == 'bye':
            print('Goodbye! Bring me a carrot next time! :3"')
            return
        print(
            "Which dataset will you be exploring today? Fisher's iris flower dataset or Wisconsin breast cancer diagnostics?")
        dataset = input("Type 'FI' or 'BC'\n")

        split = input("What % of the dataset should be split into the training set? (type a value from 0 to 100)\n")
        split = float(split) / 100

        filename = ''
        if dataset == "FI":
            filename = 'iris.csv'
        elif dataset == "BC":
            filename = 'wdbc_clean.csv'

        trainSet = []
        testSet = []

        createDataset(filename, trainSet, testSet, split)

        if info == 'knn':
            k = input("What value should k be? (# of nearest neighbors)\n")
            KNN.run(trainSet, testSet, int(k))
        elif info == 'naive bayes':
            NaiveBayes.run(trainingSet=trainSet, testSet=testSet)
        elif info == 'bagging':
            k = input("What value should k be? (# of nearest neighbors)\n")
            bagSize = input("How big should the bags be?\n")
            bagNum = input("How many bags should I use?\n")
            bagging(int(k), trainSet, testSet, int(bagSize), int(bagNum))
        else:
            invalid = True

        if invalid:
            print("Oops! There was some invalid input somewhere along the way.")
            print("Let's start from the top again.\n")
        else:
            print("Wow! That was fun. Let's do it again.\n")


run()
