'''
Brian Luu, Siying Li, Chiho Kim
Option C: Implementing KNN and Naive Bayes Classification

Standard dataset: Fisher's Iris

Original dataset: Wisconsin Diagnostic Breast Cancer:
http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
'''

import numpy as np
import math


def calculate_probability(x, mean, std):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent


def compute_prob(cate_prob, trainingSet, attr_cate_prob):
    for curr_cate, curr_cate_prob in cate_prob.items():
        all_cate_sample = trainingSet[trainingSet[:, -1] == curr_cate]
        for i in range(0, len(trainingSet[0]) - 1):
            attr = trainingSet[0][i]
            curr_attr_cate = all_cate_sample[:, i].astype(float)
            std = np.std(curr_attr_cate)
            mean = np.mean(curr_attr_cate)
            attr_cate_prob[(i, curr_cate)] = (mean, std)


def predict(curr_flow, cate_prob, attr_cate_prob):
    max_cate_PDF = -1
    max_cate = -1
    for curr_cate, curr_cate_prob in cate_prob.items():
        total_prob = 1
        for i in range(0, len(curr_flow) - 1):
            curr_param = attr_cate_prob[(i, curr_cate)]
            curr_prob = calculate_probability(float(curr_flow[i]), curr_param[0], curr_param[1])
            total_prob *= curr_prob
        if (total_prob > max_cate_PDF):
            max_cate_PDF = total_prob
            max_cate = curr_cate
    return max_cate_PDF, max_cate


def run(bag=False, trainingSet=[], testSet=[], test=[]):
    trainingSet = np.asarray(trainingSet)
    testSet = np.asarray(testSet)

    cate, cate_count = np.unique(trainingSet[:, -1], return_counts=True)
    cate_prob = dict(zip(cate, cate_count / len(trainingSet)))

    attr_cate_prob = {}
    compute_prob(cate_prob, trainingSet, attr_cate_prob)

    correct_count = 0
    count = 1

    if not bag:
        for curr_test in testSet:
            predict_cate_PDF, predict_cate = predict(curr_test, cate_prob, attr_cate_prob)
            print("Test #" + str(count) + ", sample: " + str(list(curr_test)))
            print("The highest probability density function value was: " + str(predict_cate_PDF))
            correct_count += (predict_cate == curr_test[-1])
            count += 1

        print('Training set length is: ' + str(len(trainingSet)))
        print('Test set length is: ' + str(len(testSet)))
        print('Prediction accuracy: ' + str(correct_count / len(testSet) * 100) + '%')
    else:
        predict_cate_PDF, predict_cate = predict(test, cate_prob, attr_cate_prob)

    return predict_cate

# run(0.7, 'iris.data')
