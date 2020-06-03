# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 03:22:38 2020

@author: Minh Duc
"""
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def tri_function(x,m,a):
    
    tri=[0]*len(x)
    l=m-a
    u=m+a
    for j in range(len(x)):
        x_=x[j]

        if x_<m and x_>l:
            tri[j]=(x_-l)/a
        if x_>m and x_<u:
            tri[j]=-(x_-u)/a
    return tri

def accuracy(predicted_targets, real_targets):
    """
    :param predicted_targets: Numpy array with predicted targets.
    :param real_targets: Numpy array with real targets.
    :return: Float value between 0 - 1, representing accuracy (ex. 0.5 = half of the targets where correct)
    """
    predicted_targets = predicted_targets.tolist()
    real_targets = real_targets.tolist()
    correct_sum = 0

    for i in range(len(predicted_targets)):
        if predicted_targets[i] == real_targets[i]:
            correct_sum += 1
    return correct_sum/len(predicted_targets)

def csv_to_data_set(file_name):

    data_set = []
    targets = []
    with open(file_name) as csv_file:
        data_file = csv.reader(csv_file)
        next(data_file)
        # temp = next(data_file)
        for row in data_file:
            data_set.append(row[1:-1])
            targets.append(row[-1])

    return (np.array(data_set).astype(float), np.array(targets))

def split_data_set(data_set, targets, test_size):
    return train_test_split(data_set, targets, test_size=test_size)

def remove_doubles(in_list):
    return list(set(in_list))


def show_confusion_matrix(test_targets, predicted_targets, labels):
    
    print("-"*30, "Confusion matrix", "-"*30)
    #print("Confusion matrix")

    print(("{: >20}"*(len(labels)+1)).format("", *labels))
    
    for key, value in enumerate(confusion_matrix(test_targets, predicted_targets, labels)):
        
        print(("{: >20}"*(len(value)+1)).format(labels[key], *value))