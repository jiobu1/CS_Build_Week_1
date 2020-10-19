import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial import distance


def euclidean_distance(point1, point2, length):
    """
    """
    distance = 0
    for x in range(length):
        distance += (point1[x]-point2[x])**2
    return np.sqrt(distance)


def get_distances(X_test, X_train):
    """
    """
    length = X_test.shape[1]
    distances = []
    for idx in range(len(X_test)):

        # Initialize empty distance array
        # Loop through each row in x_train
        for row in range(len(X_train)):
            #find the euclidean distance and append to distance list
            dist = euclidean_distance(X_train.iloc[row], X_test.iloc[idx], length)
            distances.append((row, dist))

    return distances

def get_neighbors(distances, y_train, k):
    # sort distances
    y_indices = np.argsort(distances)[:k] #sort distances and record up to k values
    # find the classes that correspond with nearest neighbors
    k_nearest_classes = [y_train[i] for i in y_indices]
    # make a predication based on the mode of the classes
    y_pred = np.argmax(np.bincount(k_nearest_classes))
    #https://www.w3resource.com/python-exercises/numpy/python-numpy-random-exercise-13.php

    pass




X_train = [[0,3,0],[2,0,0]]
X_train = DataFrame(X_train)
X_test = [[0,0,0],[1,1,1], [2,2,2]]
X_test = DataFrame(X_test)

print(get_distances(X_test, X_train))