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



X_train = [[0,3,0],[2,0,0]]
X_train = DataFrame(X_train)
X_test = [[0,0,0],[1,1,1]]
X_test = DataFrame(X_test)

print(get_distances(X_test, X_train))