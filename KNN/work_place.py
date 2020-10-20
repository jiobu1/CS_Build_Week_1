import numpy as np
import pandas as pd
from scipy import stats


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
     # Initialize empty distance array
    distances = []
    for idx in range(len(X_test)):

        # Loop through each row in x_train
        for row in range(len(X_train)):
            #find the euclidean distance and append to distance list
            dist = euclidean_distance(X_train.iloc[row], X_test.iloc[idx], length)
            distances.append(dist)
    return distances

def get_labels(distances, y_train, k):
    # sort distances
    y_indices = np.argsort(distances)[:k] #sort distances and record up to k values
    #find the classes that correspond with nearest neighbors
    k_nearest_classes = [y_train[i] for i in y_indices]
    # make a predication based on the mode of the classes
    y_pred =  [stats.mode(k_nearest_classes)][0][0][0]
    return y_pred

X_train = [[0,3,0],[2,0,0],[9,4,2],[1,7,4],[8,12,3]]
X_train = pd.DataFrame(X_train)
X_test = [[9,4,2]]
X_test = pd.DataFrame(X_test)
y_train = ['a','a','l', 'a','l']
y_train = np.array(y_train)

# print(y_train.columns.tolist())
distances = get_distances(X_test, X_train)
print("Distances: ", distances)
labels = get_labels(distances, y_train, 3)
print("Labels: ", labels)