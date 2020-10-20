import math
import numpy as np
import scipy


class DIST:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2


class KNN:
    """

    """
    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        """
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        """
        distances = self._get_distance(X_test)
        return self._get_labels(distances)

    def euclidean_distance(self, point1, point2, length):
        """
        """
        distance = 0
        for x in range(length):
            distance += (point1[x]-point2[x])**2
        return np.sqrt(distance)


    def _get_distance(self, X_test):
        """
        """
        length = X_test.shape[1]
         # Initialize empty distance array
        distances = []
        for idx in range(len(X_test)):

            # Loop through each row in x_train
            for row in self.X_train:
                #find the euclidean distance and append to distance list
                dist = self.euclidean_distance(row, X_test[idx], length)
                distances.append(dist)

        return distances

    def _get_labels(self, distances):
        """
        """
        # sort distances and pick up to nearest neighbor
        y_indices = np.argsort(distances)[:self.k]
        # pick the classes based on the nearest neighbors
        k_nearest_classes = [self.y_train[i] for i in y_indices]
        # get the most common class and assign
        y_pred = [scipy.stats.mode(k_nearest_classes)][0][0][0]

        return y_pred


# TO DO:
# create virtual environment -> pandas, numpy, scipy, sklearn
# create train_test split
# create normalizing function -> scale function
# create accuracy function
# create different distance choices
# create doc strings


# https://www.youtube.com/watch?v=ngLyX54e1LU
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/knn.py
# https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_knn_algorithm_finding_nearest_neighbors.htm


