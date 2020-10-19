import math
import numpy as np
import scipy

class DIST:
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((self.x1 - self.x2)**2))


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
        y_pred = [self.predict(x) for x in X]

        return np.array(y_pred)

    def euclidean_distance(self, point1, point2):
        """
        """

        return np.linalg.norm(np.array(point1) - np.array(point2))

    def _get_neighbors(self, x):
        """

        """
        # 1: Find the distance between each item in X_test and all items in the training set
        # For each index in X_test
        

        # sort by indices and capture only to chosen k

        # capture the labels 
    











# create train_test split
# create normalizing function
# create accuracy function
# create different distance choices


# https://www.youtube.com/watch?v=ngLyX54e1LU
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/knn.py
# https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_knn_algorithm_finding_nearest_neighbors.htm


