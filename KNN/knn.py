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
        distances = []
        for idx in range(len(X_test)):

            # Initialize empty distance array
            # Loop through each row in x_train
            for row in range(len(self.X_train)):
                #find the euclidean distance and append to distance list
                dist = self.euclidean_distance(self.X_train.iloc[row], X_test.iloc[idx], length)
                distances.append((row, dist))

        return distances

    def _get_labels(self, distances):
        y_indices = np.argsort(distances)[:self.k]
        k_nearest_classes = [self.y_train[i] for i in y_indices]
        y_pred = np.argmax(np.bincount(k_nearest_classes))

        return y_pred


# create train_test split
# create normalizing function
# create accuracy function
# create different distance choices


# https://www.youtube.com/watch?v=ngLyX54e1LU
# https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/knn.py
# https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_knn_algorithm_finding_nearest_neighbors.htm


