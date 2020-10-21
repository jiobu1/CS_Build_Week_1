import math
import numpy as np
from scipy import stats


class KNN:
    """
    K-Nearest Neighbors Algorithm implementation

    Implementation:
    Method:
    - euclidean distance(point1, point2, length):
        Returns euclidean distance between points
    - fit(X_train, y_train):
        Fits model to training data
    - predict(X_test):
        Returns predictions for X_test based on fitted model
    """

    def __init__(self, k):
        self.k = k

    def fit(self, X_train, y_train):
        """
        Fit the model using X as training data and y as target values
        Args:
            X_train ([array-like]): [description]
            y_train ([array-like]): [description]
        Passes along to _get_distance function
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predict the class labels for the provided data.
        Args:
            X_test: list or array
        Returns:
            y_pred: list of predictions for each vector in X_test
        """
        distances = self._get_distance(X_test)
        return self._get_labels(distances)

    def euclidean_distance(self, point1, point2, length):
        """
        Calculation: np.sqrt((point1 - point2)**2)

        Args:
            point1 : vector
            point2 : vector
            length (int): length of all vectors

        Returns:
            distance(float): euclidean distance between the two vectors
        """
        distance = 0
        for x in range(length):
            distance += (point1[x]-point2[x])**2
        return np.sqrt(distance)


    def _get_distance(self, X_test):
        """
        Helper Function:
        For each vector in X_test:
            - Calculates the distance between the each X_test vector and all the vectors in X_train
            - Add the distance and the index of the X_test to an ordered collection

        Args:
            X_test (array): list or array

        Returns:
            distances(list): list of each X_test value and euclidean distance to each vector in X_train
        """
        length = X_test.shape[1]
         # Initialize empty distance array
        distances = []
        for idx in range(len(X_test)):
            # record X_test id and initialize an empty array to hold distances
            distances.append([ X_test[idx], [] ])
            # Loop through each row in x_train
            for row in self.X_train:
                #find the euclidean distance and append to distance list
                dist = self.euclidean_distance(row, X_test[idx], length)
                distances[idx][1].append(dist)
        return distances

    def _get_labels(self, distances):
        """
        Helper Function:
        Gets the distances from the _get_distances helper function.
        - sorts the ordered collection of distances in ascending order
        - picks the first k entries
        - find the mode of the labels of the selected k
        - append to y_pred list

        Args:
            distances (list): list of each X_test value and euclidean distance to each vector in X_train

        Returns:
            y_pred(list): list of predictions for each vector in X_test based on the mode of k-neighbors
        """
        y_pred = []
        for row in range(len(distances)):
            # sort distances
            distance = distances[row]
            y_indices = np.argsort(distance[1])[:self.k] #sort distances and record up to k values
            #find the classes that correspond with nearest neighbors
            k_nearest_classes = [self.y_train[i%len(self.y_train)] for i in y_indices]
            # make a predication based on the mode of the classes
            label = [stats.mode(k_nearest_classes)][0][0][0]
            y_pred.append(label)
        return y_pred

