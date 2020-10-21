import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


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
        distances.append([ X_test[idx], [] ])
        # Loop through each row in x_train
        for row in X_train:
            #find the euclidean distance and append to distance list
            dist = euclidean_distance(row, X_test[idx], length)
            distances[idx][1].append(dist)
    return distances

def get_labels(distances, y_train, k):
    labels = []
    for row in range(len(distances)):
        # sort distances
        distance = distances[row]
        y_indices = np.argsort(distance[1])[:k] #sort distances and record up to k values
        #find the classes that correspond with nearest neighbors
        k_nearest_classes = [y_train[i%len(y_train)] for i in y_indices]
        # make a predication based on the mode of the classes
        y_pred = [stats.mode(k_nearest_classes)][0][0][0]
        labels.append(y_pred)
    return labels

# X_train = np.array([[0,3,0],[2,0,0],[9,4,2],[1,7,4],[8,12,3]])
# # X_train = pd.DataFrame(X_train)
# X_test = np.array([[9,4,2], [0,3,0]])
# # X_test = pd.DataFrame(X_test)
# y_train = ['a','a','l', 'a','l']
# y_train = np.array(y_train)



# Load Data
iris = load_iris()

# Separate into target from features
#Scale features
X = scale(iris.data)
y = iris.target # classes

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) # for reproducible results

distances = get_distances(X_test, X_train)
print("Distances: ", distances)
labels = get_labels(distances, y_train, 3)
print("Labels: ", labels)