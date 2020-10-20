# Imports
# Python class
from knn import KNN

# sklearn model
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Preprocessing/ Visualization
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans # determine optimal k
from sklearn.metrics import accuracy_score

# Load Data
iris = load_iris()

# Separate into target from features
#Scale features
X = scale(iris.data)

y = iris.target # classes

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42) # for reproducible results

# Sklearn - KNN Classifier

#1.Choose K based on results from elbow method
clf = KNeighborsClassifier(n_neighbors=3)

# Fit
clf.fit(X_train, y_train)

# Prediction
predict = clf.predict(X_test)
print("******SKLEARN******")
print("Prediction", predict)

# Accuracy Score
print(f"Scikit-learn KNN classifier accuracy: {accuracy_score(y_test, predict)}")

# y_pred
y_pred = clf.predict([X_test[0]])
print("y_pred", y_pred)

##########################################################################

# KNN (build model)
#1. Instantiate model
np_clf = KNN(k=3)

#2. Fit
np_clf.fit(X_train, y_train)

#3. Prediction
predict = np_clf.predict(X_test)
print("******PYTHON MODEL******")
print("Prediction", predict)

#4. Accuracy Score
print(f"KNN model accuracy: {accuracy_score(y_test, predict)}")

# y_pred
y_pred = np_clf.predict([X_test[0]])
print("y_pred", y_pred)