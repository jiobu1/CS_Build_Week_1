## CS_Build_Week_1

### What is KNN Algorithm?
K Nearest Neighbor (KNN) is one of the simplest algorithms used in Machine Learning for regression and classification.
KNN uses all the available data and classifies the new data or case based on a similarity measure (distance function).
The new data is then assigned to the class which has the most nearest neighbors.

### How is KNN used?
The K in KNN stands for the number of nearest neighbors your are using to figure out how to assign a class for your new data.
KNN is used in situations where you want to find similar items. 
- KNN is used in recommendation systems such as finding products similar to what you are searching for or to make movie recommendations.
- Searching semantically similar documents, such as Google search.
- OCR, Image or Video Recognition

## Installation
from knn import KNN

## Usage
* Initialize algorithm by passing through k(nearest neighbors)
   clf = KNN(k=3)
  
 
 * Fit the model
    clf.fit(X_train, y_train)
    
  * Make prediction on X_test
     predict = clf.predict(X_test)

### Read More
You can read more about KNN here https://jishaobukwelu.medium.com/whats-the-knn-74e84458bd24



