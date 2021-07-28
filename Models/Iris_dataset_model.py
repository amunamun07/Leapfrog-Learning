# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# read the dataset
def read_the_dataset():
    data = pd.read_csv('./datasets/iris.csv')
    variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    data = data.replace(['Setosa', 'Versicolor', 'Virginica'], [0, 1, 2])
    return data, variety_mappings


def split(data):
    X = data.iloc[:, 0:-1]
    Y = data.iloc[:, -1]
    return X, Y


def LR(X, Y):
    logreg = LogisticRegression(max_iter=1000)  # Initializing the Logistic Regression model
    logreg.fit(X, Y)
    return logreg


def classify(sepal_len, sepal_wid, petal_len, petal_wid):
    data, variety_mappings = read_the_dataset()
    X, Y = split(data)
    logreg = LR(X, Y)
    arr = np.array([sepal_len, sepal_wid, petal_len, petal_wid])
    arr = arr.astype(np.float64)
    query = arr.reshape(1, -1)
    prediction = variety_mappings[logreg.predict(query)[0]]
    return prediction


