import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

dataset = pd.read_csv(r"D:\Codes\MachineLearning\Projects\Classification\CarSafetyDeterminer\CarSafety.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer([("encoder", OneHotEncoder(), [0, 1, 2, 3, 4])],
                       remainder="passthrough")
X = np.array(ct.fit_transform(X))

ct = ColumnTransformer([("encoder", OneHotEncoder(), [18])],
                       remainder="passthrough")
X = np.array(ct.fit_transform(X))

X = X[:, 1:]

Xtrain, Xtest, yTrain, yTest = train_test_split(X,
                                                y,
                                                test_size=0.303,
                                                random_state=0)

classifier = KNeighborsClassifier(n_neighbors=10, metric="minkowski", p=2)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)

cm = confusion_matrix(yTest, yPred)

accuracy = accuracy_score(yTest, yPred)
