import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

dataset = pd.read_csv("Accounts.csv")

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

ct = ColumnTransformer([('Gender', OneHotEncoder(categories="auto"), [2])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

ct = ColumnTransformer([('Country', OneHotEncoder(categories="auto"), [3])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))
X = X.astype(float)
X = X[:, 1:]

Xtrain, Xtest, yTrain, yTest = train_test_split(X,
                                                y,
                                                test_size=0.1,
                                                random_state=0)

classifier = XGBClassifier(max_depth=10,
                           n_estimators=300,
                           gamma=0.5,
                           random_state=0)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)

cm = confusion_matrix(yTest, yPred)

accuracy = accuracy_score(yTest, yPred)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10**3) / 10**3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10**3) / 10**3
f1Score = int((2 * precision * recall / (precision + recall)) * 10**3) / 10**3
