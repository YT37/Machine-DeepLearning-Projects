import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

dataset = pd.read_csv("Accounts.csv")

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder = LabelEncoder()
X[:, 2] = labelencoder.fit_transform(X[:, 2])
ct = ColumnTransformer([("Country", OneHotEncoder(categories="auto"), [1])],
                       remainder="passthrough")
X = np.array(ct.fit_transform(X))
X = X[:, 1:]

Xtrain, Xtest, yTrain, yTest = train_test_split(X,
                                                y,
                                                test_size=0.2,
                                                random_state=0)

scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)

classifier = Sequential()
classifier.add(
    Dense(6, input_dim=11, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(6, kernel_initializer="uniform", activation="relu"))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))
classifier.compile(optimizer="rmsprop",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
classifier.fit(Xtrain, yTrain, batch_size=25, epochs=500)

yPred = classifier.predict(Xtest)
yPred = (yPred > 0.5)

cm = confusion_matrix(yTest, yPred)

accuracy = accuracy_score(yTest, yPred)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10**3) / 10**3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10**3) / 10**3
f1Score = int((2 * precision * recall / (precision + recall)) * 10**3) / 10**3

# Single Prediction
"""newPred = scX.transform(np.array([[1, 0, 0, 600, 40, 3, 60000, 2, 1, 1, 50000]]))
yPred = classifier.predict(newPred)
yPred = (yPred > 0.5)"""
