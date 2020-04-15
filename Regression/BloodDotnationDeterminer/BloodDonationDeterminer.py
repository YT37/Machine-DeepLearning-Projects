import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(
    r"D:\Codes\MachineLearning\Projects\Regression\BloodDotnationDeterminer\Blood.csv"
)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

Xtrain, Xtest, yTrain, yTest = train_test_split(X,
                                                y,
                                                test_size=0.2,
                                                random_state=0)

regressor = RandomForestRegressor(n_estimators=150, random_state=0)
regressor.fit(Xtrain, yTrain)

yPred = regressor.predict(Xtest)

yPredcm = np.zeros_like(yPred)
yPredcm[yPred > 0.6] = 1
yPredcm[yPred < 0.6] = 0

cm = confusion_matrix(yTest, yPredcm)

accuracy = accuracy_score(yTest, yPredcm)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10**3) / 10**3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10**3) / 10**3
f1Score = int((2 * precision * recall / (precision + recall)) * 10**3) / 10**3
