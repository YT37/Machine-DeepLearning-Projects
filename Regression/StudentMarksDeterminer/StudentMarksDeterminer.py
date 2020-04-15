import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score

por = r"D:\Codes\MachineLearning\Projects\Regression\StudentMarksDeterminer\StudentMath.csv"
math = r"D:\Codes\MachineLearning\Projects\Regression\StudentMarksDeterminer\StudentPor.csv"

porSize = 649
mathSize = 395

dataset = pd.read_csv(math, delimiter=";")

dataset = dataset.drop(dataset.loc[:, "address":'Walc'].columns, axis=1)

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])],
                       remainder="passthrough")
X = np.array(ct.fit_transform(X))

X = X[:, 1:]

Xtrain, Xtest, yTrain, yTest = train_test_split(X,
                                                y,
                                                test_size=0.3015,
                                                random_state=0)

regressor = LinearRegression()
regressor.fit(Xtrain, yTrain)

yPred = regressor.predict(Xtest)

yPredcm = np.zeros_like(yPred)
yPredcm[yPred > 0.6] = 1
yPredcm[yPred < 0.6] = 0

yTestcm = np.zeros_like(yTest)
yTestcm[yTest > 0.6] = 1
yTestcm[yTest < 0.6] = 0

cm = confusion_matrix(yTestcm, yPredcm)

accuracy = accuracy_score(yTestcm, yPredcm)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10**3) / 10**3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10**3) / 10**3
f1Score = int((2 * precision * recall / (precision + recall)) * 10**3) / 10**3
