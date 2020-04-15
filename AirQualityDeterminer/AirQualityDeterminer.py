import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(
    r"D:\Codes\MachineLearning\Projects\Regression\AirQualityDeterminer\AirQuality.csv"
)

X = dataset.iloc[:, [6, 7, 8, 10, 11, 12]].values
y = dataset.iloc[:, 5].values

Xtrain, Xtest, yTrain, yTest = train_test_split(X,
                                                y,
                                                test_size=0.2,
                                                random_state=0)


regressor = RandomForestRegressor(n_estimators=250, random_state=0)
regressor.fit(Xtrain, yTrain)

yPred = regressor.predict(Xtest)

