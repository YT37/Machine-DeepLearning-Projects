import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.regressor_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

dataset = pd.read_csv("CarPurchasing.csv", encoding="ISO-8859-1")

scX = MinMaxScaler()
scY = MinMaxScaler()

X = scX.fit_transform(dataset.iloc[:, 3:-1].values)
y = scY.fit_transform(dataset.iloc[:, -1].values.reshape(-1, 1))

Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=0)

regressor = Sequential()
regressor.add(Dense(25, input_dim=5, activation="relu"))
regressor.add(Dropout(0.1))
regressor.add(Dense(25, activation="relu"))
regressor.add(Dropout(0.1))
regressor.add(Dense(1, activation="linear"))
regressor.compile(optimizer="adam", loss="mean_squared_error")
regressor.fit(Xtrain, yTrain, epochs=50, batch_size=25, validation_split=0.2)

yPred = regressor.predict(Xtest)
