import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

dataset = pd.read_csv("Crimes.csv")
dataset.Date = pd.to_datetime(dataset["Date"])
dataset.index = pd.DatetimeIndex(dataset["Date"])
dataset = dataset.resample("D").size()

trainingSet = dataset.iloc[:1094].values.reshape(-1, 1)
testingSet = dataset.iloc[1035:].values.reshape(-1, 1)

sc = MinMaxScaler()
trainingSetSc = sc.fit_transform(trainingSet)
testingSetSc = sc.transform(testingSet)

Xtrain = []
yTrain = []

for i in range(60, 1094):
    Xtrain.append(trainingSetSc[i - 60:i, 0])
    yTrain.append(trainingSetSc[i, 0])

Xtrain, yTrain = np.reshape(
    np.array(Xtrain),
    [np.array(Xtrain).shape[0],
     np.array(Xtrain).shape[1], 1]), np.array(yTrain)

model = Sequential()

model.add(LSTM(50, return_sequences=True,
                   input_shape=(Xtrain.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50))

model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(Xtrain, yTrain, batch_size=32, epochs=100)

Xtest = []

for i in range(60, 92):
    Xtest.append(testingSetSc[i - 60:i, 0])

Xtest = np.reshape(np.array(Xtest),
                   [np.array(Xtest).shape[0],
                    np.array(Xtest).shape[1], 1])

predicted = np.round(sc.inverse_transform(model.predict(Xtest)))

plt.plot(dataset.iloc[1094:].values, color="red", label="Real Crime Rate")
plt.plot(predicted, color="green", label="Predicted Crime Rate")
plt.title("Crime Rate Prediction")
plt.xlabel("Time")
plt.ylabel("Crime Rate")
plt.legend()
plt.show()
