import numpy as np
import pandas as pd
from minisom import MiniSom
from pylab import bone, colorbar, pcolor, plot, show
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

dataset = pd.read_csv("CreditCard.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

sc = MinMaxScaler()
X = sc.fit_transform(X)

som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, 100)

bone()
pcolor(som.distance_map().T)
colorbar()
markers = ["o", "s"]
colors = ["r", "g"]

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], color=colors[y[i]])

show()

mappings = som.win_map(X)
frauds = sc.inverse_transform(
    np.concatenate((mappings[(2, 0)], mappings[(9, 4)]), axis=0))

customers = dataset.iloc[:, 1:].values
isFraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        isFraud[i] = 1

sc = StandardScaler()
customers = sc.fit_transform(customers)

classifier = Sequential()
classifier.add(
    Dense(2, input_dim=15, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer="adam",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])
classifier.fit(customers, isFraud, batch_size=1, epochs=2)

yPred = np.concatenate(
    (dataset.iloc[:, 0:1].values, classifier.predict(customers)),
    axis=1)[yPred[:, 1].argsort()]
