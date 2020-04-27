import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

(Xtrain, yTrain), (Xtest, yTest) = cifar10.load_data()

Xtrain, Xtest = Xtrain.astype("float32") / 255, Xtest.astype("float32") / 255

yTrain, yTest = (
    keras.utils.to_categorical(yTrain, 10),
    keras.utils.to_categorical(yTest, 10),
)

model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape=(32, 32, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(Xtrain, yTrain, batch_size=32, epochs=10, shuffle=True)
