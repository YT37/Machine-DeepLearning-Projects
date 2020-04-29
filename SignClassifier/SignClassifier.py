import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import (
    AveragePooling2D,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
)
from tensorflow.keras.models import Sequential

with open("./Dataset/Train.p", mode="rb") as training:
    trainingSet = pickle.load(training)
with open("./Dataset/Validation.p", mode="rb") as validation:
    validationSet = pickle.load(validation)
with open("./Dataset/Test.p", mode="rb") as testing:
    testingSet = pickle.load(testing)

(Xtrain, yTrain), (Xtest, yTest), (Xvalid, yValid) = (
    (trainingSet["features"], trainingSet["labels"]),
    (testingSet["features"], testingSet["labels"]),
    (validationSet["features"], validationSet["labels"]),
)

Xtrain = ((np.sum(Xtrain / 3, axis=3, keepdims=True)) - 128) / 128
Xtest = ((np.sum(Xtest / 3, axis=3, keepdims=True)) - 128) / 128
Xvalid = ((np.sum(Xvalid / 3, axis=3, keepdims=True)) - 128) / 128

model = Sequential()

model.add(Convolution2D(32, (5, 5), input_shape=(32, 32, 1), activation="relu"))
model.add(AveragePooling2D())
model.add(Dropout(0.1))

model.add(Convolution2D(64, (5, 5), activation="relu"))
model.add(AveragePooling2D())
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(43, activation="softmax"))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
    Xtrain,
    yTrain,
    batch_size=500,
    epochs=50,
    shuffle=True,
    validation_data=(Xvalid, yValid),
)

model.evaluate(Xtest, yTest)

yPred = model.predict_classes(Xtest)
cm = confusion_matrix(yTest, yPred)
