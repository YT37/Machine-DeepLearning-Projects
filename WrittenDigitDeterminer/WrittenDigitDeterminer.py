import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import normalize

(Xtrain, yTrain), (Xtest, yTest) = mnist.load_data()

Xtrain = normalize(Xtrain, axis=1)
Xtest = normalize(Xtest, axis=1)

classifier = Sequential()
classifier.add(Flatten())
classifier.add(Dense(128, activation=tf.nn.relu))
classifier.add(Dense(128, activation=tf.nn.relu))
classifier.add(Dense(10, activation=tf.nn.softmax))
classifier.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

classifier.fit(
    Xtrain,
    yTrain,
    epochs=100,
)

valLoss, valAccu = classifier.evaluate(Xtest, yTest)

yPred = np.argmax(classifier.predict([Xtest])[0])
