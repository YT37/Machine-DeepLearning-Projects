import numpy as np
from tensorflow.keras.layers import Convolution2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(
    Convolution2D(64, (3, 3), input_shape=(128, 128, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(128, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation="relu"))
classifier.add(Dense(128, activation="relu"))
classifier.add(Dense(1, activation="sigmoid"))
classifier.compile(optimizer="adam",
                   loss="binary_crossentropy",
                   metrics=["accuracy"])

trainData = ImageDataGenerator(rescale=1. / 255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True)

testData = ImageDataGenerator(rescale=1. / 255)

trainSet = trainData.flow_from_directory(directory="train",
                                         target_size=(128, 128),
                                         batch_size=32,
                                         class_mode='binary')

testSet = testData.flow_from_directory(directory="test",
                                       target_size=(128, 128),
                                       batch_size=32,
                                       class_mode='binary')

classifier.fit(
    trainSet,
    epochs=10,
    validation_data=testSet,
)

# Single Prediction
"""testImg = np.expand_dims(image.img_to_array(
    image.load_img(
        "",
        target_size=(128, 128))),
                         axis=0)

result = classifier.predict(testImg)"""
