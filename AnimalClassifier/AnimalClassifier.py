from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

trainingGen = ImageDataGenerator(
    rescale=1.0 / 255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
).flow_from_directory(
    "Dataset/Training", target_size=(128, 128), batch_size=128, class_mode="binary"
)

validationGen = ImageDataGenerator(
    rescale=1.0 / 255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
).flow_from_directory(
    "Dataset/Validation", target_size=(128, 128), batch_size=128, class_mode="binary"
)

base = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
base.trainable = False

model = Model(
    inputs=base.input,
    outputs=Dense(units=1, activation="sigmoid")(GlobalAveragePooling2D()(base.output)),
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(trainingGen, epochs=2, validation_data=validationGen)

loss, accuracy = model.evaluate(validationGen)

base.trainable = True

for layer in base.layers[:100]:
    layer.trainable = False

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(trainingGen, epochs=2, validation_data=validationGen)

loss, accuracy = model.evaluate(validationGen)
