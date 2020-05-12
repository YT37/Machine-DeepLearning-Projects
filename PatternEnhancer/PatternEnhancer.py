import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm


class DeepDream(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def tracing(self, img, steps, stepSize):
        loss = tf.constant(0.0)

        for n in tf.range(steps):
            with tf.GradientTape() as tape:
                tape.watch(img)

                activ = self.model(tf.expand_dims(img, axis=0))

                if len(activ) == 1:
                    activ = [activ]

                losses = []
                for act in activ:
                    loss = tf.math.reduce_mean(act)
                    losses.append(loss)

                loss = tf.reduce_sum(losses)

            gradients = tape.gradient(loss, img)

            gradients /= tf.math.reduce_std(gradients) + 1e-8

            img = tf.clip_by_value((img + gradients * stepSize), -1, 1)

        return loss, img


img = Image.open("Image.jpg")
img.thumbnail((500, 500))
origImg = np.array(img)

base = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")
names = ["mixed3", "mixed5"]
layers = [base.get_layer(name).output for name in names]

deepdream = DeepDream(tf.keras.Model(inputs=base.input, outputs=layers))

steps = 100
stepSize = 0.01

baseShape = tf.shape(tf.constant(np.array(origImg)))[:-1]
floatShape = tf.cast(baseShape, tf.float32)

for n in tqdm(range(-2, 3), unit=""):
    img = tf.convert_to_tensor(
        tf.keras.applications.inception_v3.preprocess_input(
            tf.image.resize(
                origImg, tf.cast(floatShape * (1.30 ** n), tf.int32)
            ).numpy()
        )
    )
    stepSize = tf.convert_to_tensor(stepSize)

    remaining = steps
    step = 0

    while remaining:
        if remaining > 100:
            run = tf.constant(100)

        else:
            run = tf.constant(remaining)

        remaining -= run
        step += run

        loss, img = deepdream.tracing(img, run, tf.constant(stepSize))

        print(f"  Step {step} Loss {loss}")


img = tf.image.convert_image_dtype(
    tf.image.resize(tf.cast((255 * (img + 1.0) / 2.0), tf.uint8), baseShape) / 255.0,
    dtype=tf.uint8,
)
Image.fromarray(np.array(img)).save("Result.jpg")
