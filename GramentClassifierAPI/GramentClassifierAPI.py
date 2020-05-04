import numpy as np
from flask import Flask, jsonify
from imageio import imread
from tensorflow.keras.models import model_from_json

with open("Model.json", "r") as f:
    json = f.read()

model = model_from_json(json)
model.load_weights("Model.h5")

app = Flask(__name__)


@app.route("/api/<string:name>", methods=["POST"])
def classify(name):
    upload = "uploads/"
    image = imread(upload + name)

    classes = [
        "T-shirt",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Boot",
    ]

    prediction = model.predict([image.reshape(1, 28 * 28)])

    return jsonify({"Object ": classes[np.argmax(prediction[0])]})


app.run(port=5000, debug=False)
