import math
import random
import re
import bert
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from bs4 import BeautifulSoup
from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalMaxPool1D
from tqdm import tqdm

cols = ["sentiment", "id", "date", "query", "user", "text"]
data = pd.read_csv(
    "Tweets.csv", header=None, names=cols, engine="python", encoding="latin1",
)

data.drop(["id", "date", "query", "user"], axis=1, inplace=True)


def cleanTweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    return re.sub(
        r" +",
        " ",
        re.sub(
            r"[^a-zA-Z.!?']",
            " ",
            re.sub(
                r"https?://[A-Za-z0-9./]+", " ", re.sub(r"@[A-Za-z0-9]+", " ", tweet)
            ),
        ),
    )


def encode(sent):
    return ["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"]


def getIDs(tokens):
    return tokenizer.convert_tokens_to_ids(tokens)


def getMask(tokens):
    return np.char.not_equal(tokens, "[PAD]").astype(int)


def getSegments(tokens):
    segIDs = []
    currentID = 0
    for tok in tokens:
        segIDs.append(currentID)
        if tok == "[SEP]":
            currentID = 1 - currentID
    return segIDs


cleanData = [cleanTweet(tweet) for tweet in tqdm(data.text)]

labels = data.sentiment.values
labels[labels == 4] = 1

tokenizer = bert.bert_tokenization.FullTokenizer
bertLayer = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False
)
tokenizer = tokenizer(
    bertLayer.resolved_object.vocab_file.asset_path.numpy(),
    bertLayer.resolved_object.do_lower_case.numpy(),
)

dataInputs = [encode(sentence) for sentence in tqdm(cleanData)]

lenData = [[sent, labels[i], len(sent)] for i, sent in enumerate(dataInputs)]
random.shuffle(lenData)
lenData.sort(key=lambda x: x[2])

sortedData = [
    ([getIDs(sent[0]), getMask(sent[0]), getSegments(sent[0])], sent[1])
    for sent in lenData
    if sent[2] > 7
]

dataset = tf.data.Dataset.from_generator(
    lambda: sortedData, output_types=(tf.int32, tf.int32)
)

batchSize = 32
batched = dataset.padded_batch(
    batchSize, padded_shapes=((3, None), ()), padding_values=(0, 0)
)

batches = math.ceil(len(sortedData) / batchSize)
testBatch = batches // 10
batched.shuffle(batches)
testDataset = batched.take(testBatch)
trainDataset = batched.skip(testBatch)

print("Finished Processing Data")


class DCNN(tf.keras.Model):
    def __init__(
        self,
        filters=50,
        FFN=512,
        classes=2,
        dropoutRate=0.1,
        training=False,
        name="dcnn",
    ):
        super(DCNN, self).__init__(name=name)

        self.bertLayer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2",
            trainable=False,
        )

        self.bigram = Conv1D(
            filters=filters, kernel_size=2, padding="valid", activation="relu"
        )
        self.trigram = Conv1D(
            filters=filters, kernel_size=3, padding="valid", activation="relu"
        )
        self.fourgram = Conv1D(
            filters=filters, kernel_size=4, padding="valid", activation="relu"
        )

        self.pool = GlobalMaxPool1D()

        self.dense1 = Dense(units=FFN, activation="relu")
        self.dropout = Dropout(rate=dropoutRate)

        if classes == 2:
            self.dense2 = Dense(units=1, activation="sigmoid")
        else:
            self.dense2 = Dense(units=classes, activation="softmax")

    def call(self, inputs, training):
        _, x = self.bertLayer([inputs[:, 0, :], inputs[:, 1, :], inputs[:, 2, :]])

        x1 = self.pool(self.bigram(x))
        x2 = self.pool(self.trigram(x))
        x3 = self.pool(self.fourgram(x))

        output = self.dense2(
            self.dropout(self.dense1(tf.concat([x1, x2, x3], axis=-1)), training)
        )

        return output


filters = 100
FFN = 256
classes = 2
dropoutRate = 0.2
epochs = 5

dcnn = DCNN(filters=filters, FFN=FFN, classes=classes, dropoutRate=dropoutRate)

if classes == 2:
    dcnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

else:
    dcnn.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["sparse_categorical_accuracy"],
    )

dcnn.fit(trainDataset, epochs=epochs)


results = dcnn.evaluate(testDataset)
print(results)


def pred(sentence):
    tokens = encode(sentence)

    inputs = tf.expand_dims(
        tf.stack(
            [
                tf.cast(getIDs(tokens), dtype=tf.int32),
                tf.cast(getMask(tokens), dtype=tf.int32),
                tf.cast(getSegments(tokens), dtype=tf.int32),
            ],
            axis=0,
        ),
        0,
    )

    output = dcnn(inputs, training=False)

    sentiment = math.floor(output * 2)

    if sentiment == 0:
        print(
            f"Prediction Accuracy: {output[0][0]:.2f}\nPredicted Sentiment: Negative."
        )

    elif sentiment == 1:
        print(
            f"Prediction Accuracy: {output[0][0]*100:.2f}\nPredicted Sentiment: Positive."
        )


pred("This movie was convulated.")
