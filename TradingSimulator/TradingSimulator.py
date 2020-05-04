import math
import random
import numpy as np
import tqdm
from collections import deque
import pandas_datareader as reader
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


class Trader:
    def __init__(self, size, space=3):

        self.size = size
        self.space = space
        self.memory = deque(maxlen=2000)
        self.inventory = []

        self.gamma = 0.95
        self.epsilon = 1.0
        self.final = 0.01
        self.decay = 0.995

        self.model = self.builder()

    def builder(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.size))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dense(units=128, activation="relu"))
        model.add(Dense(units=self.space, activation="linear"))
        model.compile(loss="mse", optimizer="adam")

        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.space)

        actions = self.model.predict(state)

        return np.argmax(actions[0])

    def batchTrade(self, batchSize):
        batch = []

        for i in range(len(self.memory) - batchSize + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, nextState, complete in batch:
            reward = reward

            if not complete:
                reward = reward + self.gamma * np.amax(self.model.predict(nextState)[0])

            target = self.model.predict(state)
            target[0][action] = reward

            self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.final:
            self.epsilon *= self.decay


def stockFormat(n):
    if n < 0:
        return f"- $ {abs(n):2f}"

    else:
        return f"$ {abs(n):2f}"


def creator(data, timestep, size):

    id = timestep - size + 1

    if id >= 0:
        data = data[id : timestep + 1]
    else:
        data = -id * [data[0]] + list(data[0 : timestep + 1])

    state = []
    for i in range(size - 1):
        state.append(1 / (1 + math.exp(-(data[i + 1] - data[i]))))

    return np.array([state])


data = dataset = reader.DataReader("AAPL", data_source="yahoo")["Close"]

size = 60
episodes = 1
batchSize = 32
samples = len(data) - 1

trader = Trader(size)


for episode in range(1, episodes + 1):

    print(f"Episode: {episode}/{episodes}")

    state = creator(data, 0, size + 1)

    profit = 0
    trader.inventory = []

    for s in tqdm.tqdm(range(samples)):

        action = trader.trade(state)

        nextState = creator(data, s + 1, size + 1)
        reward = 0

        if action == 1:
            trader.inventory.append(data[s])
            print(f"Bought: {stockFormat(data[s])}")

        elif action == 2 and len(trader.inventory) > 0:
            buyPrice = trader.inventory.pop(0)

            reward = max(data[s] - buyPrice, 0)
            profit += data[s] - buyPrice

            print(
                f"Sold: {stockFormat(data[s])}, Profit: {stockFormat(data[s] - buyPrice)}"
            )

        if s == samples - 1:
            complete = True
        else:
            complete = False

        trader.memory.append((state, action, reward, nextState, complete))

        state = nextState

        if complete:
            print("########################")
            print(f"Total Profit: {profit}")
            print("########################")

        if len(trader.memory) > batchSize:
            trader.batchTrade(batchSize)
