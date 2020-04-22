import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv("Movies.csv",
                     delimiter="::",
                     engine="python",
                     encoding="latin-1")

ratings = pd.read_csv("Ratings.csv", encoding="latin-1")

trainingSet = np.array(pd.read_csv("TrainingSet.csv"), dtype="int")
testingSet = np.array(pd.read_csv("TestingSet.csv"), dtype="int")

nbUser = int(max(max(trainingSet[:, 0]), max(testingSet[:, 0])))
nbMovies = int(max(max(trainingSet[:, 1]), max(testingSet[:, 1])))


def convert(data):
    dataset = []

    for userId in range(1, nbUser + 1):
        movieId = data[:, 1][data[:, 0] == userId]
        ratingsId = data[:, 2][data[:, 0] == userId]
        ratings = np.zeros(nbMovies)
        ratings[movieId - 1] = ratingsId
        dataset.append(list(ratings))

    return dataset


trainingSet, testingSet = torch.FloatTensor(
    convert(trainingSet)), torch.FloatTensor(convert(testingSet))

trainingSet[trainingSet == 0] = -1
trainingSet[trainingSet == 1] = 0
trainingSet[trainingSet == 2] = 0
trainingSet[trainingSet >= 3] = 1

testingSet[testingSet == 0] = -1
testingSet[testingSet == 1] = 0
testingSet[testingSet == 2] = 0
testingSet[testingSet >= 3] = 1


class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nbMovies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nbMovies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.fc4(
            self.activation(
                self.fc3(
                    self.activation(self.fc2(self.activation(self.fc1(x)))))))


sae = SAE()

epoch = 500
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

for nbEpoch in range(1, epoch + 1):
    loss = 0
    s = 0.0

    for user in range(nbUser):
        input = Variable(trainingSet[user]).unsqueeze(0)
        target = input.clone()

        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            trainLoss = criterion(output, target)
            corrector = nbMovies / float(torch.sum(target.data > 0) + 1e-10)
            trainLoss.backward()
            loss += np.sqrt(trainLoss.data * corrector)
            s += 1.0
            optimizer.step()

    print(f"Epoch: {nbEpoch} Loss: {loss/s}")

loss = 0
s = 0.0

for user in range(nbUser):
    input = Variable(trainingSet[user]).unsqueeze(0)
    target = Variable(testingSet[user]).unsqueeze(0)

    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        testLoss = criterion(output, target)
        corrector = nbMovies / float(torch.sum(target.data > 0) + 1e-10)
        loss += np.sqrt(testLoss.data * corrector)
        s += 1.0

print(f"Test Loss: {loss/s}")
