import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv("AdsCTR.csv")

d = 10
N = 10000

noSelections = [0] * d
sumRewards = [0] * d
totalReward = 0

adsSelected = []

for n in range(0, N):
    ad = 0
    maxUCB = 0
    for i in range(0, d):
        if noSelections[i] > 0:
            avgReward = sumRewards[i] / noSelections[i]
            deltaI = math.sqrt(3 / 2 * math.log(n + 1) / noSelections[i])
            UCB = avgReward + deltaI

        else:
            UCB = 1e400

        if UCB > maxUCB:
            maxUCB = UCB
            ad = i

    adsSelected.append(ad)
    noSelections[ad] = noSelections[ad] + 1
    reward = dataset.values[n, ad]
    sumRewards[ad] = sumRewards[ad] + reward
    totalReward = totalReward + reward

plt.hist(adsSelected)
plt.title('Histogram Of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('Number Of Times Each Ad Was Selected')
plt.show()
