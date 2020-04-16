import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("Ads.csv")

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

Xtrain, Xtest, yTrain, yTest = train_test_split(
    X, y, test_size=0.25, random_state=0)

scX = StandardScaler()
Xtrain = scX.fit_transform(Xtrain)
Xtest = scX.transform(Xtest)

classifier = GaussianNB()
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)

cm = confusion_matrix(yTest, yPred)

accuracy = accuracy_score(yTest, yPred)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10**3) / 10**3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10**3) / 10**3
f1Score = int((2 * precision * recall / (precision + recall)) * 10**3) / 10**3

Xset, ySet = Xtest, yTest
X1, X2 = np.meshgrid(np.arange(start=Xset[:, 0].min() - 1, stop=Xset[:, 0].max() + 1, step=0.01),
                     np.arange(start=Xset[:, 1].min() - 1, stop=Xset[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(ySet)):
    plt.scatter(Xset[ySet == j, 0], Xset[ySet == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Ads Classification')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
