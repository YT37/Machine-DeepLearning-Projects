import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

dataset = pd.read_csv("MovieReviews.tsv", delimiter="\t", quoting=3)

filtered = [word for word in stopwords.words("english") if word != "not"]

corpus = []
for review in range(0, 25000):
    ps = PorterStemmer()

    review = re.sub("[^a-zA-Z]", " ", dataset["review"][review]).lower().split()
    review = [ps.stem(word) for word in review if not word in filtered]
    review = " ".join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1]

Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = SVC(kernel="linear", random_state=0)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)

cm = confusion_matrix(yTest, yPred)

accuracy = accuracy_score(yTest, yPred)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10 ** 3) / 10 ** 3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10 ** 3) / 10 ** 3
f1Score = int((2 * precision * recall / (precision + recall)) * 10 ** 3) / 10 ** 3
