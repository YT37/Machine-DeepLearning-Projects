import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv("Spam.csv")

filtered = [word for word in stopwords.words("english") if word != "not"]
filtered.append("subject")

corpus = []
for text in range(0, 5728):
    ps = PorterStemmer()

    text = re.sub("[^a-zA-Z]", " ", dataset["Text"][text]).lower().split()
    text = [ps.stem(word) for word in text if not word in filtered]
    text = " ".join(text)

    corpus.append(text)

cv = CountVectorizer(max_features=1500)
X = TfidfTransformer().fit_transform(cv.fit_transform(corpus).toarray())
y = dataset.iloc[:, 1]

Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = MultinomialNB(alpha=0.01)
classifier.fit(Xtrain, yTrain)

yPred = classifier.predict(Xtest)

cm = confusion_matrix(yTest, yPred)

accuracy = accuracy_score(yTest, yPred)
precision = int((cm[1][1] / (cm[1][1] + cm[0][1])) * 10 ** 3) / 10 ** 3
recall = int((cm[1][1] / (cm[1][1] + cm[1][0])) * 10 ** 3) / 10 ** 3
f1Score = int((2 * precision * recall / (precision + recall)) * 10 ** 3) / 10 ** 3
