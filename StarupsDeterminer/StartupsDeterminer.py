import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("Startups.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

ct = ColumnTransformer([("encoder", OneHotEncoder(), [3])], remainder="passthrough")
X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:, 1:]

Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(Xtrain, yTrain)

yPred = regressor.predict(Xtest)

# Feature Selection
"""import statsmodels.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

Xopt = X[:, [0, 1, 2, 3, 4, 5]]
regressorOls = sm.OLS(endog=y, exog=Xopt).fit()
regressorOls.summary()

Xopt = X[:, [0, 1, 3, 4, 5]]
regressorOls = sm.OLS(endog=y, exog=Xopt).fit()
regressorOls.summary()

Xopt = X[:, [0, 3, 4, 5]]
regressorOls = sm.OLS(endog=y, exog=Xopt).fit()
regressorOls.summary()

Xopt = X[:, [0, 3, 5]]
regressorOls = sm.OLS(endog=y, exog=Xopt).fit()
regressorOls.summary()"""
