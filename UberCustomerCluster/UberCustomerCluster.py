import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv(
    r"D:\Codes\MachineLearning\Projects\Clustering\UberCustomerCluster\Uber.csv"
)

X = dataset.iloc[:, [2, 3]].values

kmeans = KMeans(n_clusters=3,
                init="k-means++",
                max_iter=300,
                n_init=10,
                random_state=0)
yKmeans = kmeans.fit_predict(X)

plt.scatter(X[yKmeans == 0, 0],
            X[yKmeans == 0, 1],
            s=50,
            c="red",
            label="Medium")
plt.scatter(X[yKmeans == 1, 0],
            X[yKmeans == 1, 1],
            s=50,
            c="blue",
            label="High")
plt.scatter(X[yKmeans == 2, 0],
            X[yKmeans == 2, 1],
            s=50,
            c="green",
            label="Low")
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=100,
            c="magenta",
            label="Centeroids")
plt.title("Clusters Of Customers")
plt.xlabel("Active")
plt.ylabel("Trips")
plt.legend()
plt.show()
