import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv(
    r"D:\Codes\MachineLearning\Projects\Clustering\MallCustomerCluster\MallCustomers.csv"
)

X = dataset.iloc[:, [3, 4]].values

kmeans = KMeans(n_clusters=5,
                init="k-means++",s
                max_iter=300,
                n_init=10,
                random_state=0)
yKmeans = kmeans.fit_predict(X)

plt.scatter(X[yKmeans == 0, 0],
            X[yKmeans == 0, 1],
            s=100,
            c="red",
            label="Careful")
plt.scatter(X[yKmeans == 1, 0],
            X[yKmeans == 1, 1],
            s=100,
            c="blue",
            label="Standard")
plt.scatter(X[yKmeans == 2, 0],
            X[yKmeans == 2, 1],
            s=100,
            c="green",
            label="Target")
plt.scatter(X[yKmeans == 3, 0],
            X[yKmeans == 3, 1],
            s=100,
            c="cyan",
            label="Careless")
plt.scatter(X[yKmeans == 4, 0],
            X[yKmeans == 4, 1],
            s=100,
            c="yellow",
            label="Sensible")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="magenta",
    label="Centeroids",
)
plt.title("Clusters Of Customers")
plt.xlabel("Anual Income (K$)")
plt.ylabel("Spendeing Scoore (1-100)")
plt.legend()
plt.show()
