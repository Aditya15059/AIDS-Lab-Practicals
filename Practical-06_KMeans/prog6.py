import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data
scaled = StandardScaler().fit_transform(X)
inertia = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
    inertia.append(km.inertia_)
print("Inertia:", np.round(inertia, 2))
plt.plot(range(1, 9), inertia, marker="o")
plt.xlabel("K"); plt.ylabel("Inertia"); plt.title("Elbow Method"); plt.grid(); plt.show()
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(scaled)
print("Cluster sizes:", np.bincount(labels))
print("Centroids (scaled):\n", km.cluster_centers_.round(2))
plt.scatter(X[:, 2], X[:, 3], c=labels)
plt.xlabel("Petal length"); plt.ylabel("Petal width"); plt.title("K-Means (K=3)")
plt.show()
