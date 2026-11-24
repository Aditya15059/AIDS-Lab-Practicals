import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X, y = iris.data, iris.target
scaled = StandardScaler().fit_transform(X)
full = PCA().fit(scaled)
print("Explained variance:", full.explained_variance_ratio_.round(4))
print("Cumulative variance:", np.cumsum(full.explained_variance_ratio_).round(4))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(scaled)
print("Original shape:", X.shape, "Reduced shape:", X_pca.shape)
print("Variance kept:", pca.explained_variance_ratio_.sum())
plt.bar(range(1, 5), full.explained_variance_ratio_)
plt.plot(range(1, 5), np.cumsum(full.explained_variance_ratio_), marker="o", color="red")
plt.title("Scree Plot"); plt.xlabel("Principal Component"); plt.ylabel("Variance"); plt.show()
for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=name)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA of Iris"); plt.legend(); plt.show()
