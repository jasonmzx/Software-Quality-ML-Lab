import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from clustering import KMeans_Cluster, Mini_KMeans_Cluster, Agglomerative_Cluster

# Assuming KMeans_Cluster is defined as previously described

kmeans_cluster = Agglomerative_Cluster("./iris.csv")
pca_data = kmeans_cluster.get_pca_data()
centroids = kmeans_cluster.get_centroids()

print("PCA Data:\n", pca_data[:5])  # Printing the first 5 rows as an example
print("\nCentroids:\n", centroids)

# Plotting the PCA-reduced data with cluster assignments
plt.figure(figsize=(8, 6))

plt.scatter(pca_data[:, 0], 
            pca_data[:, 1], 
            c=kmeans_cluster.get_clust_obj().labels_, 
            cmap='viridis', 
            marker='o', edgecolor='k', label='Cluster Points')

# Plotting the cluster centers
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('Agglomerative Cluster of Iris Dataset (PCA-reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

