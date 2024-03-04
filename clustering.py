import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class KMeans_Cluster:
    def __init__(self, filename: str):
        # Load the dataset
        self.iris_data = pd.read_csv(filename)

        # Assuming the last column is the species or target, we'll use all but the last column for clustering
        X = self.iris_data.iloc[:, :-1].values

        # Scaling the features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)

        # K-Means clustering
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.kmeans.fit_predict(self.X_scaled)
        
        # Perform PCA for dimensionality reduction to visualize in 2D
        self.pca = PCA(n_components=2)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

    def get_pca_data(self):
        return self.X_pca

    def get_centroids(self):
        return self.pca.transform(self.kmeans.cluster_centers_)


