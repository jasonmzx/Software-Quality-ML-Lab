import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np

#* Code By: Jason Manarroo | 100825106

class KMeans_Cluster:
    def __init__(self, filename: str):
        # Load the dataset
        self.iris_data = pd.read_csv(filename)

        # Split all the Numerical Data, from Catergorical String Data
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
    
    def get_clust_obj(self): # Gives Main or test fn. the cluster object
        return self.kmeans



class Mini_KMeans_Cluster:
    def __init__(self, filename: str):
        self.iris_data = pd.read_csv(filename)
        X = self.iris_data.iloc[:, :-1].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.kmeans_mini = MiniBatchKMeans(n_clusters=3, random_state=42)
        self.kmeans_mini.fit_predict(X_scaled)
        
        pca = PCA(n_components=2)
        self.X_pca = pca.fit_transform(X_scaled)
        
        self.centroids = pca.transform(self.kmeans_mini.cluster_centers_)

    def get_pca_data(self):
        return self.X_pca

    def get_centroids(self):
        return self.centroids

    def get_clust_obj(self): # Gives Main or test fn. the cluster object
        return self.kmeans_mini

class Agglomerative_Cluster:
    def __init__(self, filename: str):
        self.iris_data = pd.read_csv(filename)
        X = self.iris_data.iloc[:, :-1].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.agglomerative = AgglomerativeClustering(n_clusters=3)
        self.cluster_labels = self.agglomerative.fit_predict(X_scaled)
        
        pca = PCA(n_components=2)
        self.X_pca = pca.fit_transform(X_scaled)
        
        # Since AgglomerativeClustering doesn't directly provide cluster centers, we calculate them
        self.centroids = self.calculate_centroids(X_scaled)

    def calculate_centroids(self, X_scaled):
        centroids = np.zeros((3, X_scaled.shape[1]))
        for i in range(3):
            cluster = X_scaled[self.cluster_labels == i]
            cluster_mean = cluster.mean(axis=0)
            centroids[i, :] = cluster_mean
        return PCA(n_components=2).fit_transform(centroids)

    def get_pca_data(self):
        return self.X_pca

    def get_centroids(self):
        return self.centroids
    
    def get_clust_obj(self): # Gives Main or test fn. the cluster object
        return self.agglomerative

