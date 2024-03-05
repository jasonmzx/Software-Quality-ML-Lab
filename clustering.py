#* Code By: Jason Manarroo | 100825106

import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import silhouette_score


# Base Class AKA Interface

class ClusterBase(ABC):
    def __init__(self, filename: str):
        self.csv_data = pd.read_csv(filename)

        #Seperates Numerical Data from Categoric, As last Column is labels (For later performance testing, etc...)
        X = self.csv_data.iloc[:, :-1].values

        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)
        self.pca = PCA(n_components=2)  # Initialize here, but fit in subclasses

    @abstractmethod
    def get_pca_data(self):
        pass

    @abstractmethod
    def get_centroids(self):
        pass

    @abstractmethod
    def get_labels(self):
        pass

    @abstractmethod
    def get_clust_obj(self):
        pass

    @abstractmethod
    def get_score(self):
        pass

#* ---------------------------------- // Clustering Algorithm Class Impls (using ClusterBase) // -------------------

class KMeans_Cluster(ClusterBase):
    def __init__(self, filename: str):
        super().__init__(filename)  # Initialize the base class

        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.kmeans.fit(self.X_scaled)
        
        self.X_pca = self.pca.fit_transform(self.X_scaled)

    def get_pca_data(self):
        return self.X_pca

    def get_centroids(self):
        return self.pca.transform(self.kmeans.cluster_centers_)

    def get_labels(self):
        return self.kmeans.labels_
    
    def get_clust_obj(self):
        return self.kmeans

    def get_score(self):
        return silhouette_score(self.X_scaled, self.kmeans.labels_)
    
    def get_inertia(self): #* ONLY FOR K-MEANS VARIANTS, as it relies on accurate centroid
        return self.kmeans.inertia_

class Mini_KMeans_Cluster(ClusterBase):
    def __init__(self, filename: str):
        super().__init__(filename)  # Base class initialization

        self.kmeans_mini = MiniBatchKMeans(n_clusters=3, random_state=42)
        self.kmeans_mini.fit(self.X_scaled)
        
        self.X_pca = self.pca.fit_transform(self.X_scaled)

    def get_pca_data(self):
        return self.X_pca

    def get_centroids(self):
        return self.pca.transform(self.kmeans_mini.cluster_centers_)

    def get_labels(self):
        return self.kmeans_mini.labels_
    
    def get_clust_obj(self):
        return self.kmeans_mini

    def get_score(self):
        return silhouette_score(self.X_scaled, self.kmeans_mini.labels_)

    def get_inertia(self): #* ONLY FOR K-MEANS VARIANTS, as it relies on accurate centroid
        return self.kmeans_mini.inertia_

class Agglomerative_Cluster(ClusterBase):
    def __init__(self, filename: str):
        super().__init__(filename)  # Base class initialization

        self.agglomerative = AgglomerativeClustering(n_clusters=3)
        self.cluster_labels = self.agglomerative.fit_predict(self.X_scaled)
        
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.centroids = self.calculate_centroids()

    def calculate_centroids(self):
        centroids = np.zeros((3, self.X_scaled.shape[1]))
        for i in range(3):
            cluster = self.X_scaled[self.cluster_labels == i]
            cluster_mean = cluster.mean(axis=0)
            centroids[i, :] = cluster_mean
        return self.pca.transform(centroids)

    def get_pca_data(self):
        return self.X_pca

    def get_centroids(self):
        return self.centroids

    def get_labels(self):
        return self.cluster_labels
    
    def get_clust_obj(self):
        return self.agglomerative

    def get_score(self):
        return silhouette_score(self.X_scaled, self.agglomerative.labels_)

