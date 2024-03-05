import unittest
import time
from clustering import KMeans_Cluster, Mini_KMeans_Cluster, Agglomerative_Cluster

class ClusteringPerformanceTest(unittest.TestCase):
    
    def setUp(self):
        self.dataset_path = "./iris.csv"
    
    def measure_performance(self, clustering_class):
        start_time = time.time()
        clustering_instance = clustering_class(self.dataset_path)
        end_time = time.time()
        execution_time = end_time - start_time
        silhouette = clustering_instance.get_score()
        inertia = None
        if hasattr(clustering_instance, 'get_inertia'):
            inertia = clustering_instance.get_inertia()
        return execution_time, silhouette, inertia
    
    def test_kmeans_vs_mini_kmeans_speed(self):
        kmeans_time, _, _ = self.measure_performance(KMeans_Cluster)
        mini_kmeans_time, _, _ = self.measure_performance(Mini_KMeans_Cluster)
        print(f"KMeans Time: {kmeans_time}, Mini KMeans Time: {mini_kmeans_time}")
        self.assertLess(mini_kmeans_time, kmeans_time, "Mini KMeans should be faster than KMeans")
    
    def test_mini_kmeans_vs_agglomerative_speed(self):
        mini_kmeans_time, _, _ = self.measure_performance(Mini_KMeans_Cluster)
        agglomerative_time, _, _ = self.measure_performance(Agglomerative_Cluster)
        print(f"Mini KMeans Time: {mini_kmeans_time}, Agglomerative Time: {agglomerative_time}")
        # No direct assertion here since speed can vary based on implementation details and data characteristics
    
    def test_kmeans_vs_agglomerative_speed(self):
        kmeans_time, _, _ = self.measure_performance(KMeans_Cluster)
        agglomerative_time, _, _ = self.measure_performance(Agglomerative_Cluster)
        print(f"KMeans Time: {kmeans_time}, Agglomerative Time: {agglomerative_time}")
        # No direct assertion here since speed can vary based on implementation details and data characteristics

if __name__ == '__main__':
    unittest.main()