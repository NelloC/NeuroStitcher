import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from storage import NeuronDataStorage  # Importa la tua classe NeuronDataStorage

class ClusteringManager:
    def run_dbscan(self, neuron_data: NeuronDataStorage, min_samples=10, k=10, percentile=25):
        # Estrai i punti direttamente da neuron_data
        points = neuron_data.points

        # Esegue DBSCAN con parametri eps e density_threshold stimati dinamicamente
        eps_estimate = self.estimate_eps(points, k=k)
        dbscan = DBSCAN(eps=eps_estimate, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        density_threshold_estimate = self.estimate_density_threshold(labels, counts, percentile=percentile)
        
        avg_cluster_size = np.mean(counts[counts > 1]) if len(counts[counts > 1]) > 0 else 0
        tipo_struttura = 'densa' if avg_cluster_size > density_threshold_estimate else 'sparse'
        
        return labels, unique_labels, counts, tipo_struttura

    def estimate_eps(self, points, k=10):
        # Stima eps per DBSCAN usando la distanza dal k-esimo vicino più vicino
        neighbors = NearestNeighbors(n_neighbors=k).fit(points)
        distances, _ = neighbors.kneighbors(points)
        k_distances_sorted = np.sort(distances[:, k-1])
        return np.percentile(k_distances_sorted, 10)  # Percentile per stimare eps

    def estimate_density_threshold(self, labels, counts, percentile=5):
        # Calcola la soglia di densità basata sulla dimensione del cluster
        cluster_sizes = counts[labels[labels >= 0]]
        return np.percentile(cluster_sizes, percentile) if len(cluster_sizes) else 0

    def get_adaptive_params(self, cluster_points, avg_cluster_size, labels, unique_labels, counts, tipo_struttura, default_min_samples=10):
        # Funzione adattiva per regolare num_neighbors e distance_factor in base alla densità locale
        density_threshold = self.estimate_density_threshold(labels, counts)
        
        num_neighbors = default_min_samples
        distance_factor = 0.0
        
        if avg_cluster_size >= density_threshold:
            num_neighbors = max(3, default_min_samples - int((avg_cluster_size - density_threshold) / density_threshold * 3))
            distance_factor = max(0.1, 0.9 - (avg_cluster_size - density_threshold) / density_threshold * 0.1)
        else:
            num_neighbors = min(5, default_min_samples + int((density_threshold - avg_cluster_size) / density_threshold * 5))
            distance_factor = min(0.9, 0.1 + (density_threshold - avg_cluster_size) / density_threshold * 0.1)
        
        num_neighbors = min(5, max(3, num_neighbors))
        distance_factor = min(0.9, max(0.1, distance_factor))
        
        return num_neighbors, distance_factor


