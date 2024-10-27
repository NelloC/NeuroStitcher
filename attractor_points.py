import numpy as np
from scipy.spatial import KDTree
from storage import NeuronDataStorage  # Importa la classe NeuronDataStorage

class AttractorPointsManager:
    def __init__(self, vector_calculator):
        self.vector_calculator = vector_calculator
        self.kdtree = None
        self.attractorPointsCount = {}

    def calculate_attractor_points(self, neuron_data: NeuronDataStorage, pieces, clustering_manager, num_neighbors=5, density_threshold=100):
        # Estrai i punti direttamente da neuron_data
        points = neuron_data.points
        
        attractor_points = {}
        growth_vectors = {}
        neighbor_growth_vectors = {}
        direction_vectors = {}
        distances = {}
        distance_factors = {}
        
        # Costruisci KDTree per la ricerca dei vicini
        self.kdtree = self.build_kdtree(points)
        
        for piece_id in pieces:
            piece_points = points[piece_id]
            growth_vector = self.vector_calculator.calculate_growth_vector(piece_points)
            
            # Ottieni vicini e distanze usando KDTree
            neighbors_indices, neighbors_distances = self.find_neighbors(piece_points[-1], num_neighbors)
            neighbor_points = points[neighbors_indices]
            
            growth_vectors[piece_id] = growth_vector
            distances[piece_id] = neighbors_distances
            neighbor_growth_vectors[piece_id] = []
            direction_vectors[piece_id] = []
            
            for neighbor_point in neighbor_points:
                neighbor_growth_vector = self.vector_calculator.calculate_growth_vector(neighbor_point)
                neighbor_growth_vectors[piece_id].append(neighbor_growth_vector)
                
                direction_vector = self.vector_calculator.calculate_direction_vector(piece_points[-1], neighbor_point)
                direction_vectors[piece_id].append(direction_vector)
                
            attractor_points[piece_id] = {
                "growth_vector": growth_vector,
                "neighbor_points": neighbor_points,
                "neighbor_growth_vectors": neighbor_growth_vectors[piece_id],
                "direction_vectors": direction_vectors[piece_id],
                "distances": neighbors_distances
            }
            self.attractorPointsCount[piece_id] = len(attractor_points[piece_id])

        # Ordinamento per densità
        density_mapping = {pid: len(pts) / max(1, sum(distances[pid])) for pid, pts in attractor_points.items()}
        sorted_attractor_points = {k: v for k, v in sorted(attractor_points.items(), key=lambda item: density_mapping[item[0]], reverse=True)}
        
        return sorted_attractor_points, self.attractorPointsCount, growth_vectors, neighbor_growth_vectors, direction_vectors, distances, distance_factors

    def build_kdtree(self, points):
        return KDTree(points)

    def find_neighbors(self, point, num_neighbors=5):
        if self.kdtree is None:
            raise ValueError("KDTree non è stato costruito. Costruisci la KDTree prima di cercare i vicini.")
        
        distances, indices = self.kdtree.query(point, k=num_neighbors)
        return indices, distances




