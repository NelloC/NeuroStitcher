import numpy as np
from scipy.spatial import KDTree
from storage import NeuronDataStorage  # Importa la classe NeuronDataStorage

class StitchingManager:
    def __init__(self, vector_calculator, clustering_manager, attractor_points_manager):
        self.vector_calculator = vector_calculator
        self.clustering_manager = clustering_manager
        self.attractor_points_manager = attractor_points_manager

    def find_best_stitches(self, neuron_data: NeuronDataStorage, pieces):
        # Calcola i punti di attrazione e i cluster con i dati neuronali
        attractor_points, _, _, _, _, distances, _ = self.attractor_points_manager.calculate_attractor_points(neuron_data, pieces, self.clustering_manager)
        labels, unique_labels, counts, tipo_struttura = self.clustering_manager.run_dbscan(neuron_data.points)
        stitches = []
        
        # Adattamento dinamico dei pesi
        w_dist, w_growth, w_dir, w_cur = self.adapt_weights_based_on_dbscan_density(neuron_data.points, labels, unique_labels, counts, tipo_struttura)

        for piece_id in pieces:
            best_score = None
            best_stitch = None
            endings = self.get_endings(piece_id, neuron_data.lines)
            
            for ending in endings:
                endpoint = self.get_endpoint(ending, neuron_data.points)
                cluster_points = neuron_data.points[labels == labels[ending]]
                growth_vector = self.vector_calculator.calculate_growth_vector(cluster_points)
                
                neighbors, neighbor_distances = self.attractor_points_manager.find_neighbors(endpoint)
                
                for neighbor, distance in zip(neighbors, neighbor_distances):
                    direction_vector = self.vector_calculator.calculate_direction_vector(endpoint, neighbor)
                    angle = self.vector_calculator.calculate_angle_between_vectors(growth_vector, direction_vector)
                    curvature_change = self.vector_calculator.calculate_curvature_change(growth_vector, neighbor)
                    
                    score = self.calculate_score_with_sigmoid(distance, angle, growth_vector, direction_vector, curvature_change, w_dist, w_growth, w_dir, w_cur)
                    
                    if best_score is None or score < best_score:
                        best_score = score
                        best_stitch = (piece_id, neighbor)
            
            if best_stitch:
                stitches.append(best_stitch)
        
        return stitches

    def adapt_weights_based_on_dbscan_density(self, points, labels, unique_labels, counts, tipo_struttura, percentile=50):
        avg_cluster_size = np.mean(counts[counts > 1]) if len(counts[counts > 1]) > 0 else 0
        density_threshold_estimate = self.clustering_manager.estimate_density_threshold(labels, counts, percentile=percentile)
        
        if avg_cluster_size > density_threshold_estimate:
            w_dist, w_growth, w_dir, w_cur = 0.1, 0.4, 0.3, 0.2
        else:
            w_dist, w_growth, w_dir, w_cur = 0.3, 0.25, 0.25, 0.2
            
        return w_dist, w_growth, w_dir, w_cur

    def calculate_score_with_sigmoid(self, distance, angle, growth_vector, direction_vector, curvature_change, w_dist, w_growth, w_dir, w_cur):
        norm_dist = 1 / (1 + np.exp(-distance))
        norm_growth = 1 / (1 + np.exp(-angle))
        norm_direction = 1 / (1 + np.exp(-np.linalg.norm(direction_vector)))
        norm_curvature = 1 / (1 + np.exp(-curvature_change))
        
        score = (w_dist * norm_dist +
                 w_growth * norm_growth +
                 w_dir * norm_direction +
                 w_cur * norm_curvature)
        return score

    def calculate_curvature_change(self, growth_vector, neighbor_growth_vector):
        norm_growth_vector = growth_vector / np.linalg.norm(growth_vector)
        norm_neighbor_growth_vector = neighbor_growth_vector / np.linalg.norm(neighbor_growth_vector)
        dot_product = np.dot(norm_growth_vector, norm_neighbor_growth_vector)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        curvature_change = np.arccos(dot_product)
        return curvature_change

    def get_endings(self, piece_id, lines):
        return [line for line in lines if line[0] == piece_id]

    def get_endpoint(self, ending, points):
        return points[ending]


