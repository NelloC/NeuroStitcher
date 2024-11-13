import numpy as np
from scipy.spatial import KDTree
from storage import NeuronDataStorage

class StitchingManager:
    def __init__(self, vector_calculator, clustering_manager, attractor_points_manager, neuron_data):
        self.vector_calculator = vector_calculator
        self.clustering_manager = clustering_manager
        self.attractor_points_manager = attractor_points_manager
        self.neuron_data = neuron_data

        # Inizializza i pesi
        self.weights = {
            "distance": 0.4,
            "growth": 0.2,
            "direction": 0.2,
            "curvature": 0.2
        }

    def find_best_stitches(self, neuron_data, pieces):
        attractor_points = self.attractor_points_manager.calculate_attractor_points(neuron_data, pieces, self.clustering_manager)
        labels, unique_labels, counts, tipo_struttura = self.clustering_manager.run_dbscan(neuron_data.points)

        best_stitches = {}

        for piece_id in pieces:
            piece_id_tuple = tuple(piece_id)

            if not any(np.array_equal(piece_id_tuple, key) for key in attractor_points.keys()):
                continue

            attractor_info = attractor_points[piece_id_tuple]
            growth_vector = attractor_info["growth_vector"]
            neighbor_points = attractor_info["neighbor_points"]
            neighbor_growth_vectors = attractor_info["neighbor_growth_vectors"]
            direction_vectors = attractor_info["direction_vectors"]
            distances = attractor_info["distances"]
            piece_points = np.array([self.neuron_data.points[idx][:3] for idx in piece_id])
            endings = [piece_points[0], piece_points[-1]]  # Modificato per coerenza con AttractorPointsManager
            #endings = self.get_endings(piece_id)
            for ending in endings:
                valid_distances = distances[distances > 0]
                if valid_distances.size == 0:
                    continue
                min_distance = np.min(valid_distances)

                _, distance_factor = self.clustering_manager.get_adaptive_params(
                    cluster_points=neuron_data.points,
                    avg_cluster_size=np.mean(counts) if counts.size > 0 else 1,
                    labels=labels,
                    unique_labels=unique_labels,
                    counts=counts,
                    tipo_struttura=tipo_struttura
                )
                threshold = min_distance + min_distance * distance_factor
                # Usa adapt_weights per ottenere i pesi dinamici
                w_dist, w_growth, w_dir, w_cur = self.adapt_weights_based_on_dbscan_density(
                    points=neuron_data.points,
                    labels=labels,
                    unique_labels=unique_labels,
                    counts=counts,
                    tipo_struttura=tipo_struttura
                )
                best_candidate = None
                best_score = float('inf')

                for i, (neighbor_point, neighbor_growth_vector, direction_vector, distance) in enumerate(zip(neighbor_points, neighbor_growth_vectors, direction_vectors, distances)):
                    if distance <= 0 or distance > threshold:
                        continue
                    else:

                        if np.linalg.norm(growth_vector) == 0 or np.linalg.norm(neighbor_growth_vector) == 0:
                            continue

                        growth_angle = self.vector_calculator.calculate_curvature_change(growth_vector, neighbor_growth_vector)
                        direction_angle = self.vector_calculator.calculate_curvature_change(growth_vector, direction_vector)

                        score = self.calculate_score_with_sigmoid(
                            distance=distance,
                            growth_angle=growth_angle,
                            direction_angle=direction_angle,
                            curvature_change=growth_angle,
                            w_dist=w_dist,
                            w_growth=w_growth,
                            w_dir=w_dir,
                            w_cur=w_cur
                        )

                        if score < best_score:
                            best_score = score
                            best_candidate = (ending, np.array(neighbor_point[:3]))

                if best_candidate:
                    if piece_id_tuple not in best_stitches:
                        best_stitches[piece_id_tuple] = []

                    best_stitches[piece_id_tuple].append({
                        "ending": best_candidate[0],
                        "candidate": best_candidate[1],
                        "score": best_score,
                        "attractors": neighbor_points,
                        "growth_vector": growth_vector,
                        "neighbor_growth_vector":neighbor_growth_vector,
                        "direction_vectors": direction_vectors,
                        "cluster_labels": labels
                    })

        # Ordina i candidati per piece_id in base al punteggio
        for piece_id_tuple in best_stitches:
            best_stitches[piece_id_tuple] = sorted(best_stitches[piece_id_tuple], key=lambda x: x["score"])

        return best_stitches





    def get_candidates(self, piece_id, lines):
        candidates = []
        for line in lines:
            if not np.array_equal(line[0], piece_id):
                candidates.append(line[1])
        print(f"[DEBUG] Candidati per il pezzo {piece_id}: {candidates}")
        return candidates

    def adapt_weights_based_on_dbscan_density(self, points, labels, unique_labels, counts, tipo_struttura, percentile=50):
        avg_cluster_size = np.mean(counts[counts > 1]) if len(counts[counts > 1]) > 0 else 0
        density_threshold_estimate = self.clustering_manager.estimate_density_threshold(labels, counts, percentile=percentile)
        
        if avg_cluster_size > density_threshold_estimate:
            w_dist, w_growth, w_dir, w_cur = 0.4, 0.2, 0.3, 0.1
        else:
            w_dist, w_growth, w_dir, w_cur = 0.2, 0.3, 0.3, 0.2
            
        #print(f"[DEBUG] Pesi adattati: w_dist={w_dist}, w_growth={w_growth}, w_dir={w_dir}, w_cur={w_cur}")
        return w_dist, w_growth, w_dir, w_cur

    def calculate_score_with_sigmoid(self, distance, growth_angle, direction_angle, curvature_change, w_dist, w_growth, w_dir, w_cur):
        norm_dist = 1 / (1 + np.exp(-distance))
        norm_growth = 1 / (1 + np.exp(-growth_angle))
        norm_direction = 1 / (1 + np.exp(-direction_angle))
        norm_curvature = 1 / (1 + np.exp(-curvature_change))

        score = (w_dist * norm_dist +
                w_growth * norm_growth +
                w_dir * norm_direction +
                w_cur * norm_curvature)

        #print(f"[DEBUG] Final Score Components -> Distance: {norm_dist}, Growth: {norm_growth}, "
            #f"Direction: {norm_direction}, Curvature: {norm_curvature}")
        return score



    def calculate_curvature_change(self, growth_vector, neighbor_growth_vector):
        try:
            if np.linalg.norm(growth_vector[:3]) == 0 or np.linalg.norm(neighbor_growth_vector[:3]) == 0:
                print("[DEBUG] Zero norm detected, assigning max curvature.")
                return np.pi
            norm_growth_vector = growth_vector[:3] / np.linalg.norm(growth_vector[:3])
            norm_neighbor_growth_vector = neighbor_growth_vector[:3] / np.linalg.norm(neighbor_growth_vector[:3])
            dot_product = np.clip(np.dot(norm_growth_vector, norm_neighbor_growth_vector), -1.0, 1.0)
            return np.arccos(dot_product)
        except Exception as e:
            print(f"[ERROR] Error calculating curvature change: {e}")
            return np.pi

    def get_endings(self, piece_id):
        """
        Ritorna gli ending points come singoli punti (iniziale e finale).
        """
        endings = []
        for line_id in piece_id:
            if line_id < 0 or line_id >= len(self.neuron_data.lines):
                continue  # Evita errori di indice
            
            line = self.neuron_data.lines[line_id]
            
            # Recupera il punto iniziale e finale della linea
            starting_point = self.neuron_data.points[line[1]][:3]
            ending_point = self.neuron_data.points[line[1] + line[2] - 1][:3]
            
            endings.append(starting_point)
            endings.append(ending_point)
            
        return endings


    def get_endpoint(self, ending, points):
        return points[ending]




