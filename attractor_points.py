import numpy as np
from scipy.spatial import KDTree
from storage import NeuronDataStorage  # Importa la classe NeuronDataStorage

class AttractorPointsManager:
    def __init__(self, vector_calculator):
        self.vector_calculator = vector_calculator
        self.kdtree = None

    def calculate_attractor_points(self, neuron_data, pieces, clustering_manager, window_size=7):
        points = neuron_data.points
        attractor_points = {}

        # Estrai i punti finali di tutti i pezzi
        end_points = self.get_end_points(neuron_data, pieces)
        self.kdtree = self.build_kdtree(end_points)

        labels, unique_labels, counts, tipo_struttura = clustering_manager.run_dbscan(points)
        num_neighbors, distance_factor = clustering_manager.get_adaptive_params(
            points, np.mean(counts[counts > 1]) if len(counts[counts > 1]) > 0 else 0, labels, unique_labels, counts, tipo_struttura
        )

        # Mappa gli ending points ai rispettivi pezzi per un accesso più rapido
        piece_endings_map = {
            tuple(ending): piece
            for piece in pieces
            for ending in self.get_endings(np.array([points[idx] for idx in piece if idx < len(points)]))
            if len(ending) > 0  # Verifica se ending non è vuoto
        }

        for piece_id in pieces:
            piece_id_tuple = tuple(piece_id)
            piece_points = np.array([points[idx] for idx in piece_id if idx < len(points)])

            if piece_points.size == 0:
                continue

            # Trova tutti i punti finali per il pezzo corrente
            ending_points = self.get_endings(piece_points)

            attractor_points[piece_id_tuple] = []
            for current_ending_point in ending_points:
                growth_vector = self.vector_calculator.calculate_growth_vector(piece_points[-window_size:])

                # Trova gli ending vicini a quello attuale
                neighbors_indices, neighbors_distances = self.find_neighbors(current_ending_point, num_neighbors, end_points)

                neighbor_points = end_points[neighbors_indices]
                neighbor_growth_vectors = []
                direction_vectors = []

                # Mappa neighbor points ai loro pezzi originali
                for neighbor_point in neighbor_points:
                    neighbor_piece = piece_endings_map.get(tuple(neighbor_point))
                    if neighbor_piece is not None:
                        neighbor_piece_points = np.array([points[idx] for idx in neighbor_piece if idx < len(points)])

                        # Calcola growth vector per il neighbor piece
                        neighbor_growth_vector = self.vector_calculator.calculate_growth_vector(neighbor_piece_points[-window_size:])
                        neighbor_growth_vectors.append(neighbor_growth_vector)

                        # Calcola il direction vector tra l'ending corrente e il punto finale del neighbor piece
                        direction_vector = self.vector_calculator.calculate_direction_vector(current_ending_point, neighbor_point)
                        direction_vectors.append(direction_vector)

                    attractor_points[piece_id_tuple] = {
                        "growth_vector": growth_vector,
                        "neighbor_points": neighbor_points[1:],  # Escludi il primo (punto corrente)
                        "neighbor_growth_vectors": neighbor_growth_vectors,
                        "direction_vectors": direction_vectors,
                        "distances": neighbors_distances
                    }

        return attractor_points



    def get_end_points(self, neuron_data, pieces):
        """
        Estrai i punti finali di ogni pezzo.
        """
        points = neuron_data.points
        end_points = []

        for piece_id in pieces:
            piece_points = np.array([points[idx] for idx in piece_id if idx < len(points)])
            if piece_points.size > 0:
                end_points.extend(self.get_endings(piece_points))  # Punti iniziale e finale

        return np.array(end_points)

    def get_endings(self, piece_points):
        """
        Restituisce i punti finali di un pezzo (primo e ultimo).
        Se il pezzo è vuoto, restituisce una lista vuota.
        """
        if piece_points.size == 0:
            return []  # Nessun punto finale se il pezzo è vuoto
        return [piece_points[0], piece_points[-1]]

    def build_kdtree(self, points):
        return KDTree(points)

    def find_neighbors(self, point, num_neighbors, reference_points, initial_radius=60.0, growth_factor=2.0):
        """
        Trova i vicini più vicini di un punto utilizzando una ricerca tipo 'ball',
        espandendo dinamicamente il raggio fino a ottenere esattamente num_neighbors vicini.
        
        Parameters:
        - point: Il punto per cui trovare i vicini.
        - reference_points: Array di punti di riferimento su cui si basa il KDTree.
        - num_neighbors: Numero esatto di vicini richiesti.
        - initial_radius: Raggio iniziale per la ricerca dei vicini.
        - growth_factor: Fattore di crescita per il raggio se non si trovano abbastanza vicini.
        
        Returns:
        - indices: Indici dei vicini trovati nei reference_points.
        - distances: Distanze corrispondenti dal punto dato ai vicini.
        """
        # Costruzione del KDTree se non è già stato fatto
        if self.kdtree is None or self.kdtree.data.shape[0] != reference_points.shape[0]:
            self.kdtree = KDTree(reference_points)

        # Inizializzazione del raggio
        radius = initial_radius

        while True:
            # Trova i vicini entro il raggio corrente
            neighbor_indices = self.kdtree.query_ball_point(point, r=radius)

            # Se il numero di vicini trovati è sufficiente, usa query per prendere esattamente i più vicini
            if len(neighbor_indices) >= num_neighbors:
                distances, indices = self.kdtree.query(point, k=num_neighbors)
                return indices, distances

            # Espandi il raggio per includere più vicini
            radius *= growth_factor
