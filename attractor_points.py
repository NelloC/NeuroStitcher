import numpy as np
from scipy.spatial import KDTree
from storage import NeuronDataStorage  # Importa la classe NeuronDataStorage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class AttractorPointsManager:
    def __init__(self, vector_calculator):
        self.vector_calculator = vector_calculator
        self.kdtree = None

    def calculate_attractor_points(self, neuron_data, pieces, clustering_manager, window_size=7):
        points = neuron_data.points[:, :3]  # Utilizza solo le prime 3 coordinate
        attractor_points = {}

        # Estrai i punti finali di tutti i pezzi
        end_points = self.get_end_points(neuron_data, pieces)[:, :3]

        self.kdtree = self.build_kdtree(end_points)

        labels, unique_labels, counts, tipo_struttura = clustering_manager.run_dbscan(points)

        num_neighbors, distance_factor = clustering_manager.get_adaptive_params(
            points, np.mean(counts[counts > 1]) if len(counts[counts > 1]) > 0 else 0, labels, unique_labels, counts, tipo_struttura
        )

        piece_endings_map = {
            tuple(ending[:3]): piece  # Assicuriamoci di mappare solo le prime 3 coordinate
            for piece in pieces
            for ending in self.get_endings(np.array([points[idx] for idx in piece if idx < len(points)]))
            if len(ending) > 0
        }

        for piece_id in pieces:
            piece_id_tuple = tuple(piece_id)
            piece_points = np.array([points[idx] for idx in piece_id if idx < len(points)])

            if piece_points.size == 0:
                continue

            ending_points = self.get_endings(piece_points)
            for current_ending_point in ending_points:
                growth_vector = self.vector_calculator.calculate_growth_vector(piece_points[-window_size:])
                neighbors_indices, neighbors_distances = self.find_neighbors(current_ending_point, num_neighbors, end_points)

                neighbor_points = end_points[neighbors_indices]
                #self.visualize_neighbors(end_points, current_ending_point, neighbor_points, radius=neighbors_distances.max())

                neighbor_growth_vectors = []
                direction_vectors = []
                #print(f"[DEBUG] Piece {piece_id}: Neighbor Points - {neighbor_points}")

                for neighbor_point in neighbor_points:
                    neighbor_piece = self.find_closest_piece_end(neighbor_point, piece_endings_map)
                    if neighbor_piece is None:
                        continue

                    #print(f"[DEBUG] Neighbor point {neighbor_point} found in piece_endings_map.")  # Debug 1

                    neighbor_piece_points = np.array([points[idx] for idx in neighbor_piece if idx < len(points)])
                    neighbor_growth_vector = self.vector_calculator.calculate_growth_vector(neighbor_piece_points[-window_size:])
                    direction_vector = self.vector_calculator.calculate_direction_vector(current_ending_point, neighbor_point)

                    neighbor_growth_vectors.append(neighbor_growth_vector)
                    direction_vectors.append(direction_vector)
                    #self.visualize_vectors(current_ending_point, growth_vector, neighbor_points, neighbor_growth_vectors, direction_vectors)

                    #print(f"[DEBUG] Neighbor: Point {neighbor_point}, Growth Vector: {neighbor_growth_vector}, Direction Vector: {direction_vector}")  # Debug 2

                attractor_points[piece_id_tuple] = {
                    "growth_vector": growth_vector,
                    "neighbor_points": neighbor_points[1:],  # Escludi il primo (punto corrente)
                    "neighbor_growth_vectors": neighbor_growth_vectors,
                    "direction_vectors": direction_vectors,
                    "distances": neighbors_distances
                }

        return attractor_points

    def get_end_points(self, neuron_data, pieces):
        points = neuron_data.points
        lines = neuron_data.lines
        end_points = []

        for piece_id in pieces:
            
            for line_id in piece_id:
                if line_id < 0 or line_id >= len(lines):
                    continue
                line = lines[line_id]
                starting_point = points[line[1]][:3]
                ending_point = points[line[1] + line[2] - 1][:3]
                end_points.append(starting_point)
                end_points.append(ending_point)
                #print(f"[DEBUG] Line {line_id}: Start {starting_point}, End {ending_point}")  # Debug
   
        return np.array(end_points)

    def get_endings(self, piece_points):
        piece_points = np.array(piece_points)
        if piece_points.size == 0:
            return []
        return [piece_points[0], piece_points[-1]]

    def build_kdtree(self, points):
        return KDTree(points)

    def find_neighbors(self, point, num_neighbors, reference_points, initial_radius=1.0, growth_factor=2.0):
        if self.kdtree is None or self.kdtree.data.shape[0] != reference_points.shape[0]:
            self.kdtree = KDTree(reference_points[:, :3])

        point = point[:3]  # Considera solo le prime 3 dimensioni
        radius = initial_radius

        while True:
            neighbor_indices = self.kdtree.query_ball_point(point, r=radius)
            if len(neighbor_indices) >= num_neighbors:
                distances, indices = self.kdtree.query(point, k=num_neighbors)
                return indices, distances

            radius *= growth_factor

    def find_closest_piece_end(self, neighbor_point, piece_endings_map, tol=1e-3):
        for key, piece in piece_endings_map.items():
            if np.allclose(neighbor_point, np.array(key), atol=tol):
                return piece
        return None



    def visualize_neighbors(self, end_points, current_ending_point, neighbor_points, radius):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all end points
        ax.scatter(end_points[:, 0], end_points[:, 1], end_points[:, 2], c='gray', label='All End Points')

        # Plot current ending point
        ax.scatter(current_ending_point[0], current_ending_point[1], current_ending_point[2], c='red', s=100, label='Current Ending')

        # Plot neighbor points
        ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], c='blue', s=50, label='Neighbor Points')

        # Draw a sphere around the current point to visualize the search radius
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = radius * np.cos(u) * np.sin(v) + current_ending_point[0]
        y = radius * np.sin(u) * np.sin(v) + current_ending_point[1]
        z = radius * np.cos(v) + current_ending_point[2]
        ax.plot_wireframe(x, y, z, color="green", alpha=0.3, label='Search Radius')

        ax.legend()
        plt.show()

    def visualize_vectors(self, current_point, growth_vector, neighbor_points, neighbor_growth_vectors, direction_vectors):
        """
        Visualizza i vettori di crescita e direzione per il punto corrente e i suoi vicini.

        :param current_point: Punto di partenza per i vettori.
        :param growth_vector: Vettore di crescita del punto corrente.
        :param neighbor_points: Punti vicini.
        :param neighbor_growth_vectors: Vettori di crescita dei vicini.
        :param direction_vectors: Vettori direzionali dai vicini al punto corrente.
        """
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Visualizza il punto corrente
        ax.scatter(current_point[0], current_point[1], current_point[2], c='red', s=100, label='Current Point')

        # Visualizza i punti vicini
        ax.scatter(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2], c='blue', s=50, label='Neighbor Points')

        # Visualizza il vettore di crescita del punto corrente
        ax.quiver(
            current_point[0], current_point[1], current_point[2],
            growth_vector[0], growth_vector[1], growth_vector[2],
            color='green', label='Growth Vector', arrow_length_ratio=0.2
        )

        # Verifica la lunghezza di neighbor_growth_vectors e direction_vectors
        min_length = min(len(neighbor_points), len(neighbor_growth_vectors), len(direction_vectors))

        # Visualizza i vettori dei vicini
        for i in range(min_length):
            neighbor = neighbor_points[i]
            ax.quiver(
                neighbor[0], neighbor[1], neighbor[2],
                neighbor_growth_vectors[i][0], neighbor_growth_vectors[i][1], neighbor_growth_vectors[i][2],
                color='purple', label='Neighbor Growth Vector' if i == 0 else "", arrow_length_ratio=0.2
            )

            # Visualizza il vettore direzionale dal vicino al punto corrente
            ax.quiver(
                neighbor[0], neighbor[1], neighbor[2],
                direction_vectors[i][0], direction_vectors[i][1], direction_vectors[i][2],
                color='orange', label='Direction Vector' if i == 0 else "", arrow_length_ratio=0.2
            )

        ax.legend()
        plt.show()

