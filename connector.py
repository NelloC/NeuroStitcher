import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from morphology import NeuronMorphology


class NeuronConnector:
    def __init__(self, neuron_morphology: NeuronMorphology):
        self.neuron = neuron_morphology
        self.medial_trees = None
        self.k_nn_graph = None
        self.connected_tree = None
        self.component_endpoints = None

    def connect_components(self, k=10, gamma=2 / 3):
        self.find_medial_trees()
        self.construct_k_nn_graph(k, gamma)
        self.find_minimum_spanning_tree()
        self.ensure_full_connectivity()
        print(f"Number of medial trees: {len(self.medial_trees)}")
        print(f"K-NN graph shape: {self.k_nn_graph.shape}")
        print(f"Connected tree shape: {self.connected_tree.shape}")
        n_connected, labels = connected_components(
            csr_matrix(self.connected_tree)
        )
        print(f"Number of connected components after MST: {n_connected}")
        print(f"Unique labels: {np.unique(labels)}")

    def find_medial_trees(self):
        self.medial_trees = self.neuron.get_pieces_of_neurite()
        self.component_endpoints = [
            self.get_endpoints(line_id) for line_id in self.medial_trees.keys()
        ]

    def construct_k_nn_graph(self, k, gamma):
        n_components = len(self.medial_trees)
        all_endpoints = np.vstack(self.component_endpoints)
        component_indices = np.repeat(np.arange(n_components), 2)

        tree = cKDTree(all_endpoints)

        self.k_nn_graph = np.full((n_components, n_components), np.inf)

        for i in range(n_components):
            # Query more neighbors than k to ensure connectivity
            distances, indices = tree.query(
                all_endpoints[i * 2], k=n_components
            )

            connected = False
            for j, dist in zip(indices, distances):
                if component_indices[j] != i:
                    comp_j = component_indices[j]
                    alignment_cost = self.calculate_alignment_cost(
                        all_endpoints[i * 2], all_endpoints[j]
                    )
                    total_cost = gamma * dist + (1 - gamma) * alignment_cost
                    self.k_nn_graph[i, comp_j] = min(
                        self.k_nn_graph[i, comp_j], total_cost
                    )
                    connected = True
                    if np.sum(np.isfinite(self.k_nn_graph[i])) >= k:
                        break

            # If still not connected, connect to the closest component
            if not connected:
                closest = np.argmin(distances[component_indices[indices] != i])
                comp_closest = component_indices[indices[closest]]
                self.k_nn_graph[i, comp_closest] = distances[closest]

        # Ensure symmetry
        self.k_nn_graph = np.minimum(self.k_nn_graph, self.k_nn_graph.T)

        # Ensure connectivity
        self.ensure_graph_connectivity()

    def ensure_graph_connectivity(self):
        n_components = self.k_nn_graph.shape[0]
        connected_components = np.zeros(n_components, dtype=int)
        component_count = 0

        for i in range(n_components):
            if connected_components[i] == 0:
                component_count += 1
                self.dfs(i, component_count, connected_components)

        while component_count > 1:
            min_distance = np.inf
            min_i, min_j = -1, -1
            for i in range(n_components):
                for j in range(i + 1, n_components):
                    if connected_components[i] != connected_components[j]:
                        dist = np.min(
                            [
                                np.linalg.norm(ei - ej)
                                for ei in self.component_endpoints[i]
                                for ej in self.component_endpoints[j]
                            ]
                        )
                        if dist < min_distance:
                            min_distance = dist
                            min_i, min_j = i, j

            self.k_nn_graph[min_i, min_j] = self.k_nn_graph[min_j, min_i] = (
                min_distance
            )
            old_component = connected_components[min_j]
            new_component = connected_components[min_i]
            connected_components[connected_components == old_component] = (
                new_component
            )
            component_count -= 1

    def dfs(self, node, component, connected_components):
        connected_components[node] = component
        for neighbor in np.where(np.isfinite(self.k_nn_graph[node]))[0]:
            if connected_components[neighbor] == 0:
                self.dfs(neighbor, component, connected_components)

    def find_minimum_spanning_tree(self):
        graph = csr_matrix(self.k_nn_graph)
        self.connected_tree = minimum_spanning_tree(graph)

    def ensure_full_connectivity(self):
        n_components = self.k_nn_graph.shape[0]
        n_connected, labels = connected_components(
            csr_matrix(self.connected_tree)
        )

        if n_connected > 1:
            print(
                f"Found {n_connected} disconnected components. Connecting them..."
            )
            for i in range(1, n_connected):
                comp_i = np.where(labels == 0)[0][0]
                comp_j = np.where(labels == i)[0][0]
                min_dist = np.inf
                for ci in np.where(labels == 0)[0]:
                    for cj in np.where(labels == i)[0]:
                        dist = np.min(
                            [
                                np.linalg.norm(ei - ej)
                                for ei in self.component_endpoints[ci]
                                for ej in self.component_endpoints[cj]
                            ]
                        )
                        if dist < min_dist:
                            min_dist = dist
                            comp_i, comp_j = ci, cj
                self.connected_tree[comp_i, comp_j] = self.connected_tree[
                    comp_j, comp_i
                ] = min_dist
                labels[labels == i] = 0
            print("All components are now connected.")
        else:
            print(
                "All components are already connected according to the minimum spanning tree."
            )

    def get_endpoints(self, line_id):
        line = self.neuron.lines[line_id]
        return [
            self.neuron.points[line[1]][:3],
            self.neuron.points[line[1] + line[2] - 1][:3],
        ]

    def calculate_alignment_cost(self, point1, point2):
        # Simplified alignment cost calculation
        return np.linalg.norm(point1 - point2)

    def get_connected_morphology(self):
        new_lines = self.neuron.lines.copy()
        new_points = self.neuron.points.copy()

        component_to_line = {
            i: line_id
            for i, (line_id, _) in enumerate(self.medial_trees.items())
        }

        for i, j in zip(*self.connected_tree.nonzero()):
            if i < j:  # Avoid duplicate connections
                line_id_i = component_to_line[i]
                line_id_j = component_to_line[j]

                endpoints_i = self.component_endpoints[i]
                endpoints_j = self.component_endpoints[j]

                # Find the closest pair of endpoints between the two components
                distances = [
                    np.linalg.norm(ei - ej)
                    for ei in endpoints_i
                    for ej in endpoints_j
                ]
                min_distance_idx = np.argmin(distances)
                start_point = endpoints_i[min_distance_idx // 2]
                end_point = endpoints_j[min_distance_idx % 2]

                # Create two new points for the stitch line
                new_point_idx1 = len(new_points)
                new_points = np.vstack((new_points, np.append(start_point, 1)))

                new_point_idx2 = len(new_points)
                new_points = np.vstack((new_points, np.append(end_point, 1)))

                # Create a new line to connect the components
                new_line = [5, new_point_idx1, 2, line_id_i, 0]
                new_lines.append(new_line)

                # Update the parent of line_j to be the new connecting line
                new_lines[line_id_j][3] = len(new_lines) - 1

        new_neuron_dict = self.neuron.to_dict()
        new_neuron_dict["treeLines"]["data"] = new_lines
        new_neuron_dict["treePoints"]["data"] = new_points.tolist()

        return NeuronMorphology(
            new_neuron_dict, self.neuron.unit_orientation_origin
        )

    def visualize_graph(self, graph, title):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Plot all endpoints
        all_endpoints = np.vstack(self.component_endpoints)
        ax.scatter(
            all_endpoints[:, 0],
            all_endpoints[:, 1],
            all_endpoints[:, 2],
            c="blue",
            s=20,
        )

        # Plot edges
        for i in range(graph.shape[0]):
            for j in range(i + 1, graph.shape[1]):
                if graph[i, j] != 0 and not np.isinf(graph[i, j]):
                    start = self.component_endpoints[i][
                        0
                    ]  # Using the first endpoint of each component
                    end = self.component_endpoints[j][0]
                    ax.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        c="red",
                        alpha=0.5,
                    )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def visualize_knn_graph(self):
        self.visualize_graph(self.k_nn_graph, "k-NN Graph")

    def visualize_minimum_spanning_tree(self):
        mst_matrix = self.connected_tree.toarray()
        self.visualize_graph(mst_matrix, "Minimum Spanning Tree")
