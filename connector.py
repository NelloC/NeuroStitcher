import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from vector_calculations import VectorCalculator
from clustering import ClusteringManager
from storage import NeuronDataStorage

class NeuronConnector:
    def __init__(self, neuron_data_storage: NeuronDataStorage, vector_calculator: VectorCalculator, clustering_manager: ClusteringManager):
        self.neuron = neuron_data_storage
        self.vector_calculator = vector_calculator
        self.clustering_manager = clustering_manager
        self.k_nn_graph = None
        self.connected_tree = None

    def connect_components(self, k=10, gamma=2 / 3):
        component_endpoints = self.find_component_endpoints()
        self.construct_k_nn_graph(component_endpoints, k, gamma)
        self.find_minimum_spanning_tree()
        self.ensure_full_connectivity(component_endpoints)

    def find_component_endpoints(self):
        endpoints = []
        for line in self.neuron.lines:
            start_point = self.neuron.points[line[1]]
            end_point = self.neuron.points[line[1] + line[2] - 1]
            endpoints.append((start_point[:3], end_point[:3]))
        return endpoints

    def construct_k_nn_graph(self, component_endpoints, k, gamma):
        n_components = len(component_endpoints)
        all_endpoints = np.vstack([ep for pair in component_endpoints for ep in pair])
        tree = cKDTree(all_endpoints)
        self.k_nn_graph = np.full((n_components, n_components), np.inf)

        for i in range(n_components):
            distances, indices = tree.query(all_endpoints[i * 2], k=k)
            for j, dist in zip(indices, distances):
                if j // 2 != i:
                    comp_j = j // 2
                    alignment_cost = self.vector_calculator.calculate_alignment_cost(
                        all_endpoints[i * 2], all_endpoints[j]
                    )
                    total_cost = gamma * dist + (1 - gamma) * alignment_cost
                    self.k_nn_graph[i, comp_j] = min(self.k_nn_graph[i, comp_j], total_cost)

        self.k_nn_graph = np.minimum(self.k_nn_graph, self.k_nn_graph.T)

    def find_minimum_spanning_tree(self):
        graph = csr_matrix(self.k_nn_graph)
        self.connected_tree = minimum_spanning_tree(graph)

    def ensure_full_connectivity(self, component_endpoints):
        n_components = self.k_nn_graph.shape[0]
        n_connected, labels = connected_components(csr_matrix(self.connected_tree))

        if n_connected > 1:
            for i in range(1, n_connected):
                comp_i = np.where(labels == 0)[0][0]
                comp_j = np.where(labels == i)[0][0]
                min_dist = np.inf
                for ci in np.where(labels == 0)[0]:
                    for cj in np.where(labels == i)[0]:
                        dist = np.linalg.norm(
                            np.array(component_endpoints[ci][0]) - np.array(component_endpoints[cj][1])
                        )
                        if dist < min_dist:
                            min_dist = dist
                            comp_i, comp_j = ci, cj
                self.connected_tree[comp_i, comp_j] = self.connected_tree[comp_j, comp_i] = min_dist

    def get_connected_data_storage(self):
        new_lines = self.neuron.lines.copy()
        new_points = self.neuron.points.copy()

        for i, j in zip(*self.connected_tree.nonzero()):
            if i < j:
                start_point, end_point = self.find_closest_endpoints(i, j)
                new_point_idx1 = len(new_points)
                new_points = np.vstack((new_points, np.append(start_point, 1)))
                new_point_idx2 = len(new_points)
                new_points = np.vstack((new_points, np.append(end_point, 1)))
                new_lines.append([5, new_point_idx1, 2, i, 0])

        new_neuron_dict = self.neuron.to_dict()
        new_neuron_dict["treeLines"]["data"] = new_lines
        new_neuron_dict["treePoints"]["data"] = new_points.tolist()

        return NeuronDataStorage(new_neuron_dict, self.neuron.unit_orientation_origin)

    def find_closest_endpoints(self, i, j):
        start_point = self.neuron.points[self.neuron.lines[i][1]][:3]
        end_point = self.neuron.points[self.neuron.lines[j][1] + self.neuron.lines[j][2] - 1][:3]
        return start_point, end_point

