from typing import Dict, List, Any, Set
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from collections import defaultdict
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from functools import cached_property
from storage import NeuronDataStorage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedialTree:
    def __init__(self, component_data):
        self.data = component_data

    @cached_property
    def center_of_mass(self):
        return np.average(self.data[:, 2:5], axis=0)

    @cached_property
    def terminal_samples(self):
        node_ids = set(self.data[:, 0])
        parent_ids = set(self.data[:, 6])

        root_node = list(node_ids)[parent_ids == -1]
        leaf_nodes = node_ids - (parent_ids - {-1})

        # Combine root and leaf nodes
        terminal_nodes = set([root_node]).union(leaf_nodes)
        # Create a boolean mask for terminal nodes
        terminal_mask = np.isin(self.data[:, 0], list(terminal_nodes))

        # Return the terminal points
        return self.data[terminal_mask]

    @property
    def terminal_points(self):
        return (self.terminal_samples[:, 2:5]).astype(float)

    @cached_property
    def terminal_tangents(self):
        tangents = []
        # Compute vectors for each terminal point
        for ts in self.terminal_samples:
            # If it's root point, consider its child as ancestor
            if ts[6] == -1:
                ancestor = self.data[self.data[:, 6] == ts[0]][0]
            else:
                ancestor = self.data[self.data[:, 0] == ts[6]][0]
            tangents.append((ts[2:5] - ancestor[2:5]).astype(float))

        return tangents


class NeuronGraphProcessor:
    def __init__(
        self, data_storage: NeuronDataStorage, gamma: float = 0.5, k: int = 10
    ):
        self.data = data_storage
        self.gamma = gamma
        self.k = k
        self.swc_data = self.data.to_swc()
        self.medial_trees: List[MedialTree] = []
        self.kdtree: cKDTree = None
        self.k_nn_graph: np.ndarray = None
        self.mst: csr_matrix = None

    @cached_property
    def connected_swc_samples(self) -> List[Set[int]]:
        # Efficiently collect connected components from SWC data using iterative DFS.
        # Create an adjacency list representation of the graph
        graph = defaultdict(list)
        roots = set()

        for entry in self.swc_data:
            node_id, parent_id = entry[0], entry[6]
            if parent_id == -1:
                roots.add(node_id)
            else:
                graph[parent_id].append(node_id)
                graph[node_id].append(parent_id)  # Bidirectional for efficiency

        # Iterative depth-first search to collect connected components.
        def iterative_dfs(start_node: int) -> Set[int]:
            stack = [start_node]
            component = set()

            while stack:
                node = stack.pop()
                if node not in component:
                    component.add(node)
                    stack.extend(
                        neighbor
                        for neighbor in graph[node]
                        if neighbor not in component
                    )

            return component

        connected_samples = []
        visited = set()

        # Start DFS from each root node
        for root in roots:
            if root not in visited:
                component = iterative_dfs(root)
                connected_samples.append(component)
                visited.update(component)

        # Check for any disconnected components not reached from roots
        for node in graph:
            if node not in visited:
                component = iterative_dfs(node)
                connected_samples.append(component)
                visited.update(component)

        return connected_samples

    def get_swc_chunks(self) -> List[np.ndarray]:
        # Chunk the whole SWC data to a list of SWC segments of connected samples
        # Create a mapping from node ID to component index
        node_to_component = {}
        for i, component in enumerate(self.connected_swc_samples):
            for node_id in component:
                node_to_component[node_id] = i

        # Pre-allocate lists for each component
        chunked_swc = [[] for _ in range(len(self.connected_swc_samples))]

        # Single pass through the SWC data
        for row in self.swc_data:
            node_id = row[0]
            if node_id in node_to_component:
                component_index = node_to_component[node_id]
                chunked_swc[component_index].append(row)

        # Convert lists to numpy arrays
        return [np.array(chunk) for chunk in chunked_swc if chunk]

    def _collect_components(self):
        self.medial_trees = [
            MedialTree(swc_chunk) for swc_chunk in self.get_swc_chunks()
        ]

    def _build_kdtree(self) -> None:
        # Collect the components if they hadn't been collected
        if len(self.medial_trees) == 0:
            self._collect_components()
        # Initialise the KDTree with their center of mass
        medial_tree_centers = np.array(
            [mt.center_of_mass for mt in self.medial_trees]
        )
        self.kdtree = cKDTree(medial_tree_centers)

    # For each component, compute the costs associated to all the possibile combination of its leaf nodes
    # Start with simple distance computation first
    def compute_connections_costs(self):
        connection_costs = {}

        if self.kdtree is None:
            self._build_kdtree()

        for i, mt in enumerate(self.medial_trees):
            mt_connection_costs = {}
            print(f"Computing distances for medial tree n {i}")
            medial_tree_leaves = mt.terminal_points
            _, k_nearest_trees = self.kdtree.query(mt.center_of_mass, self.k)
            for nearest_tree_id in k_nearest_trees:
                # Skip self connections
                if nearest_tree_id == i:
                    continue
                nearest_tree = self.medial_trees[nearest_tree_id]
                # Compute the distances among all terminal points using numpy broadcasting
                medial_tree_leaves = medial_tree_leaves[:, np.newaxis, :]
                nearest_tree_leaves = nearest_tree.terminal_points[
                    np.newaxis, :, :
                ]
                # Compute the difference
                diff = medial_tree_leaves - nearest_tree_leaves
                # Compute the squared distances
                squared_distances = np.sum(diff**2, axis=2)
                # Compute the Euclidean distances
                distances = np.sqrt(squared_distances)
                mt_connection_costs[nearest_tree_id] = np.min(distances)
                # print(f"Distance {i}-{nearest_tree_id}: {np.min(distances)}")
            connection_costs[i] = mt_connection_costs

        return connection_costs

    def construct_k_nn_graph(self) -> None:
        n_components = len(self.medial_trees)
        self.k_nn_graph = np.full((n_components, n_components), np.inf)

        # Get the connection costs dictionary to populate the matrix
        edges = self.compute_connections_costs()

        for i, dist_dict in edges.items():
            for j, v in dist_dict.items():
                self.k_nn_graph[i][j] = v

    """
    def _compute_alignment_cost(
        self, tangent1: np.ndarray, tangent2: np.ndarray, direction: np.ndarray
    ) -> float:
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 0:
            direction /= direction_norm
        else:
            return np.pi  # Maximum misalignment if points are identical

        phi = self._compute_rotation_angle(tangent1, direction)
        theta = self._compute_rotation_angle(tangent2, -direction)

        return abs(phi) + abs(theta)

    @staticmethod
    def _compute_rotation_angle(tangent: np.ndarray, axis: np.ndarray) -> float:
        dot_product = np.clip(
            np.dot(tangent, axis), -1.0, 1.0
        )  # Clip to avoid numerical instability
        return np.arccos(dot_product)
    """

    def _compute_mst(self) -> None:
        graph = csr_matrix(self.k_nn_graph)
        self.mst = minimum_spanning_tree(graph)

    def _connect_components(self) -> np.ndarray:
        # if self.mst is None:
        #    self._compute_mst()

        connected_swc = self.swc_data.copy()
        # mst_coo = self.mst.tocoo()

        cc = self.compute_connections_costs()

        new_node_id = int(np.max(self.swc_data[:, 0])) + 1
        # for i, j in zip(mst_coo.row, mst_coo.col):
        #    if i < j:  # Avoid duplicate connections
        for i in cc.keys():
            # Find the best pair of terminal nodes to connect
            best_cost = np.inf
            best_pair = None

            mt_i = self.medial_trees[i]

            for j in cc[i].keys():

                mt_j = self.medial_trees[j]

                for node_i in mt_i.terminal_samples:
                    for node_j in mt_j.terminal_samples:
                        point_i = node_i[2:5]
                        point_j = node_j[2:5]
                        # tangent_i = self.terminal_nodes[node_i]["tangent"]
                        # tangent_j = self.terminal_nodes[node_j]["tangent"]

                        distance = np.linalg.norm(point_i - point_j)
                        # alignment_cost = self._compute_alignment_cost(
                        #    tangent_i, tangent_j, point_j - point_i
                        # )
                        total_cost = distance
                        # (
                        #    self.gamma * distance
                        #    + (1 - self.gamma) * alignment_cost
                        # )

                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_pair = (node_i, node_j)

            if best_pair:
                node_i, node_j = best_pair
                point_i = node_i[2:5]
                point_j = node_j[2:5]
                mid_point = (point_i + point_j) / 2
                new_node = np.array([new_node_id, 5, *mid_point, 1, node_i[0]])
                connected_swc = np.vstack((connected_swc, new_node))

                # Update the parent of node_j to be the new node
                idx_j = np.where(connected_swc[:, 0] == node_j[0])[0][0]
                connected_swc[idx_j, 6] = new_node_id

                new_node_id += 1
                logger.info(
                    f"Connected components {i} and {j} with cost {best_cost}"
                )
            else:
                logger.warning(
                    f"Could not find suitable terminal nodes to connect components {i} and {j}"
                )

        return connected_swc

    """
    def get_connected_morphology(self, connected_lines) -> NeuronDataStorage:

        new_lines = connected_lines  # .tolist()
        new_points = self.data.points.tolist()

        mst_coo = self.mst.tocoo()
        for i, j, _ in zip(mst_coo.row, mst_coo.col, mst_coo.data):
            if i < j:  # Avoid duplicate connections
                line_id_i = i
                line_id_j = j

                # Get endpoints for both lines
                endpoints_i = [
                    self.data.points[self.data.lines[line_id_i][1]][:3],
                    self.data.points[
                        self.data.lines[line_id_i][1]
                        + self.data.lines[line_id_i][2]
                        - 1
                    ][:3],
                ]
                endpoints_j = [
                    self.data.points[self.data.lines[line_id_j][1]][:3],
                    self.data.points[
                        self.data.lines[line_id_j][1]
                        + self.data.lines[line_id_j][2]
                        - 1
                    ][:3],
                ]

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
                new_points.append(np.append(start_point, 1).tolist())

                new_point_idx2 = len(new_points)
                new_points.append(np.append(end_point, 1).tolist())

                # Create a new line to connect the components
                new_line = [
                    5,
                    new_point_idx1,
                    2,
                    line_id_i + 1,
                    0,
                ]  # +1 because line indices start from 1
                new_lines.append(new_line)

                # Update the parent of line_j to be the new connecting line
                new_lines[line_id_j][3] = len(
                    new_lines
                )  # No need for -1 as we're using 1-based indexing for lines

        new_neuron_dict = self.data.to_dict()
        new_neuron_dict["treeLines"]["data"] = np.array(new_lines).reshape(
            -1, 5
        )
        new_neuron_dict["treePoints"]["data"] = new_points

        return NeuronDataStorage(
            new_neuron_dict, self.data.unit_orientation_origin
        )
        """
