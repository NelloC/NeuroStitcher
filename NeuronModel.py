import numpy as np
from storage import NeuronDataStorage
from connector import NeuronConnector
from typing import List, Set
from collections import defaultdict
from functools import cached_property

class NeuronModel:
    def __init__(self, data_storage: NeuronDataStorage, connector: NeuronConnector):
        self.data_storage = data_storage
        self.connector = connector
        self.connected_data_storage = None

    @cached_property
    def connected_swc_samples(self) -> List[Set[int]]:
        """
        Trova componenti connessi utilizzando DFS iterativo.

        Returns:
            List[Set[int]]: Liste di campioni connessi.
        """
        graph = defaultdict(list)
        roots = set()

        for entry in self.data_storage.to_swc():
            node_id, parent_id = entry[0], entry[6]
            if parent_id == -1:
                roots.add(node_id)
            else:
                graph[parent_id].append(node_id)
                graph[node_id].append(parent_id)  # Connessione bidirezionale

        def iterative_dfs(start_node: int) -> Set[int]:
            stack = [start_node]
            component = set()
            while stack:
                node = stack.pop()
                if node not in component:
                    component.add(node)
                    stack.extend(neighbor for neighbor in graph[node] if neighbor not in component)
            return component

        connected_samples = []
        visited = set()

        for root in roots:
            if root not in visited:
                component = iterative_dfs(root)
                connected_samples.append(component)
                visited.update(component)

        return connected_samples

    def get_swc_chunks(self) -> list:
        """
        Suddivide i dati SWC in pezzi.

        Returns:
            list: Lista di array SWC.
        """
        node_to_component = {}
        for i, component in enumerate(self.connected_swc_samples):
            for node_id in component:
                node_to_component[node_id] = i

        chunked_swc = [[] for _ in range(len(self.connected_swc_samples))]

        for row in self.data_storage.to_swc():
            node_id = row[0]
            if node_id in node_to_component:
                component_index = node_to_component[node_id]
                chunked_swc[component_index].append(row)

        return [np.array(chunk) for chunk in chunked_swc if chunk]

    def get_points(self):
        """
        Restituisce i punti originali.

        Returns:
            np.ndarray: Punti 3D.
        """
        return self.data_storage.points[:, :3]

    def get_lines(self):
        """
        Restituisce linee valide.

        Returns:
            list: Array di indici linea.
        """
        valid_lines = []
        for line in self.data_storage.lines:
            if len(line) >= 3:
                valid_lines.append(np.arange(line[1], line[1] + line[2]))
            else:
                print(f"[DEBUG] Linea incompleta ignorata: {line}")
        return valid_lines

    def connect_components(self, stitches):
        """
        Collega componenti con le linee di stitching.
        """
        self.connector.connect_components(stitches)
        self.connected_data_storage = self.connector.get_connected_data_storage()

    def get_connected_points(self):
        """
        Punti dei componenti connessi.
        """
        return self.connected_data_storage.points[:, :3]

    def get_connected_lines(self):
        """
        Linee dai componenti connessi.
        """
        return [
            np.arange(line[1], line[1] + line[2])
            for line in self.connected_data_storage.lines
            if len(line) >= 3
        ]

    def get_stitch_lines(self):
        """
        Linee di stitching accettate.
        """
        return [
            np.arange(line[1], line[1] + line[2])
            for line in self.connected_data_storage.lines
            if line[0] == 5  # Identifica stitching
        ]
