import numpy as np
from storage import NeuronDataStorage
from connector import NeuronConnector
from typing import List, Set
from collections import defaultdict
from functools import cached_property

class NeuronModel:
    def __init__(
        self,
        data_storage: NeuronDataStorage,
        connector: NeuronConnector,
    ):
        self.data_storage = data_storage
        self.connector = connector
        self.connected_data_storage = None

    @cached_property
    def connected_swc_samples(self) -> List[Set[int]]:
        # Raccoglie i componenti connessi dai dati SWC usando DFS iterativo.
        graph = defaultdict(list)
        roots = set()

        for entry in self.data_storage.to_swc():
            node_id, parent_id = entry[0], entry[6]
            if parent_id == -1:
                roots.add(node_id)
            else:
                graph[parent_id].append(node_id)
                graph[node_id].append(parent_id)  # Bidirezionale per efficienza

        # DFS iterativa per raccogliere i componenti connessi.
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

        # Inizia DFS da ciascun nodo radice
        for root in roots:
            if root not in visited:
                component = iterative_dfs(root)
                connected_samples.append(component)
                visited.update(component)

        # Controlla eventuali componenti scollegati
        for node in graph:
            if node not in visited:
                component = iterative_dfs(node)
                connected_samples.append(component)
                visited.update(component)

        return connected_samples

    def get_swc_chunks(self) -> list:
        # Suddivide i dati SWC in segmenti di campioni connessi
        node_to_component = {}
        for i, component in enumerate(self.connected_swc_samples):
            for node_id in component:
                node_to_component[node_id] = i

        chunked_swc = [[] for _ in range(len(self.connected_swc_samples))]

        # Itera sui dati SWC per assegnare ogni nodo al proprio componente
        for row in self.data_storage.to_swc():
            node_id = row[0]
            if node_id in node_to_component:
                component_index = node_to_component[node_id]
                chunked_swc[component_index].append(row)

        # Converte ciascun chunk in un array NumPy
        return [np.array(chunk) for chunk in chunked_swc if chunk]

    def get_points(self):
        # Restituisce i punti originali
        return self.data_storage.points[:, :3]

    def get_lines(self):
        # Restituisce le linee in base agli indici di punto iniziale e finale
        return [
            np.arange(line[1], line[1] + line[2])
            for line in self.data_storage.lines
        ]

    def connect_components(self):
        # Usa il connector per costruire la morfologia connessa
        connected_morphology = self.connector.get_connected_morphology()
        self.connected_data_storage = connected_morphology

    def get_connected_points(self):
        # Restituisce i punti dei componenti connessi
        return self.connected_data_storage.points[:, :3]

    def get_connected_lines(self):
        # Restituisce le linee dei componenti connessi
        return [
            np.arange(line[1], line[1] + line[2])
            for line in self.connected_data_storage.lines
        ]

    def get_stitch_lines(self):
        # Restituisce le linee di stitching con line[0] == 5 come da tua logica
        return [
            np.arange(line[1], line[1] + line[2])
            for line in self.connected_data_storage.lines
            if line[0] == 5
        ]
