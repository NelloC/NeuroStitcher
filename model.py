import numpy as np
from storage import NeuronDataStorage
from graph import NeuronGraphProcessor


class NeuronModel:
    def __init__(
        self,
        data_storage: NeuronDataStorage,
        graph_processor: NeuronGraphProcessor,
    ):
        self.data_storage = data_storage
        self.graph_processor = graph_processor
        self.connected_data_storage = None

    def get_points(self):
        return self.data_storage.points[:, :3]

    def get_lines(self):
        return [
            np.arange(line[1], line[1] + line[2])
            for line in self.data_storage.lines
        ]

    def connect_components(self):
        self.connected_data_storage = self.graph_processor.process()

    def get_connected_points(self):
        return self.connected_data_storage.points[:, :3]

    def get_connected_swc_components(self):
        return self.graph_processor.get_swc_chunks()

    def get_stitch_lines(self):
        return [
            np.arange(line[1], line[1] + line[2])
            for line in self.connected_data_storage.lines
            if line[0] == 5
        ]
