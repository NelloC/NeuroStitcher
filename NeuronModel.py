import numpy as np
from storage import NeuronDataStorage
from connector import NeuronConnector  # Usa il tuo NeuronConnector


class NeuronModel:
    def __init__(
        self,
        data_storage: NeuronDataStorage,
        connector: NeuronConnector,
    ):
        self.data_storage = data_storage
        self.connector = connector
        self.connected_data_storage = None

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

