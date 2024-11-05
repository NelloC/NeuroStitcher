import numpy as np
from NeuronModel import NeuronModel

class NeuronController:
    def __init__(self, model: NeuronModel, view):
        self.model = model  # Qui assegniamo l'istanza di NeuronModel
        self.view = view

    def visualize(self):
        # Ottiene i punti e le linee originali
        original_points = self.model.get_points()
        original_lines = self.model.get_lines()

        # Esegue il collegamento dei componenti
        self.model.connect_components()

        # Ottiene i punti e le linee connessi
        connected_points = self.model.get_connected_points()
        connected_lines = self.model.get_connected_lines()
        stitch_lines = self.model.get_stitch_lines()

        # Visualizza i dati originali e connessi
        self.view.render_original_and_connected(
            original_points,
            original_lines,
            connected_points,
            connected_lines,
            stitch_lines,
        )

    def visualize_components(self):
        # Converti ogni componente in un array con tutti i dati SWC necessari
        connected_components = self.model.get_swc_chunks()
        self.view.render_connected_components(connected_components)


    def run(self):
        # Avvia la visualizzazione
        self.view.show()

    def close(self):
        # Chiude la visualizzazione
        self.view.close()


