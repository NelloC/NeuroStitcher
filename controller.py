import numpy as np
from NeuronModel import NeuronModel

class NeuronController:
    def __init__(self, model, view):
        self.NeuronModel = NeuronModel
        self.view = view

    def visualize(self):
        original_points = self.NeuronModel.get_points()
        original_lines = self.NeuronModel.get_lines()

        self.model.connect_components()

        connected_points = self.NeuronModel.get_connected_points()
        connected_lines = self.NeuronModel.get_connected_lines()
        stitch_lines = self.NeuronModel.get_stitch_lines()

        self.view.render_original_and_connected(
            original_points,
            original_lines,
            connected_points,
            connected_lines,
            stitch_lines,
        )

    def visualize_components(self):
        connected_components = self.model.connected_swc_samples
        self.view.render_connected_components(connected_components)

    def run(self):
        self.view.show()

    def close(self):
        self.view.close()

