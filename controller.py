import numpy as np


class NeuronController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def visualize(self):
        original_points = self.model.get_points()
        original_lines = self.model.get_lines()

        self.model.connect_components()

        connected_points = self.model.get_connected_points()
        connected_lines = self.model.get_connected_lines()
        stitch_lines = self.model.get_stitch_lines()

        self.view.render_original_and_connected(
            original_points,
            original_lines,
            connected_points,
            connected_lines,
            stitch_lines,
        )

    def visualize_components(self):
        connected_components = self.model.get_connected_swc_components()
        self.view.render_connected_components(connected_components)

    def run(self):
        self.view.show()

    def close(self):
        self.view.close()

