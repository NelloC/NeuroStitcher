import pyvista as pv
import numpy as np
import random


class NeuronView:
    def __init__(self):
        self.plotter = pv.Plotter()
        self.neuron_actor = None
        self.connected_neuron_actor = None
        self.stitch_actor = None

    def render_neuron(self, points, lines, color="black", line_width=2):
        cells = []
        for line in lines:
            cells.extend([len(line)] + line.tolist())

        mesh = pv.PolyData(points, lines=cells)
        return self.plotter.add_mesh(mesh, color=color, line_width=line_width)

    def render_original_and_connected(
        self,
        original_points,
        original_lines,
        connected_points,
        connected_lines,
        stitch_lines,
    ):
        # Clear existing actors
        self.plotter.clear()

        # Render original morphology
        self.neuron_actor = self.render_neuron(
            original_points, original_lines, color="black", line_width=2
        )

        # Render connected morphology (excluding stitches)
        self.connected_neuron_actor = self.render_neuron(
            connected_points, connected_lines, color="blue", line_width=2
        )

        # Render stitches
        self.stitch_actor = self.render_neuron(
            connected_points, stitch_lines, color="red", line_width=3
        )

    def render_connected_components(self, connected_components):
        for component in connected_components:
            # Extract coordinates and parent information
            point_ids = component[:, 0].astype(int)
            coords = component[:, 2:5].astype(np.float32)
            parent_ids = component[:, 6].astype(int)

            # Create a dictionary mapping point_ids to their index
            id_to_index = {id: index for index, id in enumerate(point_ids)}

            # Create lines and identify parent and child points
            lines = []
            has_parent = set()
            has_child = set()
            for i, parent_id in enumerate(parent_ids):
                if parent_id != -1:
                    lines.append(
                        [id_to_index[point_ids[i]], id_to_index[parent_id]]
                    )
                    has_parent.add(point_ids[i])
                    has_child.add(parent_id)

            # Convert lines to the format expected by PyVista
            cells = []
            for line in lines:
                cells.extend([2, line[0], line[1]])

            # Create the mesh for the lines
            line_mesh = pv.PolyData(coords, lines=cells)

            # Create a mesh for the points
            point_mesh = pv.PolyData(coords)

            # Add the lines to the plotter
            random_color = "%06x" % random.randint(0, 0xFFFFFF)
            self.plotter.add_mesh(line_mesh, color=random_color, line_width=2)

        # Set up the camera for an isometric view
        self.plotter.camera_position = "iso"

    def show(self):
        self.plotter.show()

    def close(self):
        self.plotter.close()

