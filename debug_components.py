import numpy as np
import pyvista as pv
from storage import NeuronDataStorage
from loaders import Loader
from graph import NeuronGraphProcessor
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot(matrices, points, tangents):
    plotter = pv.Plotter()

    for matrix, pts, tans in zip(matrices, points, tangents):
        # Extract coordinates and parent information
        point_ids = matrix[:, 0].astype(int)
        coords = matrix[:, 2:5].astype(np.float32)
        parent_ids = matrix[:, 6].astype(int)

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
        point_mesh = pv.PolyData(pts)

        # Add the meshes to the plotter
        random_color = "%06x" % random.randint(0, 0xFFFFFF)
        plotter.add_mesh(line_mesh, color=random_color, line_width=2)
        plotter.add_mesh(point_mesh, color=random_color)
        for pt, tan in zip(pts, tans):
            arrow = pv.Arrow(
                start=pt,
                direction=tan * 10,
                tip_length=0.1,
                tip_radius=0.05,
                tip_resolution=20,
                shaft_radius=0.025,
                shaft_resolution=20,
                scale=None,
            )
            plotter.add_mesh(arrow, color=random_color)

    # Set up the camera for an isometric view
    plotter.camera_position = "iso"

    # Show the plot
    plotter.show()


def render_morphology_3d(matrix):
    # Extract coordinates and parent information
    point_ids = matrix[:, 0].astype(int)
    coords = matrix[:, 2:5].astype(np.float32)
    parent_ids = matrix[:, 6].astype(int)

    # Create a dictionary mapping point_ids to their index
    id_to_index = {id: index for index, id in enumerate(point_ids)}

    # Create lines and identify parent and child points
    lines = []
    has_parent = set()
    has_child = set()
    for i, parent_id in enumerate(parent_ids):
        if parent_id != -1:
            lines.append([id_to_index[point_ids[i]], id_to_index[parent_id]])
            has_parent.add(point_ids[i])
            has_child.add(parent_id)

    # Convert lines to the format expected by PyVista
    cells = []
    for line in lines:
        cells.extend([2, line[0], line[1]])

    # Create the mesh for the lines
    line_mesh = pv.PolyData(coords, lines=cells)

    # Create a plotter
    plotter = pv.Plotter()

    # Add the lines to the plotter
    plotter.add_mesh(line_mesh, color="black", line_width=2)

    # Set up the camera for an isometric view
    plotter.camera_position = "iso"

    # Show the plot
    plotter.show()


def visualize_graph(nodes, connections):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot all endpoints
    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c="blue", s=20)

    # Plot edges
    for [i, j] in connections:
        ax.plot(
            [nodes[i, 0], nodes[j, 0]],
            [nodes[i, 1], nodes[j, 1]],
            [nodes[i, 2], nodes[j, 2]],
            c="red",
            alpha=0.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "data/ms1821_alignment_soloVPM_nocontours_neuritescorrected_aligned.json"
    unit_orientation_origin = ["um", "RAS", "corner"]
    loader = Loader()

    # Load the initial neuron data
    neuron_data = loader.load_morphology_from_file(
        file_path, unit_orientation_origin=unit_orientation_origin
    )

    # matrix = neuron_data.to_swc()
    graph_processor = NeuronGraphProcessor(neuron_data)

    swc = neuron_data.to_swc()
    cc = graph_processor.get_swc_chunks()
    graph_processor._collect_components()
    mt = graph_processor.medial_trees
    coms = [(t.terminal_points).astype(float) for t in mt]
    tans = [t.terminal_tangents for t in mt]

    graph_processor.construct_k_nn_graph()
    # valid_connections = ~np.isinf(graph_processor.k_nn_graph)

    # edges = np.vstack(np.where(valid_connections)).T

    plot(cc, coms, tans)
