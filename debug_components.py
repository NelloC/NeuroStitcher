import pyvista as pv
import numpy as np
import random

def random_color():
    return [random.random(), random.random(), random.random()]

def plot_neuron_analysis(
    matrices,
    attractor_points,
    growth_vectors,
    direction_vectors,
    clustering_labels,
    search_radius=100
):
    """
    Visualizza i dati del neurone, incluse le connessioni, i punti attrattori e le zone di clustering.
    
    Parameters:
    - matrices: Lista di matrici SWC per ciascun pezzo.
    - attractor_points: Dizionario con punti attrattori e loro informazioni.
    - growth_vectors: Lista di vettori di crescita.
    - direction_vectors: Lista di vettori di direzione.
    - clustering_labels: Etichette dei cluster per ogni punto.
    - search_radius: Raggio di ricerca per visualizzare le "bolle concentriche".
    """
    plotter = pv.Plotter()

    for matrix, attractor_info, growth_vecs, dir_vecs in zip(
        matrices, attractor_points, growth_vectors, direction_vectors
    ):
        # Coordinate e connessioni dai dati SWC
        coords = matrix[:, 2:5]
        parent_ids = matrix[:, 6].astype(int)

        # Definizione delle linee tra parent e child
        lines = []
        for i, parent_id in enumerate(parent_ids):
            if parent_id != -1:
                lines.append([2, i, parent_id])

        line_mesh = pv.PolyData(coords, lines=np.hstack(lines))
        plotter.add_mesh(line_mesh, color="gray", line_width=2, opacity=0.5)

        # Visualizza i cluster
        unique_labels = np.unique(clustering_labels)
        for label in unique_labels:
            cluster_points = coords[clustering_labels == label]
            plotter.add_mesh(
                pv.PolyData(cluster_points),
                color=random_color(),
                point_size=10,
                render_points_as_spheres=True,
                label=f"Cluster {label}"
            )

        # Visualizza gli attractor points e le "bolle"
        for point, attractor_data in attractor_info.items():
            neighbors = attractor_data["neighbor_points"]
            distances = attractor_data["distances"]

            plotter.add_mesh(
                pv.PolyData([point]), color="red", point_size=12, render_points_as_spheres=True
            )

            # Bolle concentriche attorno al punto
            for radius in np.linspace(0, search_radius, num=5):
                sphere = pv.Sphere(radius=radius, center=point)
                plotter.add_mesh(sphere, color="lightblue", opacity=0.1)

            # Visualizza i vettori di connessione
            for neighbor in neighbors:
                vector = neighbor - point
                arrow = pv.Arrow(start=point, direction=vector, scale=np.linalg.norm(vector))
                plotter.add_mesh(arrow, color="yellow")

    plotter.camera_position = "iso"
    plotter.show()
