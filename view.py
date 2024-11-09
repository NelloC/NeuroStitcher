import pyvista as pv
import numpy as np

class NeuronView:
    def __init__(self):
        self.plotter = pv.Plotter()
        self.neuron_actor = None
        self.connected_neuron_actor = None
        self.stitch_actor = None

    def render_neuron(self, points, lines, color="black", line_width=2):
        # Creazione corretta di cells in formato adatto a pyvista
        cells = []
        for line in lines:
            if isinstance(line, (list, np.ndarray)) and len(line) > 1:
                cells.append(len(line))  # Numero di punti nella linea
                cells.extend(line)       # Aggiungi i punti della linea

        cells = np.array(cells, dtype=int)  # Assicurati che cells sia un array di interi
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
        self.plotter.clear()
        
        # Stampa di debug per verificare il contenuto degli stitch
        print("Rendering dati originali e connessi...")
        print(f"Original Points: {original_points.shape}, Original Lines: {len(original_lines)}")
        print(f"Connected Points: {connected_points.shape}, Connected Lines: {len(connected_lines)}")
        print(f"Stitch Lines: {len(stitch_lines)}")
        print("Stitch Lines Data:", stitch_lines)  # Stampa i dati effettivi degli stitch

        # Render originale
        self.neuron_actor = self.render_neuron(
            original_points, original_lines, color="black", line_width=2
        )

        # Render connessioni
        self.connected_neuron_actor = self.render_neuron(
            connected_points, connected_lines, color="blue", line_width=2
        )

        # Render linee di stitching (se presenti)
        if stitch_lines:
            self.stitch_actor = self.render_neuron(
                connected_points, stitch_lines, color="red", line_width=3
            )
        else:
            print("Nessuna linea di stitching trovata per la visualizzazione.")


    def show(self):
        self.plotter.show()

    def close(self):
        self.plotter.close()

