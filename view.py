import pyvista as pv
import numpy as np
import random

class NeuronView:
    def __init__(self):
        self.plotter = pv.Plotter(off_screen=True)  # Modalità offscreen per Colab
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
        print("Inizio visualizzazione dei componenti connessi...")
        
        for i, component in enumerate(connected_components):
            print(f"Rendering componente {i + 1} di {len(connected_components)} con {len(component)} punti...")
            
            # Controlla se il componente ha dati prima di continuare
            if len(component) == 0:
                print(f"Componente {i + 1} è vuoto, saltato.")
                continue

            # Estrai gli ID dei punti, le coordinate e gli ID dei genitori
            try:
                point_ids = component[:, 0].astype(int)
                coords = component[:, 2:5].astype(np.float32)
                parent_ids = component[:, 6].astype(int)
            except IndexError as e:
                print(f"Errore nell'accesso ai dati del componente {i + 1}: {e}")
                continue

            # Mappa gli ID dei punti per creare linee di connessione
            id_to_index = {id: index for index, id in enumerate(point_ids)}
            lines = []
            for idx, parent_id in enumerate(parent_ids):
                if parent_id != -1 and parent_id in id_to_index:
                    lines.append([id_to_index[point_ids[idx]], id_to_index[parent_id]])

            # Conversione in formato PyVista
            cells = []
            for line in lines:
                cells.extend([2, line[0], line[1]])

            # Creazione delle mesh di linee e punti
            line_mesh = pv.PolyData(coords, lines=cells)
            point_mesh = pv.PolyData(coords)

            # Renderizza il componente con un colore casuale per distinguerlo
            random_color = "#%06x" % random.randint(0, 0xFFFFFF)
            self.plotter.add_mesh(line_mesh, color=random_color, line_width=2)
            print(f"Componente {i + 1} renderizzato con colore {random_color}")

        # Imposta la vista isometrica per la visualizzazione
        self.plotter.camera_position = "iso"
        print("Visualizzazione dei componenti connessi completata.")

    def show(self):
        # Salva l'immagine in modalità offscreen
        screenshot_path = "/content/drive/MyDrive/Thesis/GitHub/github ultimo/neuron.png"
        self.plotter.show(auto_close=False)  # Mantenerla aperta per salvare
        self.plotter.screenshot(screenshot_path)
        print(f"Immagine salvata come {screenshot_path}")

    def close(self):
        self.plotter.close()

