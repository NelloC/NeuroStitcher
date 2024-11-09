import numpy as np
from storage import NeuronDataStorage
from vector_calculations import VectorCalculator
from clustering import ClusteringManager

class NeuronConnector:
    def __init__(self, neuron_data, vector_calculator, clustering_manager):
        self.neuron_data = neuron_data
        self.vector_calculator = vector_calculator
        self.clustering_manager = clustering_manager
        self.stitch_lines = []  # Per tenere traccia delle linee di stitching aggiunte

    def connect_components(self, stitches):
        for stitch in stitches:
            piece_id = stitch["ending"]
            neighbor_point = stitch["candidate"]

            start_idx = self.neuron_data.add_point(piece_id[:3])
            end_idx = self.neuron_data.add_point(neighbor_point[:3])
            line_idx = self.neuron_data.add_line(
                tp=5,
                first_point_idx=start_idx,
                num_points=(end_idx - start_idx + 1),
                parent_line_id=-1
            )

            # Salva indici reali per rimozione
            stitch["start_idx"] = start_idx
            stitch["end_idx"] = end_idx
            self.stitch_lines.append((line_idx, start_idx, end_idx))
            print(f"[DEBUG] Linea aggiunta con indice {line_idx}: start={start_idx}, end={end_idx}")


    def remove_connection(self, stitch):
        start_idx = stitch.get("start_idx")
        end_idx = stitch.get("end_idx")

        if start_idx is None or end_idx is None:
            print("[ERROR] Indici mancanti nel candidato.")
            return

        try:
            for line_idx, s_idx, e_idx in self.stitch_lines:
                if s_idx == start_idx and e_idx == end_idx:
                    self.neuron_data.remove_line(line_idx)
                    self.neuron_data.remove_point(start_idx)
                    self.neuron_data.remove_point(end_idx)
                    self.stitch_lines.remove((line_idx, s_idx, e_idx))
                    print(f"[INFO] Linea di stitching rimossa tra {start_idx} e {end_idx}")
                    return

            print(f"[WARN] Connessione di stitching non trovata tra {start_idx} e {end_idx}")
        except IndexError as e:
            print(f"[ERROR] Errore durante la rimozione: {e}")



            
    def find_component_endpoints(self):
        """
        Trova i punti finali per ogni componente esistente.

        Returns:
            list: Lista di tuple (start_point, end_point).
        """
        endpoints = []
        for line in self.neuron_data.lines:
            start_idx = line[1]
            end_idx = line[1] + line[2] - 1
            start_point = self.neuron_data.points[start_idx][:3]
            end_point = self.neuron_data.points[end_idx][:3]
            endpoints.append((start_point, end_point))
        return endpoints

