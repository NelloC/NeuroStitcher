import os
from storage import NeuronDataStorage
from NeuronModel import NeuronModel
from view import NeuronView
from controller import NeuronController
from loaders import Loader
from connector import NeuronConnector
from vector_calculations import VectorCalculator
from clustering import ClusteringManager
from neuron_stitcher import NeuronStitcher
from interactive_stitch_visualizer import StitchVisualizer  # Importa il visualizzatore interattivo
from debug_components import plot_neuron_analysis  # Importa la funzione di visualizzazione
import numpy as np

# Inizializzazione del sistema
print("Inizio esecuzione di main.py")
file_path = "data/ms1821_alignment_soloVPM_nocontours_neuritescorrected_aligned.json"
unit_orientation_origin = ["um", "RAS", "corner"]

# Caricamento dati del neurone
loader = Loader()
neuron_data = loader.load_morphology_from_file(
    file_path, unit_orientation_origin=unit_orientation_origin
)
print("Dati del neurone caricati")

# Inizializzazione dei componenti necessari
vector_calculator = VectorCalculator()
clustering_manager = ClusteringManager()
connector = NeuronConnector(neuron_data, vector_calculator, clustering_manager)

# Creazione del modello, vista e controller
model = NeuronModel(neuron_data, connector)
view = NeuronView()
controller = NeuronController(model, view)

# Configura e avvia NeuronStitcher per calcolare gli stitching
print("Esecuzione dello stitching...")
# Inizializzazione dello stitcher
# Calcolo dei best stitches
stitcher = NeuronStitcher(file_path, unit_orientation_origin)
best_stitches = stitcher.find_best_stitches(pieces=model.get_lines())

# Organizza i candidati per visualizzazione
candidates = {
    piece_id: stitch_list
    for piece_id, stitch_list in best_stitches.items()
}

# Inizializza il visualizzatore interattivo
best_stitches_interactive = []
visualizer = StitchVisualizer(neuron_data, candidates, best_stitches_interactive, connector)
visualizer.start()




# Aggiorna il modello con le linee di stitching finali (interattive)
if model.connected_data_storage is None:
    model.connected_data_storage = model.data_storage

model.connect_components(stitches=best_stitches_interactive)
print("Visualizzazione della morfologia con stitching...")
controller.visualize()
controller.run()

# Aggiungi una chiamata di debug per visualizzare le analisi tramite plot
if __name__ == "__main__":
    matrices = model.get_swc_chunks()
    clustering_labels = clustering_manager.get_labels(neuron_data.points)
    
    growth_vectors = []  # Puoi popolarlo usando lo stitcher
    direction_vectors = []  # Puoi popolarlo usando lo stitcher
    
    plot_neuron_analysis(
        matrices,
        candidates,  # Modificato per visualizzare i candidati corretti
        growth_vectors,
        direction_vectors,
        clustering_labels
    )
