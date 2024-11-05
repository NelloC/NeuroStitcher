import os
from storage import NeuronDataStorage
from NeuronModel import NeuronModel
from view import NeuronView
from controller import NeuronController
from loaders import Loader
from connector import NeuronConnector
from vector_calculations import VectorCalculator
from clustering import ClusteringManager

# Inizio script principale
print("Inizio esecuzione di main.py")

# Caricamento dei dati del neurone
print("Caricamento dati del neurone...")
file_path = "data/ms1821_alignment_soloVPM_nocontours_neuritescorrected_aligned.json"
unit_orientation_origin = ["um", "RAS", "corner"]
loader = Loader()

neuron_data = loader.load_morphology_from_file(
    file_path, unit_orientation_origin=unit_orientation_origin
)
print("Dati del neurone caricati")

# Inizializzazione dei componenti principali
print("VectorCalculator e ClusteringManager inizializzati")
vector_calculator = VectorCalculator()
clustering_manager = ClusteringManager()

print("NeuronConnector creato")
connector = NeuronConnector(neuron_data, vector_calculator, clustering_manager)

print("Model, View e Controller creati")
model = NeuronModel(neuron_data, connector)
view = NeuronView()
controller = NeuronController(model, view)

# Visualizzazione dei componenti della morfologia
print("Visualizzazione dei componenti della morfologia...")
controller.visualize_components()

# Esecuzione della visualizzazione
print("Esecuzione della visualizzazione...")
controller.run()

# Chiudi la visualizzazione
controller.close()



