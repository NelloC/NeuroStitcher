import os
from storage import NeuronDataStorage
from NeuronModel import NeuronModel
from view import NeuronView
from controller import NeuronController
from loaders import Loader
from connector import NeuronConnector
from vector_calculations import VectorCalculator
from clustering import ClusteringManager
from IPython.display import Image, display  # Per visualizzare l'immagine in Colab

# Installa e avvia xvfb per il rendering offscreen
print("Inizio esecuzione di main.py")
print("Installazione e avvio di Xvfb...")

!apt-get install -y xvfb > /dev/null 2>&1
os.system("Xvfb :99 -screen 0 1024x768x24 &")
os.environ["DISPLAY"] = ":99"

print("Display virtuale attivato")

# Caricamento dei dati
print("Caricamento dati del neurone...")
file_path = "data/ms1821_alignment_soloVPM_nocontours_neuritescorrected_aligned.json"
unit_orientation_origin = ["um", "RAS", "corner"]
loader = Loader()

neuron_data = loader.load_morphology_from_file(
    file_path, unit_orientation_origin=unit_orientation_origin
)
print("Dati del neurone caricati")

print("VectorCalculator e ClusteringManager inizializzati")
vector_calculator = VectorCalculator()
clustering_manager = ClusteringManager()

print("NeuronConnector creato")
connector = NeuronConnector(neuron_data, vector_calculator, clustering_manager)

print("Model, View e Controller creati")
model = NeuronModel(neuron_data, connector)
view = NeuronView()
controller = NeuronController(model, view)

print("Visualizzazione dei componenti della morfologia...")
controller.visualize_components()

print("Esecuzione della visualizzazione...")
controller.run()

# Visualizza l'immagine salvata in Colab
display(Image("/content/neuron_visualization.png"))

# Chiudi la visualizzazione
controller.close()

