from storage import NeuronDataStorage
from NeuronModel import NeuronModel
from view import NeuronView
from controller import NeuronController
from loaders import Loader
from connector import NeuronConnector
from vector_calculations import VectorCalculator
from clustering import ClusteringManager

print("Inizio esecuzione di main.py")

# Setup virtual display
from pyvirtualdisplay import Display
import pyvista as pv
display = Display(visible=0, size=(800, 600))
display.start()
pv.start_xvfb()
print("Display virtuale attivato")

file_path = "data/ms1821_alignment_soloVPM_nocontours_neuritescorrected_aligned.json"
unit_orientation_origin = ["um", "RAS", "corner"]

# Carica i dati
loader = Loader()
print("Caricamento dati del neurone...")
neuron_data = loader.load_morphology_from_file(
    file_path, unit_orientation_origin=unit_orientation_origin
)
print("Dati del neurone caricati")

# Inizializza VectorCalculator e ClusteringManager
vector_calculator = VectorCalculator()
clustering_manager = ClusteringManager()
print("VectorCalculator e ClusteringManager inizializzati")

# Crea il NeuronConnector
connector = NeuronConnector(neuron_data, vector_calculator, clustering_manager)
print("NeuronConnector creato")

# Crea il Model, View e Controller
model = NeuronModel(neuron_data, connector)
view = NeuronView()
controller = NeuronController(model, view)
print("Model, View e Controller creati")

# Visualizza la morfologia originale e connessa
print("Visualizzazione dei componenti della morfologia...")
controller.visualize_components()

# Mostra la visualizzazione
print("Esecuzione della visualizzazione...")
controller.run()

# Chiudi la visualizzazione quando finito
controller.close()
print("Visualizzazione completata e chiusa")
