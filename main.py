from storage import NeuronDataStorage
from model import NeuronModel
from view import NeuronView
from controller import NeuronController
from loaders import Loader
from connector import NeuronConnector
from vector_calculations import VectorCalculator
from clustering import ClusteringManager

if __name__ == "__main__":
    file_path = "data/ms1821_alignment_soloVPM_nocontours_neuritescorrected_aligned.json"
    unit_orientation_origin = ["um", "RAS", "corner"]
    loader = Loader()

    # Carica i dati iniziali del neurone
    neuron_data = loader.load_morphology_from_file(
        file_path, unit_orientation_origin=unit_orientation_origin
    )

    # Inizializza VectorCalculator e ClusteringManager
    vector_calculator = VectorCalculator()
    clustering_manager = ClusteringManager()

    # Crea il NeuronConnector passando tutti gli argomenti necessari
    connector = NeuronConnector(neuron_data, vector_calculator, clustering_manager)

    # Crea il Model, View e Controller
    model = NeuronModel(neuron_data, connector)
    view = NeuronView()
    controller = NeuronController(model, view)

    # Visualizza la morfologia originale e connessa
    controller.visualize_components()

    # Mostra la visualizzazione
    controller.run()

    # Chiudi la visualizzazione quando finito
    controller.close()
