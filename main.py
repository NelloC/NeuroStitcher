from storage import NeuronDataStorage
from NeuronModel import NeuronModel
from view import NeuronView
from controller import NeuronController
from loaders import Loader
from connector import NeuronConnector  # Usa NeuronConnector al posto di NeuronGraphProcessor

if __name__ == "__main__":
    file_path = "data/ms1821_alignment_soloVPM_nocontours_neuritescorrected_aligned.json"
    unit_orientation_origin = ["um", "RAS", "corner"]
    loader = Loader()

    # Carica i dati iniziali del neurone
    neuron_data = loader.load_morphology_from_file(
        file_path, unit_orientation_origin=unit_orientation_origin
    )

    # Crea il NeuronConnector per gestire le connessioni dei componenti
    connector = NeuronConnector(neuron_data)

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
