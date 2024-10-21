from graph import NeuronGraphProcessor
from storage import NeuronDataStorage
from model import NeuronModel
from view import NeuronView
from controller import NeuronController
from loaders import Loader

if __name__ == "__main__":
    file_path = "NeuroStitcher/data/ms1821_alignment_soloVPM_nocontours_neuritescorrected_aligned.json"
    unit_orientation_origin = ["um", "RAS", "corner"]
    loader = Loader()

    # Load the initial neuron data
    neuron_data = loader.load_morphology_from_file(
        file_path, unit_orientation_origin=unit_orientation_origin
    )

    # Create NeuronGraphProcessor
    graph_processor = NeuronGraphProcessor(neuron_data)

    # Create the Model, View, and Controller
    model = NeuronModel(neuron_data, graph_processor)
    view = NeuronView()
    controller = NeuronController(model, view)

    # Visualize the original and connected morphology
    controller.visualize_components()

    # Show the plot
    controller.run()

    # Close the plot when done
    controller.close()
