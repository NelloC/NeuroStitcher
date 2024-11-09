import numpy as np
from loaders import Loader  # Modificato per usare il caricatore di morphologia
from vector_calculations import VectorCalculator
from clustering import ClusteringManager
from attractor_points import AttractorPointsManager
from stitching import StitchingManager
from storage import NeuronDataStorage  # Importa NeuronDataStorage

class NeuronStitcher:
    def __init__(self, file_path, unit_orientation_origin):
        # Usa Loader per caricare i dati in NeuronDataStorage
        self.neuron_data = Loader.load_morphology_from_file(file_path, unit_orientation_origin=unit_orientation_origin)
        
        # Inizializza i componenti
        self.vector_calculator = VectorCalculator()
        self.clustering_manager = ClusteringManager()
        self.attractor_points_manager = AttractorPointsManager(self.vector_calculator)
        self.stitching_manager = StitchingManager(self.vector_calculator, self.clustering_manager, self.attractor_points_manager, self.neuron_data)

        # Assegna l'origine e orientamento unitario
        self.unit_orientation_origin = unit_orientation_origin

    def find_best_stitches(self, pieces):
        # Utilizza StitchingManager per trovare i migliori stitches, passando direttamente neuron_data
        best_stitches = self.stitching_manager.find_best_stitches(self.neuron_data, pieces)
        return best_stitches

