from abc import ABC, abstractmethod
import json
from storage import NeuronDataStorage


class MorphologyLoader(ABC):
    @abstractmethod
    def load(self, file_path: str, **kwargs) -> NeuronDataStorage:
        pass


class SWCLoader(MorphologyLoader):
    def load(self, file_path: str, **kwargs) -> NeuronDataStorage:
        raise Exception("SWC file handling not implemented yet!")


class JSONLoader(MorphologyLoader):
    def load(self, file_path: str, **kwargs) -> NeuronDataStorage:
        with open(file_path, "r") as f:
            neuron_data = json.load(f)
        u_o_o = kwargs.get("unit_orientation_origin", ["um", "RAS", "corner"])
        return NeuronDataStorage(neuron_data, u_o_o)


# Using factory pattern, creating Loaders from a superclass, based on the filetype passed
# If a path ends with "JSON" the loader automatically calls the JSON loader to create the morphology
class Loader:
    @staticmethod
    def load_morphology_from_file(
        file_path: str, **kwargs
    ) -> NeuronDataStorage:
        if file_path.endswith(".swc"):
            _loader = SWCLoader()
            return _loader.load(file_path, **kwargs)
        elif file_path.endswith(".json"):
            _loader = JSONLoader()
            return _loader.load(file_path, **kwargs)
        else:
            raise ValueError("Unsupported file format")
