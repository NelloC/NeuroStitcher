from typing import Dict, List, Any, Set
import numpy as np
from utils import AllenSpaceConverter


class NeuronDataStorage:
    def __init__(
        self, neuron_dict: Dict[str, Any], unit_orientation_origin: List[str]
    ):
        self.meta_data = neuron_dict["metaData"]
        self.custom_types = neuron_dict["customTypes"]
        self.custom_properties = neuron_dict["customProperties"]
        self.lines = neuron_dict["treeLines"]["data"]
        self.line_columns = neuron_dict["treeLines"]["columns"]
        self.points = np.array(neuron_dict["treePoints"]["data"])
        self.point_columns = neuron_dict["treePoints"]["columns"]
        self.unit_orientation_origin = unit_orientation_origin
        self.space_converter = AllenSpaceConverter()

    def get_type_to_id_mapping(self) -> Dict[str, int]:
        type_to_id = {"soma": 1, "axon": 2, "dendrite": 3, "apical": 4}
        for k, v in self.custom_types.items():
            type_to_id[k] = v["id"]
        return type_to_id

    def get_object_properties(self) -> Dict[int, Dict[str, Any]]:
        object_properties = {}
        for assignment in self.custom_properties.get("for", []):
            props = assignment["set"]
            for obj_id in assignment.get("objects", []):
                if obj_id not in object_properties:
                    object_properties[obj_id] = props.copy()
                else:
                    object_properties[obj_id].update(props)
        return object_properties

    def set_object_properties(
        self, object_properties: Dict[int, Dict[str, Any]]
    ) -> None:
        property_groups = {}
        for obj_id, props in object_properties.items():
            key = frozenset(props.items())
            if key not in property_groups:
                property_groups[key] = []
            property_groups[key].append(obj_id)

        property_assignments = [
            {"objects": obj_ids, "set": dict(props)}
            for props, obj_ids in property_groups.items()
        ]
        self.custom_properties["for"] = property_assignments

    def get_reoriented_point(
        self, point_idx: int, unit_orientation_origin: List[str]
    ) -> np.ndarray:
        A = AllenSpaceConverter.convert_allen_space(
            self.unit_orientation_origin, unit_orientation_origin
        )
        point = self.points[point_idx, :].copy()
        point[3] = 1.0
        return point @ A.T

    def reorient(self, unit_orientation_origin: List[str]) -> None:
        A = AllenSpaceConverter.convert_allen_space(
            self.unit_orientation_origin, unit_orientation_origin
        )
        warped_points = self.points.copy()
        warped_points[:, 3] = 1.0
        warped_points = warped_points @ A.T
        self.unit_orientation_origin = unit_orientation_origin

        warped_points[:, 3] = np.linalg.det(A) ** (1 / 3) * self.points[:, 3]
        self.points = warped_points

    def add_point(self, point: np.ndarray) -> int:
        self.points = np.vstack((self.points, point))
        return len(self.points) - 1

    def add_line(
        self,
        tp: int,
        first_point_idx: int,
        num_points: int,
        parent_line_id: int,
        neg_offset: int = 0,
    ) -> int:
        self.lines.append(
            [tp, first_point_idx, num_points, parent_line_id, neg_offset]
        )
        return len(self.lines) - 1

    def add_neurite_type(self, geom: str, attrs: Dict[str, Any]) -> int:
        max_type_id = max(ctype["id"] for ctype in self.custom_types.values())
        new_type_id = max_type_id + 1
        new_type = attrs.copy()
        new_type["id"] = new_type_id
        self.custom_types[geom] = new_type
        return new_type_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metaData": self.meta_data,
            "customTypes": self.custom_types,
            "customProperties": self.custom_properties,
            "treeLines": {"columns": self.line_columns, "data": self.lines},
            "treePoints": {
                "columns": self.point_columns,
                "data": self.points.tolist(),
            },
        }

    def to_swc(self, include_lines: List[int] = None) -> np.ndarray:
        if include_lines:
            tree_lines = np.zeros((len(include_lines) + 1, 5), dtype=np.uint32)
            new_line_ids = {}
            for i, line_id in enumerate(include_lines, start=1):
                tree_lines[i] = self.lines[line_id]
                new_line_ids[line_id] = i

            for i in range(1, tree_lines.shape[0]):
                parent_id = tree_lines[i, 3]
                tree_lines[i, 3] = new_line_ids.get(parent_id, 0)
        else:
            tree_lines = np.array(self.lines).reshape(-1, 5)

        tree_points = self.points

        # Initialize the swc-like data matrix
        swc_data = np.empty((len(tree_points), 7), dtype=object)
        # Reconstruct the ordered list of sample points
        sample_ids = {}
        new_sample_id = 1
        for line in tree_lines:
            samples_type, starting_sample, num_samples, parent, offset = line
            parent_point = 0
            # If the line has a parent line
            if parent > 0:
                parent_line = tree_lines[parent]
                # Represents this line's samples offset
                # from last sample of parent line
                parent_point = parent_line[1] + parent_line[2] - 1 - offset
            for i in range(starting_sample, starting_sample + num_samples):
                sample_ids[i] = new_sample_id
                # Zero-ing the parent ID if it's not found
                parent_sample_id = sample_ids.get(parent_point, 0)
                # Filling the SWC data array
                swc_data[new_sample_id - 1, 0] = new_sample_id
                swc_data[new_sample_id - 1, 1] = samples_type
                swc_data[new_sample_id - 1, 2:6] = tree_points[i]
                swc_data[new_sample_id - 1, 6] = (
                    parent_sample_id if parent_sample_id > 0 else -1
                )
                # Re-initalize quantities
                parent_point = i
                new_sample_id += 1
        # Filter unused matrix cells and reshape to desired dimensions
        return swc_data[swc_data != np.array(None)].reshape(-1, 7)
