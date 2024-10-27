import re
import numpy as np
from typing import Tuple, List, Optional


class AllenSpaceConverter:
    ANNOTATION_25_RAS_SHAPE = np.array([456, 528, 320], dtype=float)
    TARGET_ORIGIN_MM_RAS_CENTER = ANNOTATION_25_RAS_SHAPE / 2 * 25e-3
    AC_ORIGIN_VOXEL_RAS = np.array([228, 313, 113], dtype=float)

    @staticmethod
    def parse_unit_code(unit_code: str) -> Tuple[str, Optional[List[float]]]:
        """Parse the unit code string."""
        match = re.match(r"(m|mm|um)(?:\((.*)\))?$", unit_code)
        if not match:
            raise ValueError(f"Invalid unit code: {unit_code}")

        unit, multiplier_str = match.groups()
        multiplier = (
            [float(x) for x in multiplier_str.split(",")]
            if multiplier_str
            else None
        )
        return unit, multiplier

    @staticmethod
    def parse_orientation_code(
        orientation_code: str,
    ) -> Tuple[List[int], List[int]]:
        """Parse the orientation code string."""
        orientation_patterns = [
            r"(R|L)(A|P)(S|I)",
            r"(R|L)(S|I)(A|P)",
            r"(A|P)(R|L)(S|I)",
            r"(A|P)(S|I)(R|L)",
            r"(S|I)(R|L)(A|P)",
            r"(S|I)(A|P)(R|L)",
        ]
        dims_mapping = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ]

        for pattern, dims in zip(orientation_patterns, dims_mapping):
            match = re.match(pattern, orientation_code)
            if match:
                flip = [0, 0, 0]
                target_orientation = "RAS"
                for i in range(3):
                    flip[i] = (
                        1
                        if match.group(i + 1) == target_orientation[dims[i]]
                        else -1
                    )
                return dims, flip

        raise ValueError(f"Invalid orientation code: {orientation_code}")

    @classmethod
    def parse_origin_code(
        cls, origin_code: str, orientation_code: str
    ) -> np.ndarray:
        """Parse the origin code string."""
        if origin_code == "center":
            return cls.TARGET_ORIGIN_MM_RAS_CENTER
        elif origin_code == "ac":
            return cls.AC_ORIGIN_VOXEL_RAS * 25e-3
        elif origin_code == "corner":
            dims, flip = cls.parse_orientation_code(orientation_code)
            origin = np.zeros(3)
            for i in range(3):
                if flip[i] < 0:
                    origin[dims[i]] = (
                        cls.ANNOTATION_25_RAS_SHAPE[dims[i]] * 25e-3
                    )
            return origin
        else:
            raise ValueError(f"Invalid origin code: {origin_code}")

    @staticmethod
    def get_affine_unit(
        unit: str, multiplier: Optional[List[float]]
    ) -> np.ndarray:
        """Get the affine transformation matrix for unit conversion."""
        to_mm = {"m": 1e3, "mm": 1, "um": 1e-3}[unit]
        multiplier = multiplier or [1, 1, 1]
        multiplier += [multiplier[-1]] * (3 - len(multiplier))
        return np.diag([to_mm * m for m in multiplier] + [1])

    @staticmethod
    def get_affine_orientation(dims: List[int], flip: List[int]) -> np.ndarray:
        """Get the affine transformation matrix for orientation."""
        A = np.zeros((4, 4))
        for i in range(3):
            A[dims[i], i] = flip[i]
        A[3, 3] = 1
        return A

    @classmethod
    def get_affine_origin(cls, origin_mm_ras: np.ndarray) -> np.ndarray:
        """Get the affine transformation matrix for origin translation."""
        origin_shift = origin_mm_ras - cls.TARGET_ORIGIN_MM_RAS_CENTER
        return np.array(
            [
                [1, 0, 0, origin_shift[0]],
                [0, 1, 0, origin_shift[1]],
                [0, 0, 1, origin_shift[2]],
                [0, 0, 0, 1],
            ]
        )

    @classmethod
    def to_allen_mm_ras_center(
        cls, unit_code: str, orientation_code: str, origin_code: str
    ) -> np.ndarray:
        """Convert coordinates to Allen space (mm, RAS, center)."""
        unit, multiplier = cls.parse_unit_code(unit_code)
        dims, flip = cls.parse_orientation_code(orientation_code)
        origin = cls.parse_origin_code(origin_code, orientation_code)

        A_unit = cls.get_affine_unit(unit, multiplier)
        A_reorient = cls.get_affine_orientation(dims, flip)
        A_origin = cls.get_affine_origin(origin)

        return A_origin @ A_reorient @ A_unit

    @classmethod
    def convert_allen_space(
        cls, from_params: List[str], to_params: List[str]
    ) -> np.ndarray:
        """Convert between different Allen space coordinate systems."""
        to_standard = cls.to_allen_mm_ras_center(*from_params)
        to_target = cls.to_allen_mm_ras_center(*to_params)
        return np.linalg.inv(to_target) @ to_standard


def test_to_allen_mm_ras_center():
    """Test the to_allen_mm_ras_center method."""
    converter = AllenSpaceConverter()

    # Test unit code parsing
    assert converter.parse_unit_code("m") == ("m", None)
    assert converter.parse_unit_code("mm(25)") == ("mm", [25.0])
    assert converter.parse_unit_code("um(10,10,200)") == (
        "um",
        [10.0, 10.0, 200.0],
    )

    # Test affine unit matrix
    A_unit = converter.get_affine_unit("um", [10, 10, 200])
    A_unit_expected = np.diag([0.01, 0.01, 0.2, 1.0])
    np.testing.assert_array_almost_equal(A_unit, A_unit_expected)

    # Test orientation code parsing and affine orientation matrix
    dims, flip = converter.parse_orientation_code("PIR")
    dims, flip = converter.parse_orientation_code("PIR")
    assert dims == [1, 2, 0] and flip == [
        -1,
        1,
        1,
    ], f"Expected dims=[1, 2, 0] and flip=[-1, 1, 1], but got dims={dims} and flip={flip}"
    A_reorient = converter.get_affine_orientation(dims, flip)
    A_reorient_expected = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    np.testing.assert_array_equal(A_reorient, A_reorient_expected)

    # Test origin code parsing
    origin_mm_ras = converter.parse_origin_code("center", "RAS")
    np.testing.assert_array_almost_equal(
        origin_mm_ras, np.array([5.7, 6.6, 4.0])
    )

    # Test full conversion
    A = converter.to_allen_mm_ras_center("um", "PIR", "corner")
    A_expected = np.array(
        [
            [0.0, 0.0, 0.001, -5.7],
            [-0.001, 0.0, 0.0, 5.35],
            [0.0, -0.001, 0.0, 5.15],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_almost_equal(A, A_expected, decimal=3)

    print("All tests passed!")


if __name__ == "__main__":
    test_to_allen_mm_ras_center()

