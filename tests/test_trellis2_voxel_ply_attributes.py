import importlib.util
import io
import unittest
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

spec = importlib.util.spec_from_file_location("native_voxel_ply", Path(__file__).resolve().parents[1] / "o-voxel/o_voxel/io/ply.py")
voxel_ply = importlib.util.module_from_spec(spec)
spec.loader.exec_module(voxel_ply)


class VoxelPlyAttributeTest(unittest.TestCase):
    def test_shared_prefix_attributes_roundtrip_independently(self):
        coordinates = torch.tensor([[1, 2, 3], [4, 5, 6]])
        attributes = {
            "color": torch.tensor([[10, 20], [30, 40]], dtype=torch.uint8),
            "color_extra": torch.tensor([[.25], [.75]], dtype=torch.float32),
            "material_roughness": torch.tensor([[.1], [.9]], dtype=torch.float32),
        }
        stream = io.BytesIO()
        voxel_ply.write_ply(stream, coordinates, attributes)
        stream.seek(0)
        recovered_coord, recovered_attr = voxel_ply.read_ply(stream)
        torch.testing.assert_close(recovered_coord, coordinates)
        self.assertEqual(set(recovered_attr), set(attributes))
        for key in attributes:
            torch.testing.assert_close(recovered_attr[key], attributes[key])

    def test_channels_with_two_digit_indices_keep_numeric_order(self):
        coordinates = torch.tensor([[1, 2, 3]])
        attributes = {"features": torch.arange(12).float().reshape(1, 12)}
        stream = io.BytesIO()
        voxel_ply.write_ply(stream, coordinates, attributes)
        stream.seek(0)
        _, recovered = voxel_ply.read_ply(stream)
        torch.testing.assert_close(recovered["features"], attributes["features"])

    def test_noncontiguous_channel_indices_report_invalid_layout(self):
        data = np.zeros(1, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("feature_1", "f4")])
        stream = io.BytesIO()
        PlyData([PlyElement.describe(data, "vertex")]).write(stream)
        stream.seek(0)
        with self.assertRaises(ValueError):
            voxel_ply.read_ply(stream)


if __name__ == "__main__":
    unittest.main()
