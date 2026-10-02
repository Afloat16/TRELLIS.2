import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

spec = importlib.util.spec_from_file_location("native_mesh_utils", Path(__file__).resolve().parents[1] / "trellis2/utils/mesh_utils.py")
mesh_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mesh_utils)


class MeshPlyLayoutTest(unittest.TestCase):
    vertices = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]], dtype=np.float32)
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    quads = np.array([[0, 1, 2, 3]], dtype=np.int32)

    def assert_mesh(self, actual):
        for received, expected in zip(actual, (self.vertices, self.triangles, self.quads)):
            np.testing.assert_array_equal(received, expected)

    def test_binary_writer_with_rgb_and_rgba_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            for channels in [3, 4]:
                colors = np.arange(4 * channels, dtype=np.uint8).reshape(4, channels) + 20
                mesh_utils.write_ply(path, self.vertices, self.triangles, self.quads, colors)
                self.assert_mesh(mesh_utils.read_ply(path))

    def test_header_controls_property_order_types_and_endianness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            vertices = np.empty(4, dtype=[("red", "u1"), ("z", "f8"), ("x", "f8"), ("y", "f8"), ("nx", "f4")])
            vertices["red"] = 77
            for i, key in enumerate(["x", "y", "z"]):
                vertices[key] = self.vertices[:, i]
            vertices["nx"] = .5
            faces = np.empty(2, dtype=[("vertex_indices", object)])
            faces["vertex_indices"] = [self.triangles[0], self.quads[0]]
            for endian in ["<", ">"]:
                PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(faces, "face")],
                        byte_order=endian).write(path)
                self.assert_mesh(mesh_utils.read_ply(path))

    def test_ascii_and_empty_face_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh.ply"
            mesh_utils.write_ply(path, self.vertices, self.triangles, self.quads, ascii=True)
            self.assert_mesh(mesh_utils.read_ply(path))
            mesh_utils.write_ply(path, self.vertices, np.empty((0, 3), np.int32), np.empty((0, 4), np.int32))
            vertices, triangles, quads = mesh_utils.read_ply(path)
            np.testing.assert_array_equal(vertices, self.vertices)
            self.assertEqual(triangles.shape, (0, 3))
            self.assertEqual(quads.shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
