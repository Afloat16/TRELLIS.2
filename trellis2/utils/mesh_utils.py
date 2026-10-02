from typing import Tuple, Dict
import numpy as np
from trimesh import grouping, util, remesh
from plyfile import PlyData, PlyElement


def read_ply(filename):
    """
    Read a PLY file and return vertices, triangle faces, and quad faces.
    
    Args:
        filename (str): The file path to read from.
        
    Returns:
        vertices (np.ndarray): Array of shape [N, 3] containing vertex positions.
        tris (np.ndarray): Array of shape [M, 3] containing triangle face indices (empty if none).
        quads (np.ndarray): Array of shape [K, 4] containing quad face indices (empty if none).
    """
    ply_data = PlyData.read(filename)
    vertex_data = ply_data['vertex'].data
    vertices = np.stack([vertex_data[key] for key in ('x', 'y', 'z')], axis=1)
    vertices = vertices.astype(np.float32, copy=False)
    face_data = ply_data['face'].data
    index_key = next(
        (key for key in ('vertex_indices', 'vertex_index') if key in face_data.dtype.names),
        None,
    )
    if index_key is None:
        raise ValueError("Face vertex indices not found in PLY properties")
    tris, quads = [], []
    for indices in face_data[index_key]:
        if len(indices) == 3:
            tris.append(indices)
        elif len(indices) == 4:
            quads.append(indices)
        elif not ply_data.text:
            raise ValueError(f"Unsupported face with {len(indices)} vertices")
    tris = np.asarray(tris, dtype=np.int32).reshape(-1, 3)
    quads = np.asarray(quads, dtype=np.int32).reshape(-1, 4)
    return vertices, tris, quads


def write_ply(
    filename: str,
    vertices: np.ndarray,
    tris: np.ndarray,
    quads: np.ndarray,
    vertex_colors: np.ndarray = None,
    ascii: bool = False
):
    """
    Write a mesh to a PLY file, with the option to save in ASCII or binary format,
    and optional per-vertex colors.
    
    Args:
        filename (str): The filename to write to.
        vertices (np.ndarray): [N, 3] The vertex positions.
        tris (np.ndarray): [M, 3] The triangle indices.
        quads (np.ndarray): [K, 4] The quad indices.
        vertex_colors (np.ndarray, optional): [N, 3] or [N, 4] UInt8 colors for each vertex (RGB or RGBA).
        ascii (bool): If True, write in ASCII format; otherwise binary little-endian.
    """
    import struct

    num_vertices = len(vertices)
    num_faces = len(tris) + len(quads)

    # Build header
    header_lines = [
        "ply",
        f"format {'ascii 1.0' if ascii else 'binary_little_endian 1.0'}",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
    ]

    # Add vertex color properties if provided
    has_color = vertex_colors is not None
    if has_color:
        # Expect uint8 values 0-255
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
        # Include alpha if RGBA
        if vertex_colors.shape[1] == 4:
            header_lines.append("property uchar alpha")

    header_lines += [
        f"element face {num_faces}",
        "property list uchar int vertex_index",
        "end_header",
        ""
    ]
    header = "\n".join(header_lines)

    mode = 'w' if ascii else 'wb'
    with open(filename, mode) as f:
        # Write header
        if ascii:
            f.write(header)
        else:
            f.write(header.encode('utf-8'))

        # Write vertex data
        for i, v in enumerate(vertices):
            if ascii:
                line = f"{v[0]} {v[1]} {v[2]}"
                if has_color:
                    col = vertex_colors[i]
                    line += ' ' + ' '.join(str(int(c)) for c in col)
                f.write(line + '\n')
            else:
                # pack xyz as floats
                f.write(struct.pack('<fff', *v))
                if has_color:
                    col = vertex_colors[i]
                    # pack as uchar
                    if col.shape[0] == 3:
                        f.write(struct.pack('<BBB', *col))
                    else:
                        f.write(struct.pack('<BBBB', *col))

        # Write face data
        if ascii:
            for tri in tris:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
            for quad in quads:
                f.write(f"4 {quad[0]} {quad[1]} {quad[2]} {quad[3]}\n")
        else:
            for tri in tris:
                f.write(struct.pack('<B3i', 3, *tri))
            for quad in quads:
                f.write(struct.pack('<B4i', 4, *quad))
                

def write_pbr_ply(
    filename: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    base_color: np.ndarray,
    metallic: np.ndarray,
    roughness: np.ndarray,
    alpha: np.ndarray,
    ascii: bool = False
):
    """
    Write a mesh to a PLY file, with the option to save in ASCII or binary format,
    and optional per-vertex colors.
    
    Args:
        filename (str): The filename to write to.
        vertices (np.ndarray): [N, 3] The vertex positions.
        faces (np.ndarray): [M, 3] The triangle indices.
        base_color (np.ndarray): [N, 3] UInt8 colors for each vertex (RGB).
        metallic (np.ndarray): [N] UInt8 values for metallicness.
        roughness (np.ndarray): [N] UInt8 values for roughness.
        alpha (np.ndarray): [N] UInt8 values for alpha.
        ascii (bool): If True, write in ASCII format; otherwise binary little-endian.
    """
    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('metallic', 'u1'), ('roughness', 'u1'), ('alpha', 'u1')
    ]
    
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data['x'] = vertices[:, 0]
    vertex_data['y'] = vertices[:, 1]
    vertex_data['z'] = vertices[:, 2]
    vertex_data['red'] = base_color[:, 0]
    vertex_data['green'] = base_color[:, 1]
    vertex_data['blue'] = base_color[:, 2]
    vertex_data['metallic'] = metallic
    vertex_data['roughness'] = roughness
    vertex_data['alpha'] = alpha
    
    face_dtype = [
        ('vertex_indices', 'i4', (3,))
    ]
    
    face_data = np.empty(len(faces), dtype=face_dtype)
    face_data['vertex_indices'] = faces
    
    ply_data = PlyData([
        PlyElement.describe(vertex_data,'vertex'),
        PlyElement.describe(face_data, 'face'),
    ], text=ascii)
    ply_data.write(filename)
