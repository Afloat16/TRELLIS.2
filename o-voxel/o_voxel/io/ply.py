from typing import *
import io
import torch
import numpy as np
import plyfile


__all__ = [
    "read_ply",
    "write_ply",
]


DTYPE_MAP = {
    torch.uint8: 'u1',
    torch.uint16: 'u2',
    torch.uint32: 'u4',
    torch.int8: 'i1',
    torch.int16: 'i2',
    torch.int32: 'i4',
    torch.float32: 'f4',
    torch.float64: 'f8'
}


def read_ply(file) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Read a PLY file containing voxels.
    
    Args:
        file: Path or file-like object of the PLY file.
        
    Returns:
        torch.Tensor: the coordinates of the voxels.
        Dict[str, torch.Tensor]: the attributes of the voxels.
    """
    plydata = plyfile.PlyData.read(file)
    xyz = np.stack([plydata.elements[0][k] for k in ['x', 'y', 'z']], axis=1)
    coord = np.round(xyz).astype(int)
    coord = torch.from_numpy(coord)
    
    attr_channels = {}
    for key in plydata.elements[0].data.dtype.names:
        if key in ('x', 'y', 'z'):
            continue
        name, channel = key.rsplit('_', 1)
        attr_channels.setdefault(name, {})[int(channel)] = key

    attr = {}
    for name, channels in attr_channels.items():
        expected = list(range(len(channels)))
        if sorted(channels) != expected:
            raise ValueError(f"Non-contiguous channels for attribute {name}")
        attr[name] = np.stack(
            [plydata.elements[0][channels[channel]] for channel in expected], axis=1
        )
    attr = {k: torch.from_numpy(v) for k, v in attr.items()}
    
    return coord, attr


def write_ply(file, coord: torch.Tensor, attr: Dict[str, torch.Tensor]):
    """
    Write a PLY file containing voxels.
    
    Args:
        file: Path or file-like object of the PLY file.
        coord: the coordinates of the voxels.
        attr: the attributes of the voxels.
    """    
    dtypes = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    for k, v in attr.items():
        for j in range(v.shape[-1]):
            assert v.dtype in DTYPE_MAP, f"Unsupported data type {v.dtype} for attribute {k}"
            dtypes.append((f'{k}_{j}', DTYPE_MAP[v.dtype]))
    data = np.empty(len(coord), dtype=dtypes)
    all_chs = np.concatenate([coord.cpu().numpy().astype(np.float32)] + [v.cpu().numpy() for v in attr.values()], axis=1)
    data[:] = list(map(tuple, all_chs))
    plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(file)
    