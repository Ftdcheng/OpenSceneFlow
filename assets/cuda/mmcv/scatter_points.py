# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

# from utils import ext_loader
import importlib

def load_ext(possible_names, funcs):
    """Try loading module from list of possible names, return first matching."""
    for name in possible_names:
        try:
            ext = importlib.import_module('mmcv' + name)
            missing = [f for f in funcs if not hasattr(ext, f)]
            if missing:
                print(f"Missing functions in 'mmcv{name}': {missing}")
                continue
            return ext  # success
        except (ModuleNotFoundError, ImportError) as e:
            print(f"Failed to import mmcv{name}: {e}")
    raise ImportError(f"Could not load mmcv extension with functions: {funcs}")

# Usage
ext_module = load_ext(
    ['', '._ext'],
    ['dynamic_point_to_voxel_forward', 'dynamic_point_to_voxel_backward'])


class _DynamicScatter(Function):

    @staticmethod
    def forward(ctx: Any,
                feats: torch.Tensor,
                coors: torch.Tensor,
                reduce_type: str = 'max') -> Tuple[torch.Tensor, torch.Tensor]:
        """convert kitti points(N, >=3) to voxels.

        Args:
            feats (torch.Tensor): [N, C]. Points features to be reduced
                into voxels.
            coors (torch.Tensor): [N, ndim]. Corresponding voxel coordinates
                (specifically multi-dim voxel index) of each points.
            reduce_type (str, optional): Reduce op. support 'max', 'sum' and
                'mean'. Default: 'max'.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape [M, C] which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape [M, ndim].
        """
        results = ext_module.dynamic_point_to_voxel_forward(
            feats, coors, reduce_type)
        (voxel_feats, voxel_coors, point2voxel_map,
         voxel_points_count) = results
        ctx.reduce_type = reduce_type
        ctx.save_for_backward(feats, voxel_feats, point2voxel_map,
                              voxel_points_count)
        ctx.mark_non_differentiable(voxel_coors)
        return voxel_feats, voxel_coors

    @staticmethod
    def backward(ctx: Any,
                 grad_voxel_feats: torch.Tensor,
                 grad_voxel_coors: Optional[torch.Tensor] = None) -> tuple:
        (feats, voxel_feats, point2voxel_map,
         voxel_points_count) = ctx.saved_tensors
        grad_feats = torch.zeros_like(feats)
        # TODO: whether to use index put or use cuda_backward
        # To use index put, need point to voxel index
        ext_module.dynamic_point_to_voxel_backward(
            grad_feats, grad_voxel_feats.contiguous(), feats, voxel_feats,
            point2voxel_map, voxel_points_count, ctx.reduce_type)
        return grad_feats, None, None


dynamic_scatter = _DynamicScatter.apply


class DynamicScatter(nn.Module):
    """Group points by voxel coordinates and reduce each group to a voxel feature.

    This is the aggregation step of **dynamic voxelization**. Unlike hard
    voxelization, which pre-allocates a dense voxel grid and pads empty voxels,
    ``DynamicScatter`` only produces features for voxels that actually contain
    points: it collects all points sharing the same voxel coordinate, then
    reduces their features with ``'max'`` or ``'mean'``.

    Typical usage:

    - ``cluster_scatter`` in pillar/voxel encoders: compute the mean point
      position inside each voxel.
    - ``pfn_scatter`` after a point feature network: aggregate per-point
      encoded features into per-voxel features.

    Note:
        The CPU and GPU implementation get the same output, but have numerical
        difference after summation and division (e.g., 5e-7).

    Args:
        voxel_size (list): list [x, y, z] size of three dimension, shape ``(3,)``.
        point_cloud_range (list): The coordinate range of points, ``[x_min,
            y_min, z_min, x_max, y_max, z_max]``, shape ``(6,)``.
        average_points (bool): whether to use avg pooling to scatter points
            into voxel. ``True`` selects ``'mean'`` reduction, otherwise ``'max'``.
    """

    def __init__(self, voxel_size: List, point_cloud_range: List,
                 average_points: bool):
        super().__init__()

        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.average_points = average_points

    def forward_single(
            self, points: torch.Tensor,
            coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatters points into voxels.

        Args:
            points (torch.Tensor): Points/features to be reduced into voxels,
                shape ``(N, C)``. ``C`` can be any feature dimension, e.g. raw
                point channels or already-encoded features from upstream layers.
            coors (torch.Tensor): Corresponding voxel coordinates (specifically
                multi-dim voxel index) of each points, shape ``(N, ndim)``.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape ``(M, C)`` which are respectively
            reduced from input features that share the same voxel coordinates.
            The second is voxel coordinates with shape ``(M, ndim)``. ``M`` is
            the number of unique voxels among the ``N`` input points, so
            ``M <= N``.
        """
        reduce = 'mean' if self.average_points else 'max'
        return dynamic_scatter(points.contiguous(), coors.contiguous(), reduce)

    def forward(self, points: torch.Tensor,
                coors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scatters points/features into voxels.

        Supports single-frame (``coors`` without batch dimension) and batched
        (``coors`` with leading batch index) inputs.

        Args:
            points (torch.Tensor): Points/features to be reduced into voxels,
                shape ``(N, C)``. ``C`` can be any feature dimension, e.g. raw
                point channels or encoded features from upstream layers.
            coors (torch.Tensor): Corresponding voxel coordinates of each point.
                If ``coors.size(-1) == ndim``, shape is ``(N, ndim)`` and no
                batch dimension is expected. Otherwise, the first column is
                the batch index and shape is ``(N, ndim + 1)``.

        Returns:
            tuple[torch.Tensor]: A tuple contains two elements. The first one
            is the voxel features with shape ``(M, C)`` reduced from input
            features that share the same voxel coordinates. The second is voxel
            coordinates with shape ``(M, ndim)`` for the single-frame case, or
            ``(M, ndim + 1)`` for the batched case, where the first column is
            the batch index. ``M`` is the number of unique voxels among the
            ``N`` input points, so ``M <= N``.
        """
        if coors.size(-1) == 3:
            return self.forward_single(points, coors)
        else:
            batch_size = coors[-1, 0] + 1
            voxels, voxel_coors = [], []
            for i in range(batch_size):
                inds = torch.where(coors[:, 0] == i)
                voxel, voxel_coor = self.forward_single(
                    points[inds], coors[inds][:, 1:])
                coor_pad = F.pad(voxel_coor, (1, 0), mode='constant', value=i)
                voxel_coors.append(coor_pad)
                voxels.append(voxel)
            features = torch.cat(voxels, dim=0)
            feature_coors = torch.cat(voxel_coors, dim=0)

            return features, feature_coors

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'voxel_size=' + str(self.voxel_size)
        s += ', point_cloud_range=' + str(self.point_cloud_range)
        s += ', average_points=' + str(self.average_points)
        s += ')'
        return s
