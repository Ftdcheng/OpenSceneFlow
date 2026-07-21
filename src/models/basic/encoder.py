import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from assets.cuda.mmcv import Voxelization
from assets.cuda.mmcv import DynamicScatter


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create a boolean mask indicating valid positions in a padded tensor.

    Given the actual number of valid entries per sample (e.g. per voxel),
    generate a mask of shape ``(N, max_num)`` where ``True`` marks valid
    positions and ``False`` marks padding positions.

    Args:
        actual_num (torch.Tensor): Actual number of valid entries per sample,
            shape ``(N,)``.
        max_num (int): Maximum number of entries (the padded length).
        axis (int): Axis along which to insert the new dimension when tiling
            ``actual_num``. Defaults to 0, producing an intermediate shape of
            ``(N, 1)``.

    Returns:
        torch.Tensor: Boolean mask of shape ``(N, max_num)``.

    Example:
        ``actual_num = [3, 4, 2]`` and ``max_num = 5`` produces::

            [[True, True, True, False, False],
             [True, True, True, True,  False],
             [True, True, False, False, False]]
    """
    # Expand actual counts to (N, 1) so they can be compared against
    # the range [0, max_num) broadcasted to (N, max_num).
    actual_num = torch.unsqueeze(actual_num, axis + 1)

    # Build [0, 1, ..., max_num - 1] and reshape to allow broadcasting.
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int,
                           device=actual_num.device).view(max_num_shape)

    # A position is valid if its column index is strictly less than the
    # actual count for that row.
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator

class PFNLayer(nn.Module):
    """Pillar Feature Net Layer.

    The Pillar Feature Net is composed of a series of these layers, but the
    PointPillars paper results only used a single PFNLayer.

    Input layout (``forward``):
        - ``inputs``: ``(N, M, C_in)`` where ``N`` is the number of voxels,
          ``M`` is the max points per voxel (padded), and ``C_in`` is the
          number of input point feature channels.

    Output layout (``forward``):
        - If ``last_layer`` is ``True``: ``(N, 1, C_out)`` aggregated pillar
          features.
        - If ``last_layer`` is ``False``: ``(N, M, C_out)`` concatenation of
          per-point features and the aggregated pillar feature broadcast back
          to per-point resolution. ``C_out`` equals ``out_channels``.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        last_layer (bool, optional): If last_layer, there is no
            concatenation of features. Defaults to False.
        mode (str, optional): Pooling model to gather features inside voxels.
            Defaults to 'max'.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 last_layer=False,
                 mode='max'):

        super().__init__()
        self.fp16_enabled = False
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(in_channels, self.units, bias=False)

        assert mode in ['max', 'avg']
        self.mode = mode

    def forward(self, inputs, num_voxels=None, aligned_distance=None):
        """Forward function.

        Args:
            inputs (torch.Tensor): Pillar/Voxel inputs with shape
                ``(N, M, C_in)``. ``N`` is the number of voxels, ``M`` is the
                max points per voxel (padded), and ``C_in`` is the number of
                input point feature channels.
            num_voxels (torch.Tensor, optional): Number of valid points in
                each voxel, shape ``(N,)``. Required when ``mode`` is ``'avg'``.
                Defaults to None.
            aligned_distance (torch.Tensor, optional): Per-point distance to
                the voxel center, shape ``(N, M)``. Defaults to None.

        Returns:
            torch.Tensor: Pillar features. Shape is ``(N, 1, C_out)`` when
            ``last_layer`` is ``True``, otherwise ``(N, M, C_out)``.
        """
        # inputs: (N, M, C_in) -> x: (N, M, units)
        x = self.linear(inputs)
        # x: (N, M, units) -> (N, units, M) -> (N, M, units)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # x: (N, M, units)
        x = F.gelu(x)

        # x_max: (N, 1, units)
        if self.mode == 'max':
            if aligned_distance is not None:
                # aligned_distance: (N, M) -> (N, M, 1)
                x = x.mul(aligned_distance.unsqueeze(-1))
            x_max = torch.max(x, dim=1, keepdim=True)[0]
        elif self.mode == 'avg':
            if aligned_distance is not None:
                x = x.mul(aligned_distance.unsqueeze(-1))
            # num_voxels: (N,) -> (N, 1, 1)
            x_max = x.sum(dim=1, keepdim=True) / num_voxels.type_as(inputs).view(-1, 1, 1)

        if self.last_vfe:
            # x_max: (N, 1, units)
            return x_max
        else:
            # x_repeat: (N, M, units)
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            # x: (N, M, units), x_repeat: (N, M, units)
            # x_concatenated: (N, M, 2 * units)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated

class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, output_shape):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]
        self.in_channels = in_channels
        self.fp16_enabled = False

    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)

    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel.
        """
        # Create the canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        indices = coors[:, 1] * self.nx + coors[:, 2]
        indices = indices.long()
        voxels = voxel_features.t()
        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels
        # Undo the column stacking to final 4-dim tensor
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return canvas

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
                                         self.nx)

        return batch_canvas
    
class PillarFeatureNet(nn.Module):
    """Pillar Feature Net for PointPillars-style voxel encoding.

    Takes a batch of pillars (voxels), decorates each point with hand-crafted
    geometric features, and compresses each pillar into a single feature vector
    via shared Point Feature Network (PFN) layers.

    Input layout:
        - ``features``: ``(N, M, C_in)`` where
          ``N`` = number of non-empty pillars,
          ``M`` = max points per pillar (padded),
          ``C_in`` = raw point channels (3 for xyz, 4 for xyzr, etc.).
        - ``num_points``: ``(N,)`` actual point count in each pillar.
        - ``coors``: ``(N, 3)`` voxel coordinates in ``(z, y, x)`` order.

    Output layout:
        - ``pillars``: ``(N, C_out)`` where ``C_out`` is the last value in
          ``feat_channels``.

    Args:
        in_channels (int): Number of raw input channels per point, e.g. 3 for
            ``(x, y, z)`` or 4 for ``(x, y, z, r)``.
        feat_channels (tuple): Output channels of each PFNLayer. The last value
            becomes the final pillar feature dimension.
        with_distance (bool): Whether to append the Euclidean distance of each
            point to the origin as an extra channel.
        with_cluster_center (bool): Whether to append the offset from each point
            to the mean of its pillar.
        with_voxel_center (bool): Whether to append the offset from each point
            to the geometric center of its pillar.
        voxel_size (tuple[float]): Size of a voxel in ``(x, y, z)``.
        point_cloud_range (tuple[float]): Point cloud range
            ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
        mode (str): Aggregation mode in PFNLayer, ``'max'`` or ``'avg'``.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 mode='max'):
        super(PillarFeatureNet, self).__init__()
        assert len(feat_channels) > 0
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(in_filters,
                         out_filters,
                         last_layer=last_layer,
                         mode=mode))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range

    def forward(self, features, num_points, coors):
        """Encode pillars into fixed-size feature vectors.

        Args:
            features (torch.Tensor): Raw point features, shape ``(N, M, C_in)``.
                ``N`` = pillars, ``M`` = max points per pillar (padded),
                ``C_in`` = raw input channels.
            num_points (torch.Tensor): Actual number of points per pillar,
                shape ``(N,)``.
            coors (torch.Tensor): Voxel coordinates for each pillar, shape
                ``(N, 3)`` in ``(z, y, x)`` order.

        Returns:
            torch.Tensor: Per-pillar feature vectors, shape ``(N, C_out)``,
            where ``C_out`` is the last channel size in ``feat_channels``.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            f_center = torch.zeros_like(features[:, :, :3])
            f_center[:, :, 0] = features[:, :, 0] - (
                coors[:, 2].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
            f_center[:, :, 1] = features[:, :, 1] - (
                coors[:, 1].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
            f_center[:, :, 2] = features[:, :, 2] - (
                coors[:, 0].to(dtype).unsqueeze(1) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        # mask: (N, M, 1) broadcasts to (N, M, C), zeroing all channels of padding points.
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features, num_points)

        return features.squeeze(1)

class DynamicPillarFeatureNet(PillarFeatureNet):
    """Pillar Feature Net using dynamic voxelization.

    Unlike ``PillarFeatureNet`` which operates on padded pillars of fixed
    maximum size, this version operates on the actual set of points per voxel.
    It scatters points into voxels, aggregates voxel features, and then maps
    those features back to points.

    Input layout (``forward``):
        - ``features``: ``(M, C_in)`` where ``M`` is the total number of valid
          points across all pillars. ``C_in`` is the raw point channel count;
          it is typically ``3`` (``x, y, z``) or ``4`` (``x, y, z`` +
          ``intensity``), and the first three channels must be spatial
          coordinates.
        - ``coors``: ``(M, 3)`` voxel coordinates for **each point** in
          ``(z, y, x)`` order.

    Output layout (``forward``):
        - ``voxel_feats``: ``(N, C_out)`` aggregated voxel/pillar features.
        - ``voxel_coors``: ``(N, 3)`` unique voxel coordinates.
        - ``point_feats``: ``(M, C_out)`` per-point features after PFN.

    Here ``N`` is the number of unique non-empty voxels and ``M`` is the total
    number of valid points.

    Args:
        in_channels (int): Number of raw input channels per point.
        voxel_size (tuple[float]): Voxel size in ``(x, y, z)``.
        point_cloud_range (tuple[float]): Point cloud range
            ``(x_min, y_min, z_min, x_max, y_max, z_max)``.
        feat_channels (tuple): Output channels of each PFNLayer.
        with_distance (bool): Whether to append point-to-origin distance.
        with_cluster_center (bool): Whether to append offset to voxel mean.
        with_voxel_center (bool): Whether to append offset to voxel center.
        mode (str): Aggregation mode, ``'max'`` or ``'avg'``.
    """

    def __init__(self,
                 in_channels,
                 voxel_size,
                 point_cloud_range,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 mode='max'):
        super(DynamicPillarFeatureNet,
              self).__init__(in_channels,
                             feat_channels,
                             with_distance,
                             with_cluster_center=with_cluster_center,
                             with_voxel_center=with_voxel_center,
                             voxel_size=voxel_size,
                             point_cloud_range=point_cloud_range,
                             mode=mode)
        self.fp16_enabled = False
        feat_channels = [self.in_channels] + list(feat_channels)
        pfn_layers = []
        # TODO: currently only support one PFNLayer

        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False),
                    nn.BatchNorm1d(out_filters, eps=1e-3, momentum=0.01),
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.pfn_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(voxel_size,
                                              point_cloud_range,
                                              average_points=True)

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map the centers of voxels to its corresponding points.

        Args:
            pts_coors (torch.Tensor): The coordinates of each points, shape
                ``(M, 3)``. The first column is the batch index and the
                remaining two columns are the spatial voxel coordinates
                (e.g. ``(y, x)`` for pillars). ``M`` is the number of points.
            voxel_mean (torch.Tensor): The mean or aggregated features of a
                voxel, shape ``(N, C)``, where ``N`` is the number of voxels.
            voxel_coors (torch.Tensor): The coordinates of each voxel, shape
                ``(N, 3)``. The first column is the batch index and the
                remaining two columns are the spatial voxel coordinates.
        Returns:
            torch.Tensor: Corresponding voxel centers of each points, shape
                ``(M, C)``, where ``M`` is the number of points.
        """
        if pts_coors.shape[0] == 0:
            return torch.zeros((0, voxel_mean.shape[1]),
                               dtype=voxel_mean.dtype,
                               device=voxel_mean.device)
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        assert voxel_mean.shape[0] == voxel_coors.shape[
            0], f"voxel_mean.shape[0] {voxel_mean.shape[0]} != voxel_coors.shape[0] {voxel_coors.shape[0]}"
        assert pts_coors.shape[
            1] == 3, f"pts_coors.shape[1] {pts_coors.shape[1]} != 3"
        assert voxel_coors.shape[
            1] == 3, f"voxel_coors.shape[1] {voxel_coors.shape[1]} != 3"
        # 体素网格size
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        canvas_channel = voxel_mean.size(1) # 特征通道
        batch_size = pts_coors[:, 0].max() + 1 # z值当成batch_size

        canvas_len = canvas_y * canvas_x * batch_size # 画布体素总数
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        # Only include non-empty pillars
        indices = (voxel_coors[:, 0] * canvas_y * canvas_x +
                   voxel_coors[:, 2] * canvas_x + voxel_coors[:, 1])
        assert indices.long().max() < canvas_len, 'Index out of range'
        assert indices.long().min() >= 0, 'Index out of range'
        # Scatter the blob back to the canvas
        canvas[:, indices.long()] = voxel_mean.t()
        # Step 2: get voxel mean for each point
        voxel_index = (pts_coors[:, 0] * canvas_y * canvas_x +
                       pts_coors[:, 2] * canvas_x + pts_coors[:, 1])
        assert voxel_index.long().max() < canvas_len, 'Index out of range'
        assert voxel_index.long().min() >= 0, 'Index out of range'
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point

    def forward(self, features, coors):
        """Encode dynamically voxelized points into voxel and point features.

        Args:
            features (torch.Tensor): Raw point features, shape ``(M, C_in)``.
                ``M`` is the total number of valid points across all voxels.
                ``C_in`` is typically ``3`` (``x, y, z``) or ``4``
                (``x, y, z`` + ``intensity``); the first three channels must
                be spatial coordinates.
            coors (torch.Tensor): Per-point voxel coordinates, shape ``(M, 3)``
                in ``(z, y, x)`` order.

        Returns:
            tuple of three tensors:

            - ``voxel_feats`` (Tensor): Aggregated voxel features, shape
              ``(N, C_out)``.
            - ``voxel_coors`` (Tensor): Unique voxel coordinates, shape
              ``(N, 3)`` in ``(z, y, x)`` order.
            - ``point_feats`` (Tensor): Per-point features after PFN, shape
              ``(M, C_out)``.

            ``N`` is the number of unique non-empty voxels and ``M`` is the
            total number of input points.
        """
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 2].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 1].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                coors[:, 0].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        for i, pfn in enumerate(self.pfn_layers):
            # 用 PFN 对逐点特征进行升维编码。
            point_feats = pfn(features)
            # 按体素坐标聚合逐点特征，得到稀疏体素特征。
            voxel_feats, voxel_coors = self.pfn_scatter(point_feats, coors)
            if i != len(self.pfn_layers) - 1:
                # 非最后一层 PFN 时，把聚合后的体素特征映射回每个点，
                # 与当前逐点特征拼接后送入下一层。
                # 注意：coors / point_feats 是逐点的（M 条），而 voxel_feats /
                # voxel_coors 是聚合后的体素级（N 条，N <= M），因此需要用
                # map_voxel_center_to_point 按体素坐标把 N 条体素特征广播回 M 个点。
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        return voxel_feats, voxel_coors, point_feats

class HardVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range,
                 max_points_per_voxel: int):
        super().__init__()
        assert max_points_per_voxel > 0, f"max_points_per_voxel must be > 0, got {max_points_per_voxel}"

        self.voxelizer = Voxelization(voxel_size,
                                      point_cloud_range,
                                      max_points_per_voxel,
                                      deterministic=False)

    def forward(self, points: torch.Tensor):
        assert isinstance(
            points,
            torch.Tensor), f"points must be a torch.Tensor, got {type(points)}"
        not_nan_mask = ~torch.isnan(points).any(dim=2)
        return {"voxel_coords": self.voxelizer(points[not_nan_mask])}

class DynamicVoxelizer(nn.Module):

    def __init__(self, voxel_size, point_cloud_range):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxelizer = Voxelization(voxel_size,
                                      point_cloud_range,
                                      max_num_points=-1)

    def _get_point_offsets(self, points: torch.Tensor,
                           voxel_coords: torch.Tensor):
        """Compute offset from each point to the center of its voxel.

        Args:
            points: Point coordinates, shape ``(M, C)`` where ``C >= 3``.
            voxel_coords: Voxel coordinates returned by MMCV's ``Voxelization``,
                shape ``(M, 3)`` in ``(z, y, x)`` order.

        Returns:
            Per-point offset ``(M, 3)`` in ``(x, y, z)`` order.
        """
        point_cloud_range = torch.tensor(self.point_cloud_range,
                                         dtype=points.dtype,
                                         device=points.device)
        min_point = point_cloud_range[:3]
        voxel_size = torch.tensor(self.voxel_size,
                                  dtype=points.dtype,
                                  device=points.device)

        # MMCV returns voxel coords in (z, y, x); swap to (x, y, z) for geometry.
        voxel_coords = voxel_coords[:, [2, 1, 0]]

        # Voxel center = min_corner + coord * voxel_size + half_voxel
        voxel_centers = voxel_coords * voxel_size + min_point + voxel_size / 2

        return points[:, :3] - voxel_centers

    def _concatenate_batch_results(self, voxel_info_list):
        voxel_info_dict = dict()
        # concatenate keys of voxel_info_list for all batches
        for k in voxel_info_list[0].keys():
            if k != 'voxel_coords':
                voxel_info_dict[k] = torch.cat([item[k] for item in voxel_info_list], dim=0)
            else:
                coors_batch = []
                for i in range(len(voxel_info_list)):
                    coor_pad = nn.functional.pad(voxel_info_list[i][k], (1, 0), mode='constant', value=i)
                    coors_batch.append(coor_pad)
                voxel_info_dict[k] = torch.cat(coors_batch, dim=0).long()
        return voxel_info_dict

    def _split_batch_results(self, batch_voxel_info_dict):
        voxel_info_list = []

        bsz = len(batch_voxel_info_dict['voxel_coords'][:,0].unique())
        for i in range(bsz):
            voxel_info_dict = dict()
            batch_mask = batch_voxel_info_dict['voxel_coords'][:,0] == i
            for k in batch_voxel_info_dict.keys():
                voxel_info_dict[k] = batch_voxel_info_dict[k][batch_mask]
            voxel_info_list.append(voxel_info_dict)
        return voxel_info_list

    def _split_results(self, voxel_info_dict):
        full_voxel_info_list = []

        for j in range(len(voxel_info_dict['indicator'].unique())):
            indicator_mask = voxel_info_dict['indicator'] == j
            bsz = len(voxel_info_dict['voxel_coords'][indicator_mask,0].unique())
            voxel_info_list = []
            for i in range(bsz):
                info_dict = dict()
                batch_mask = voxel_info_dict['voxel_coords'][indicator_mask,0] == i
                for k in voxel_info_dict.keys():
                    info_dict[k] = voxel_info_dict[k][indicator_mask][batch_mask]
                voxel_info_list.append(info_dict)
            full_voxel_info_list.append(voxel_info_list)
        return full_voxel_info_list

    def forward(
            self,
            points: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Voxelize a batch of point clouds and compute per-point voxel offsets.

        Args:
            points: Input point cloud batch, shape ``(B, N, C)`` where ``C >= 3``.
                Padding points are expected to be filled with ``NaN``.

        Returns:
            A list of length ``B``. Each element is a dict with:

            - ``points`` (Tensor): Valid (non-NaN, in-range) points, shape ``(M, C)``.
            - ``voxel_coords`` (Tensor): Voxel coordinates in ``(z, y, x)`` order,
              shape ``(M, 3)``. See ``_get_point_offsets`` for the axis swap.
            - ``point_idxes`` (Tensor): Original indices of the kept points in the
              input point cloud, shape ``(M,)``.
            - ``point_offsets`` (Tensor): Offset from each point to its voxel center,
              shape ``(M, 3)``.

            ``M`` is the number of points that are both non-NaN and inside the
            configured ``point_cloud_range``.
        """
        batch_results = []
        for batch_idx in range(len(points)):
            batch_points = points[batch_idx]  # (N, C)

            # Track original point indices before any filtering.
            valid_point_idxes = torch.arange(batch_points.shape[0], device=batch_points.device)  # (N,)

            # Filter out padding points (NaN).
            not_nan_mask = ~torch.isnan(batch_points).any(dim=1)  # (N,)
            batch_non_nan_points = batch_points[not_nan_mask]  # (M1, C)
            valid_point_idxes = valid_point_idxes[not_nan_mask]  # (M1,)

            # Convert points to voxel coordinates. MMCV returns (z, y, x) order.
            # Out-of-range points are marked with -1.
            batch_voxel_coords = self.voxelizer(batch_non_nan_points)  # (M1, 3) in ZYX

            # Discard points that fall outside the voxel grid.
            batch_voxel_coords_mask = (batch_voxel_coords != -1).all(dim=1)  # (M1,)
            valid_batch_voxel_coords = batch_voxel_coords[batch_voxel_coords_mask]  # (M, 3)
            valid_batch_non_nan_points = batch_non_nan_points[batch_voxel_coords_mask]  # (M, C)
            valid_point_idxes = valid_point_idxes[batch_voxel_coords_mask]  # (M,)

            # Compute offset from each point to the center of its assigned voxel.
            point_offsets = self._get_point_offsets(valid_batch_non_nan_points,
                                                    valid_batch_voxel_coords)  # (M, 3)

            result_dict = {
                "points": valid_batch_non_nan_points,
                "voxel_coords": valid_batch_voxel_coords,
                "point_idxes": valid_point_idxes,
                "point_offsets": point_offsets
            }

            batch_results.append(result_dict)
        return batch_results

class DynamicEmbedder(nn.Module):

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        self.scatter = PointPillarsScatter(in_channels=feat_channels,
                                           output_shape=pseudo_image_dims)

    def forward(self, points: torch.Tensor) -> torch.Tensor:

        # List of points and coordinates for each batch
        voxel_info_list = self.voxelizer(points)

        pseudoimage_lst = []
        for voxel_info_dict in voxel_info_list:
            points = voxel_info_dict['points']
            coordinates = voxel_info_dict['voxel_coords']
            voxel_feats, voxel_coors, _ = self.feature_net(points, coordinates)
            pseudoimage = self.scatter(voxel_feats, voxel_coors)
            pseudoimage_lst.append(pseudoimage)
        # Concatenate the pseudoimages along the batch dimension
        return torch.cat(pseudoimage_lst, dim=0), voxel_info_list