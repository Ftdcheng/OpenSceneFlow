"""
# Created: 2024-11-15 21:33
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of 
# * DeltaFlow (https://github.com/Kin-Zhang/DeltaFlow)
# * OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True
from .encoder import DynamicVoxelizer, DynamicPillarFeatureNet
import dztimer

class SparseVoxelNet(nn.Module):
    """DeltaFlow 的多帧体素差分稀疏编码器。

    以未来帧 ``pc1`` 为参考锚点，将当前帧 ``pc0s`` 和历史帧 ``pch*`` 体素化到
    同一 ``point_cloud_range`` 下，然后按体素坐标逐体素计算
    ``pc1 - pcx`` 的特征差分，并按时间衰减加权累加，最终输出一个稀疏的
    delta 特征张量，供后续 3D 稀疏卷积使用。

    Args:
        voxel_size (list): 体素大小 ``[vx, vy, vz]``。
        pseudo_image_dims (list): 体素空间形状 ``[H, W, D]``。
        point_cloud_range (list): 点云范围 ``[x_min, y_min, z_min,
            x_max, y_max, z_max]``。
        feat_channels (int): 每个体素输出的特征通道数 ``C_out``。
        decay_factor (float): 历史帧差分的衰减系数。时间越早的帧权重越低，
            权重为 ``decay_factor ** time_index``。
        timer (dztimer.Timing, optional): 外部计时器，用于性能分析。
    """

    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int, decay_factor=1.0, timer=None) -> None:
        super().__init__()
        self.voxelizer = DynamicVoxelizer(voxel_size=voxel_size,
                                          point_cloud_range=point_cloud_range)
        self.feature_net = DynamicPillarFeatureNet(
            in_channels=3,
            feat_channels=(feat_channels, ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            mode='avg')
        
        self.voxel_spatial_shape = pseudo_image_dims
        self.num_feature = feat_channels
        self.decay_factor = decay_factor
        if timer is None:
            self.timer = dztimer.Timing()
            self.timer.start("Total")
        else:
            self.timer = timer

    def process_batch(self, voxel_info_list, if_return_point_feats=False):
        """把 ``DynamicVoxelizer`` 输出的逐 batch 点云转换为可构造稀疏张量的体素特征与坐标。

        处理流程：
            1. 对每个 batch 调用 ``DynamicPillarFeatureNet`` 提取体素特征。
            2. 给体素坐标补上 batch 维度，并把顺序从 ``(z, y, x)`` 转为 ``(x, y, z)``。
            3. 把所有 batch 的结果沿第 0 维拼接，得到跨 batch 的稀疏表示。

        Args:
            voxel_info_list (list[dict]): 长度为 ``B`` 的列表。每个元素包含
                ``points`` (``(M_i, C_in)``) 和 ``voxel_coords`` (``(M_i, 3)``)，
                分别表示该 batch 的有效点及其体素坐标（顺序 ``(z, y, x)``）。
            if_return_point_feats (bool): 是否同时返回逐点特征。默认 ``False``。

        Returns:
            tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list]:
            - ``voxel_feats_sp``: 所有 batch 的体素特征，形状 ``(N_total, C_out)``。
              ``N_total = sum(N_i)`` 是当前 batch 里所有非空体素的总数。
            - ``coors_batch_sp``: 对应的完整坐标，形状 ``(N_total, 4)``，格式为
              ``[batch_idx, x, y, z]``。
            - ``point_feats_lst``（可选）: 每个 batch 的逐点特征列表，第 ``i`` 项形状
              ``(M_i, C_out)``。

        Note:
            ``torch.cat`` 要求被拼接张量的维度数相同，且除拼接维度外其余维度大小一致。
            这里所有 ``voxel_feats`` 都是 ``(N_i, C_out)``，所有 ``voxel_coors_batch``
            都是 ``(N_i, 4)``，因此可以沿 ``dim=0`` 拼接。
        """
        voxel_feats_list_batch = []
        voxel_coors_list_batch = []
        point_feats_lst = []

        for batch_index, voxel_info_dict in enumerate(voxel_info_list):
            points = voxel_info_dict['points']  # (M_i, C_in)，当前 batch 的有效点
            coordinates = voxel_info_dict['voxel_coords']  # (M_i, 3)，顺序 (z, y, x)

            # 提取体素级特征、对应体素坐标和逐点特征。
            # 其中 voxel_coors 与 voxel_feats 按行一一对应，均为 (N_i, *)，
            # N_i 是当前 batch 的非空体素数，满足 N_i <= M_i。
            voxel_feats, voxel_coors, point_feats = self.feature_net(
                points, coordinates)

            if if_return_point_feats:
                point_feats_lst.append(point_feats)

            # 构造 batch 索引列，形状 (N_i, 1)。
            batch_indices = torch.full(
                (voxel_coors.size(0), 1),
                batch_index,
                dtype=torch.long,
                device=voxel_coors.device)

            # 把坐标顺序从 (z, y, x) 调整为 (x, y, z)，再在前面拼接 batch 索引，
            # 得到 spconv / sparse_coo_tensor 期望的 [batch_idx, x, y, z] 格式。
            voxel_coors_batch = torch.cat(
                [batch_indices, voxel_coors[:, [2, 1, 0]]], dim=1)

            voxel_feats_list_batch.append(voxel_feats)
            voxel_coors_list_batch.append(voxel_coors_batch)

        # 所有 batch 的体素特征沿第 0 维拼接：要求每个张量都是 (N_i, C_out)。
        voxel_feats_sp = torch.cat(voxel_feats_list_batch, dim=0)
        coors_batch_sp = torch.cat(voxel_coors_list_batch, dim=0).to(
            dtype=torch.int32)

        if if_return_point_feats:
            return voxel_feats_sp, coors_batch_sp, point_feats_lst

        return voxel_feats_sp, coors_batch_sp
    
    def forward(self, input_dict) -> torch.Tensor:
        """Compute multi-frame voxel difference features for DeltaFlow.

        Args:
            input_dict (dict): 包含多帧点云的字典。必须包含以下键：

                - ``pc0s`` (Tensor): 当前帧点云，形状 ``(B, N, C)``。
                - ``pc1s`` (Tensor): 未来/参考帧点云，形状 ``(B, N, C)``。
                - ``pch1s``, ``pch2s``, ... (Tensor, optional): 历史帧点云，
                  形状 ``(B, N, C)``。这些帧应当已经 warp 到 ``pc1`` 的坐标系下。

                所有张量的 padding 点都应填充 ``NaN``，由 ``DynamicVoxelizer``
                自动过滤。

        Returns:
            dict: 包含以下键：

                - ``delta_sparse`` (spconv.SparseConvTensor): 最终的体素差分
                  稀疏张量，特征维度为 ``C_out``。
                - ``pc0_3dvoxel_infos_lst`` / ``pc1_3dvoxel_infos_lst`` /
                  ``pch1_3dvoxel_infos_lst`` (list[dict]): 对应帧经过体素化后的
                  逐 batch 信息列表。
                - ``pc0_point_feats_lst`` (list[Tensor]): 当前帧 ``pc0s`` 的逐点
                  特征，每元素形状 ``(M_i, C_out)``。
                - ``pc0_num_voxels`` / ``pc1_num_voxels`` (int): 对应帧的非空
                  体素总数。
                - ``d_num_voxels`` (int): 最终 ``delta_sparse`` 的非空体素数。
        """
        bz_ = len(input_dict['pc0s'])

        # 历史帧键名形如 pch1s, pch2s ... 按序号降序排列，最后追加 pc0s。
        # reversed 后实际处理顺序为：pc0s -> pch1s -> pch2s（由近到远）。
        frame_keys = sorted(
            [key for key in input_dict.keys() if key.startswith('pch')],
            reverse=True)
        frame_keys += ['pc0s']

        # ---------- 1. 处理参考帧 pc1 ----------
        # pc1 作为所有差分的锚点，只需体素化一次。
        pc1_voxel_info_list = self.voxelizer(input_dict['pc1s'])
        pc1_voxel_feats_sp, pc1_coors_batch_sp = self.process_batch(
            pc1_voxel_info_list)
        pc1s_num_voxels = pc1_voxel_feats_sp.shape[0]

        # 完整稀疏张量的形状：[B, H, W, D, C_out]。
        sparse_max_size = [bz_, *self.voxel_spatial_shape, self.num_feature]

        # pc1 的稀疏特征张量。
        sparse_pc1 = torch.sparse_coo_tensor(
            pc1_coors_batch_sp.t(), pc1_voxel_feats_sp, size=sparse_max_size)

        # 用 pc1 的坐标初始化一个零值差分张量，后续逐帧累加 (pc1 - pcx)。
        sparse_diff = torch.sparse_coo_tensor(
            pc1_coors_batch_sp.t(),
            pc1_voxel_feats_sp * 0.0,
            size=sparse_max_size)

        pch1s_3dvoxel_infos_lst = None
        pc0_point_feats_lst = []

        # ---------- 2. 逐帧累加差分 ----------
        for time_index, frame_key in enumerate(reversed(frame_keys)):
            self.timer[0].start("Point Feature Voxelize")
            pc = input_dict[frame_key]
            voxel_info_list = self.voxelizer(pc)

            # 仅对当前帧 pc0s 保留逐点特征，供后续解码使用；历史帧只取体素特征。
            if frame_key == 'pc0s':
                voxel_feats_sp, coors_batch_sp, pc0_point_feats_lst = \
                    self.process_batch(voxel_info_list, if_return_point_feats=True)
            else:
                voxel_feats_sp, coors_batch_sp = self.process_batch(
                    voxel_info_list)

            # 当前帧/历史帧的稀疏张量。
            sparse_pcx = torch.sparse_coo_tensor(
                coors_batch_sp.t(), voxel_feats_sp, size=sparse_max_size)

            # 按体素坐标对齐，计算 pc1 与该帧的差分，并按时间衰减加权累加。
            # 越早的帧 time_index 越大，权重越低（decay_factor ** time_index）。
            sparse_diff = sparse_diff + \
                (sparse_pc1 - sparse_pcx) * pow(self.decay_factor, time_index)
            self.timer[0].stop()

            # 保存当前帧和历史帧的体素化信息，供后续使用。
            if frame_key == 'pc0s':
                pc0s_3dvoxel_infos_lst = voxel_info_list
                pc0s_num_voxels = voxel_feats_sp.shape[0]
            elif frame_key == 'pch1s':
                pch1s_3dvoxel_infos_lst = voxel_info_list

        # ---------- 3. 平均并转成 spconv 稀疏张量 ----------
        self.timer[2].start("D_Delta_Sparse")

        # coalesce 合并同一坐标的多个累加值，再除以参与差分的帧数得到平均差分。
        # time_index 在循环结束后等于 len(frame_keys) - 1，因此除以 time_index + 1。
        # 注意：这里与论文实现一致，是按“帧数”做简单平均，而不是按 decay^t 的
        # 权重和做归一化。decay_factor 只负责对不同时间步的差分进行缩放。
        features = sparse_diff.coalesce().values() / (time_index + 1)
        indices = sparse_diff.coalesce().indices().t().to(dtype=torch.int32)

        all_pcdiff_sparse = spconv.SparseConvTensor(
            features.contiguous(),
            indices.contiguous(),
            self.voxel_spatial_shape,
            bz_)
        self.timer[2].stop()

        output = {
            'delta_sparse': all_pcdiff_sparse,
            'pch1_3dvoxel_infos_lst': pch1s_3dvoxel_infos_lst,
            'pc0_3dvoxel_infos_lst': pc0s_3dvoxel_infos_lst,
            'pc0_point_feats_lst': pc0_point_feats_lst,
            'pc0_num_voxels': pc0s_num_voxels,
            'pc1_3dvoxel_infos_lst': pc1_voxel_info_list,
            'pc1_num_voxels': pc1s_num_voxels,
            'd_num_voxels': indices.shape[0]
        }
        return output

class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=0, indice_key=None):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(inc, outc, kernel_size=ks, stride=stride, dilation=dilation, padding=padding, bias=False, \
                              indice_key=indice_key, algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(outc),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, indice_key, ks=3):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseInverseConv3d(inc, outc, kernel_size=ks, indice_key=indice_key, bias=False, algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(outc),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, padding=0):
        super().__init__()
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(inc, outc, kernel_size=ks, stride=stride, dilation=dilation, padding=padding, bias=False, \
                                algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(outc),
            nn.ReLU(inplace=True),
            spconv.SubMConv3d(outc, outc, kernel_size=ks, stride=stride, dilation=dilation, padding=padding, bias=False, \
                                algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(outc)
        )

        if inc == (outc * self.expansion) and stride == 1:
            self.downsample = None
        else:
            self.downsample = spconv.SparseSequential(
                spconv.SubMConv3d(inc, outc, kernel_size=1, dilation=1,
                                stride=stride, algo=spconv.ConvAlgo.Native),
                nn.BatchNorm1d(outc)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x.features
        out = self.net(x)
        if self.downsample is not None:
            identity = self.downsample(x).features
        out = out.replace_feature(out.features + identity)
        out = out.replace_feature(self.relu(out.features))

        return out
    
'''
Reference when I wrote MinkUNet:
* https://github.com/PJLab-ADG/OpenPCSeg/blob/master/pcseg/model/segmentor/voxel/minkunet/minkunet.py
* https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/backbones/minkunet_backbone.py
* https://github.com/mit-han-lab/spvnas/blob/master/core/models/semantic_kitti/minkunet.py
'''
class MinkUNet(nn.Module):
    def __init__(self, 
                 cs=[16, 32, 64, 128, 256, 256, 128, 64, 32, 16], 
                 num_layer=[2, 2, 2, 2, 2, 2, 2, 2, 2]):
        super().__init__()
        
        inc = cs[0]
        cs = cs[1:] # remove the first input channel after conv_input
        self.block = ResidualBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(inc, cs[0], kernel_size=3, stride=1, padding=1, bias=False, \
                                indice_key="subm0", algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(cs[0]),
            nn.ReLU(inplace=True),

            spconv.SubMConv3d(cs[0], cs[0], kernel_size=3, stride=1, padding=1, bias=False, \
                                indice_key="subm0", algo=spconv.ConvAlgo.Native),
            nn.BatchNorm1d(cs[0]),
            nn.ReLU(inplace=True)
        )
        self.in_channels = cs[0]

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, indice_key="subm1"),
            *self._make_layer(self.block, cs[1], num_layer[0])
        )
        # inside every make_layer: self.in_channels = out_channels * block.expansion
        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, indice_key="subm2"),
            *self._make_layer(self.block, cs[2], num_layer[1])
        )
        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, indice_key="subm3"),
            *self._make_layer(self.block, cs[3], num_layer[2])
        )
        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(self.in_channels, self.in_channels, ks=2, stride=2, indice_key="subm4"),
            *self._make_layer(self.block, cs[4], num_layer[3])
        )

        self.up1 = [BasicDeconvolutionBlock(self.in_channels, cs[5], ks=2, indice_key="subm4")]
        self.in_channels = cs[5] + cs[3] * self.block.expansion
        self.up1.append(nn.Sequential(*self._make_layer(self.block, cs[5], num_layer[4])))
        self.up1 = nn.ModuleList(self.up1)

        self.up2 = [BasicDeconvolutionBlock(cs[5], cs[6], ks=2, indice_key="subm3")]
        self.in_channels = cs[6] + cs[2] * self.block.expansion
        self.up2.append(nn.Sequential(*self._make_layer(self.block, cs[6], num_layer[5])))
        self.up2 = nn.ModuleList(self.up2)

        self.up3 = [BasicDeconvolutionBlock(cs[6], cs[7], ks=2, indice_key="subm2")]
        self.in_channels = cs[7] + cs[1] * self.block.expansion
        self.up3.append(nn.Sequential(*self._make_layer(self.block, cs[7], num_layer[6])))
        self.up3 = nn.ModuleList(self.up3)

        self.up4 = [BasicDeconvolutionBlock(cs[7], cs[8], ks=2, indice_key="subm1")]
        self.in_channels = cs[8] + cs[0] * self.block.expansion
        self.up4.append(nn.Sequential(*self._make_layer(self.block, cs[8], num_layer[7])))
        self.up4 = nn.ModuleList(self.up4)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride=stride)
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(
                block(self.in_channels, out_channels)
            )
        return layers
    
    def forward(self, x):
        x = self.conv_input(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = y1.replace_feature(torch.cat([y1.features, x3.features], dim=1))
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = y2.replace_feature(torch.cat([y2.features, x2.features], dim=1))
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = y3.replace_feature(torch.cat([y3.features, x1.features], dim=1))
        y3 = self.up3[1](y3) # Dense shape: [B, C, X, Y, Z]; [B, 32, 256, 256, 16]
        
        y4 = self.up4[0](y3)
        y4 = y4.replace_feature(torch.cat([y4.features, x.features], dim=1))
        y4 = self.up4[1](y4)

        return y4