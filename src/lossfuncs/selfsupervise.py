"""
Self-supervised loss function entry points.

This module exposes:

- ``teflowLoss``: multi-frame TeFlow loss (chamfer + static + cluster RANSAC/rigid).
- ``seflowppLoss``: bidirectional SeFlow++ loss.
- ``seflowLoss``: single-frame SeFlow loss.

Implementation details live in the ``_teflow_*`` and ``_seflow_*`` helper modules.
"""
from __future__ import annotations

import torch
from assets.cuda.chamfer3D import nnChamferDis

_MyCUDAChamferDis = nnChamferDis()

from ._teflow_chamfer import batched_chamfer_related as _batched_chamfer_related
from ._teflow_chamfer import TRUNCATED_DIST as _TRUNCATED_DIST
from ._teflow_cluster import multi_frames_clusterLoss as _multi_frames_cluster_loss
from ._seflow_cluster import _seflow_cluster_loop


# from paper: https://arxiv.org/abs/2602.19053
def teflowLoss(res_dict, timer=None):
    """Temporal seflow: chamfer over all frames + static + RANSAC/rigid cluster loss."""
    pc0_list     = res_dict['pc0_list']
    flow_list    = res_dict['est_flow_list']
    pc0_lab_list = res_dict['pc0_labels_list']

    chamfer_dis, dynamic_chamfer_dis, frame_keys = _batched_chamfer_related(res_dict, timer)

    static_loss = torch.tensor(0.0, device=pc0_list[0].device)
    for fv, lab in zip(flow_list, pc0_lab_list):
        if (lab == 0).any():
            static_loss += torch.linalg.vector_norm(fv[lab == 0], dim=-1).mean()
    static_loss /= max(len(pc0_list), 1)

    cluster_weight = res_dict['loss_weights_dict'].get('cluster_based_pc0pc1', 0.0)
    if cluster_weight > 0:
        frames_dists, frames_indices = {}, {}
        for frame_id in frame_keys:
            d_list, i_list = _MyCUDAChamferDis.batched_disid_res(
                pc0_list, res_dict[f'{frame_id}_list'],
            )
            frames_dists[frame_id]   = d_list
            frames_indices[frame_id] = i_list

        moved_cluster_loss = _multi_frames_cluster_loss(
            pc0_list, pc0_lab_list, flow_list,
            frame_keys, frames_dists, frames_indices, res_dict,
            res_dict.get('cluster_loss_args', {}),
        )
    else:
        moved_cluster_loss = torch.tensor(0.0, device=pc0_list[0].device)

    return {
        'chamfer_dis':          chamfer_dis,
        'dynamic_chamfer_dis':  dynamic_chamfer_dis,
        'static_flow_loss':     static_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }


# from paper: https://arxiv.org/abs/2503.00803
def seflowppLoss(res_dict, timer=None):
    """seflow++ loss: bidirectional (pc1 + pch1) chamfer + cluster, B samples."""
    pc0_list      = res_dict['pc0_list']
    pc1_list      = res_dict['pc1_list']
    pch1_list     = res_dict['pch1_list']
    flow_list     = res_dict['est_flow_list']
    pc0_lab_list  = res_dict['pc0_labels_list']
    pc1_lab_list  = res_dict['pc1_labels_list']
    pch1_lab_list = res_dict['pch1_labels_list']
    dev           = pc0_list[0].device

    fwd_list  = [p0 + fv for p0, fv in zip(pc0_list, flow_list)]
    bwd_list  = [p0 - fv for p0, fv in zip(pc0_list, flow_list)]

    chamfer_dis  = _MyCUDAChamferDis(fwd_list, pc1_list,  truncate_dist=_TRUNCATED_DIST)
    chamfer_dis += _MyCUDAChamferDis(bwd_list, pch1_list, truncate_dist=_TRUNCATED_DIST)

    dyn_fwd, dyn_pc1   = [], []
    dyn_bwd, dyn_pch1  = [], []
    for fwd_i, bwd_i, p1_i, ph1_i, lab0_i, lab1_i, labh1_i in zip(
            fwd_list, bwd_list, pc1_list, pch1_list,
            pc0_lab_list, pc1_lab_list, pch1_lab_list):
        dyn_mask = lab0_i > 0
        if dyn_mask.sum() > 256:
            dp1 = p1_i[lab1_i > 0]
            dph = ph1_i[labh1_i > 0]
            if dp1.shape[0]  > 256: dyn_fwd.append(fwd_i[dyn_mask]); dyn_pc1.append(dp1)
            if dph.shape[0]  > 256: dyn_bwd.append(bwd_i[dyn_mask]); dyn_pch1.append(dph)

    dynamic_chamfer_dis = torch.tensor(0.0, device=dev)
    if len(dyn_fwd) >= 1:
        dynamic_chamfer_dis += _MyCUDAChamferDis(dyn_fwd, dyn_pc1, truncate_dist=_TRUNCATED_DIST)
    if len(dyn_bwd) >= 1:
        dynamic_chamfer_dis += _MyCUDAChamferDis(dyn_bwd, dyn_pch1, truncate_dist=_TRUNCATED_DIST)

    dist0_list, idx0_list = _MyCUDAChamferDis.batched_disid_res(pc0_list, pc1_list)
    static_loss, moved_cluster_loss = _seflow_cluster_loop(
        pc0_list, pc1_list, pc0_lab_list, pc1_lab_list,
        flow_list, dist0_list, idx0_list,
    )

    return {
        'chamfer_dis':          chamfer_dis / 2.0,
        'dynamic_chamfer_dis':  dynamic_chamfer_dis / 2.0,
        'static_flow_loss':     static_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }


# from paper: https://arxiv.org/abs/2407.01702
def seflowLoss(res_dict, timer=None):
    """seflow loss: single future frame (pc1), batched over B samples."""
    pc0_list     = res_dict['pc0_list']
    pc1_list     = res_dict['pc1_list']
    flow_list    = res_dict['est_flow_list']
    pc0_lab_list = res_dict['pc0_labels_list']
    pc1_lab_list = res_dict['pc1_labels_list']
    dev          = pc0_list[0].device

    fwd_list = [p0 + fv for p0, fv in zip(pc0_list, flow_list)]

    chamfer_dis = _MyCUDAChamferDis(fwd_list, pc1_list, truncate_dist=_TRUNCATED_DIST)

    dyn_fwd, dyn_pc1 = [], []
    for fwd_i, p1_i, lab0_i, lab1_i in zip(fwd_list, pc1_list, pc0_lab_list, pc1_lab_list):
        dp1 = p1_i[lab1_i > 0]
        if (lab0_i > 0).sum() > 256 and dp1.shape[0] > 256:
            dyn_fwd.append(fwd_i[lab0_i > 0])
            dyn_pc1.append(dp1)

    dynamic_chamfer_dis = torch.tensor(0.0, device=dev)
    if len(dyn_fwd) >= 1:
        dynamic_chamfer_dis = _MyCUDAChamferDis(dyn_fwd, dyn_pc1, truncate_dist=_TRUNCATED_DIST)

    dist0_list, idx0_list = _MyCUDAChamferDis.batched_disid_res(pc0_list, pc1_list)
    static_loss, moved_cluster_loss = _seflow_cluster_loop(
        pc0_list, pc1_list, pc0_lab_list, pc1_lab_list,
        flow_list, dist0_list, idx0_list,
    )

    return {
        'chamfer_dis':          chamfer_dis,
        'dynamic_chamfer_dis':  dynamic_chamfer_dis,
        'static_flow_loss':     static_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }
