"""
TeFlow multi-frame Chamfer utilities.
"""
from __future__ import annotations

import torch
from assets.cuda.chamfer3D import nnChamferDis

MyCUDAChamferDis = nnChamferDis()

# NOTE(Qingwen 24/07/06): squared, so it's sqrt(4) = 2m, in 10Hz the vel = 20m/s ~ 72km/h
# If your scenario is different, may need adjust this TRUNCATED to 80-120km/h vel.
TRUNCATED_DIST = 4

# FIXME(Qingwen 25-07-21): hardcoded 10 Hz. Adjust for datasets with different timestamps.
DELTA_T = 0.1  # seconds


# ---- helpers -----------------------------------------------------------------

def get_time_delta(frame_id):
    """Return (time_delta, factor).
    pch1->(-0.1,1), pch2->(-0.2,2), pc1->(+0.1,1), pc2->(+0.2,2)
    """
    if frame_id.startswith('pch'):
        n = int(frame_id[3:]) if len(frame_id) > 3 else 1
        return -DELTA_T * n, n
    elif frame_id.startswith('pc'):
        n = int(frame_id[2:]) if len(frame_id) > 2 else 1
        return DELTA_T * n, n
    raise ValueError(f"Unknown frame ID: {frame_id}")


def _frame_keys(res_dict):
    """Auxiliary frame ids present in res_dict (e.g. ['pc1', 'pch1']), excluding pc0."""
    return [k.replace('_list', '') for k in res_dict
            if k.endswith('_list') \
                and k != 'pc0_list' and k != 'est_flow_list' and not k.endswith('_labels_list')]


# ---- helpers shared by teflow* -----------------------------------------------

def batched_chamfer_related(res_dict, timer=None):
    """Compute Chamfer and dynamic-Chamfer distances from pc0 to all auxiliary frames.

    For every auxiliary frame key present in ``res_dict`` (e.g. ``'pc1'``,
    ``'pch1'``, ``'pch2'``), this function projects the source point cloud
    ``pc0`` using the predicted flow scaled by the temporal distance to that
    frame, then measures the Chamfer distance against the target point cloud.
    A separate dynamic-only Chamfer term is also computed on points labelled as
    dynamic in both the source and target frames.

    Args:
        res_dict (dict): Input dictionary assembled by ``ssl_loss_calculator``.
            Required keys:

            - ``pc0_list`` (list[Tensor]): Length ``B``; ``pc0_list[i]`` has
              shape ``(N_i, 3)`` and contains the source points for sample ``i``.
            - ``est_flow_list`` (list[Tensor]): Length ``B``;
              ``est_flow_list[i]`` has shape ``(N_i, 3)`` and contains the
              predicted flow from ``pc0`` over one ``DELTA_T`` interval.
            - ``pc0_labels_list`` (list[Tensor]): Length ``B``;
              ``pc0_labels_list[i]`` has shape ``(N_i,)`` with dynamic labels for
              ``pc0`` points (``0`` for static, ``>0`` for dynamic).
            - ``{frame_id}_list`` (list[Tensor]): One per auxiliary frame, each
              of length ``B``; ``{frame_id}_list[i]`` has shape ``(M_i, 3)``.
            - ``{frame_id}_labels_list`` (list[Tensor]): One per auxiliary frame,
              each of length ``B``; ``{frame_id}_labels_list[i]`` has shape
              ``(M_i,)``.
            - ``loss_weights_dict`` (dict): Weights for ``chamfer_dis`` and
              ``dynamic_chamfer_dis``.

        timer (dztimer.Timing, optional): Optional timer for profiling.

    Returns:
        tuple[Tensor, Tensor, list[str]]:

        - ``total_chamfer_dis`` (Tensor): Scalar, mean Chamfer distance over all
          auxiliary frames. Each frame's contribution is first weighted by
          ``1.0`` for ``pc1`` or ``1.0 / 2^factor`` for past frames, then
          averaged by the number of auxiliary frames ``n``.
        - ``total_dynamic_chamfer_dis`` (Tensor): Scalar, mean dynamic Chamfer
          distance over all auxiliary frames. Computed only when
          ``dynamic_chamfer_dis`` weight is positive and at least one sample has
          more than 256 dynamic points in both source and target frames;
          otherwise it stays zero.
        - ``frame_keys`` (list[str]): Length ``n``, the auxiliary frame IDs
          processed (e.g. ``['pc1', 'pch1', 'pch2']``).

    Shape notes:
        - ``B`` is the batch size.
        - ``N_i`` is the number of valid source points in sample ``i``.
        - ``M_i`` is the number of valid target points in sample ``i`` for a
          given auxiliary frame.
        - ``proj_list`` has length ``B``; ``proj_list[i]`` has shape
          ``(N_i, 3)``.
        - ``proj_dyn`` / ``tgt_dyn`` are lists of length up to ``B``; each
          included tensor has shape ``(K_i, 3)`` where ``K_i <= N_i`` and
          ``K_i > 256``.
    """
    pc0_list      = res_dict['pc0_list']
    flow_list     = res_dict['est_flow_list']
    pc0_lab_list  = res_dict['pc0_labels_list']
    frame_keys    = _frame_keys(res_dict)
    loss_w        = res_dict['loss_weights_dict']
    chamfer_w     = loss_w.get('chamfer_dis', 0.0)
    dyn_chamfer_w = loss_w.get('dynamic_chamfer_dis', 0.0)

    total_chamfer_dis       = torch.tensor(0.0, device=pc0_list[0].device)
    total_dynamic_chamfer_dis = torch.tensor(0.0, device=pc0_list[0].device)

    for frame_id in frame_keys:
        time_delta, factor = get_time_delta(frame_id)
        weight      = 1.0 if frame_id == 'pc1' else 1.0 / pow(2, factor)
        target_list = res_dict[f'{frame_id}_list'] # 目标点云帧（历史帧或者未来帧）

        # 将每个批量当前帧投影到各帧（历史帧，未来帧）
        # Projected positions: list comprehension keeps everything per-sample
        proj_list = [p0 + (fv / DELTA_T) * time_delta
                     for p0, fv in zip(pc0_list, flow_list)]
        
        # 计算投影全点云到目标帧的chamfer距离
        if chamfer_w > 0:
            total_chamfer_dis += MyCUDAChamferDis(
                proj_list, target_list, truncate_dist=TRUNCATED_DIST * factor
            ) * weight

        if dyn_chamfer_w <= 0:
            continue

        # 将目标点云和当前帧在目标点云上的投影的动态部分提取出来
        # 动态部分的点数太少就不考虑
        tgt_lab_list = res_dict[f'{frame_id}_labels_list']
        proj_dyn, tgt_dyn = [], []
        for proj_i, p0_lab_i, tgt_i, tgt_lab_i in zip(
                proj_list, pc0_lab_list, target_list, tgt_lab_list):
            dp = proj_i[p0_lab_i > 0]
            dt = tgt_i[tgt_lab_i > 0]
            if dp.shape[0] > 256 and dt.shape[0] > 256:
                proj_dyn.append(dp)
                tgt_dyn.append(dt)
                
        # 如果存在动态点足够的点云就算一下投影和目标帧点云的chamfer距离
        if len(proj_dyn) >= 1:
            total_dynamic_chamfer_dis += MyCUDAChamferDis(
                proj_dyn, tgt_dyn, truncate_dist=TRUNCATED_DIST * factor
            ) * weight

    # 多帧取平均
    n = len(frame_keys)
    if n > 0:
        total_chamfer_dis       /= n
        total_dynamic_chamfer_dis /= n

    return total_chamfer_dis, total_dynamic_chamfer_dis, frame_keys
