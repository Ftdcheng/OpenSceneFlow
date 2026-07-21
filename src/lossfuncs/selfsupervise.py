"""
# Created: 2023-07-17 00:00
# Updated: 2025-08-07 00:01
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
# 
# This file is part of 
# * SeFlow (https://github.com/KTH-RPL/SeFlow)
# * HiMo (https://kin-zhang.github.io/HiMo)
#
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
#
# Description: Self-supervised loss functions.
#
# All losses receive a unified dict from ssl_loss_calculator (trainer.py).
# Every frame is represented only as a List[Tensor] — no flat/offsets/sizes.
#
#   res_dict keys (per frame 'pc0', 'pc1', 'pch1', ...):
#     '{frame}_list'   : List[Tensor (N_i,3)]  one tensor per sample
#     '{frame}_labels' : List[Tensor (N_i,)]   one label vector per sample
#
#   'est_flow_list'    : List[Tensor (N_i,3)]
#   'batch_size'       : int
#   'loss_weights_dict': dict  (teflow* only)
#   'cluster_loss_args': dict  (teflowLoss only)
"""
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

# ---- multi-frame cluster loss (teflow) -------------------
# Based on TeFlow paper: https://arxiv.org/abs/2602.19053
def multi_frames_clusterLoss(
    pc0_list, pc0_lab_list, flow_list,
    frame_keys, frames_dists, frames_indices, res_dict, args={}
):
    """RANSAC-weighted cluster consistency loss across multiple temporal frames (TeFlow Eq. 2-9).

    For every dynamic cluster (label > 1) in every sample of the batch, this
    function gathers ``top_k_candidates`` nearest-neighbor flow hypotheses from
    each auxiliary frame, combines them with the network's own average estimate,
    and selects a consensus target flow via weighted RANSAC voting. The final
    loss has a point-level MSE term plus a cluster-level mean-residual term.

    Args:
        pc0_list (list[Tensor]): Source point clouds, length ``B``.
            ``pc0_list[i]`` has shape ``(N_i, 3)``.
        pc0_lab_list (list[Tensor]): Per-point cluster labels, length ``B``.
            ``pc0_lab_list[i]`` has shape ``(N_i,)``. Labels ``<= 1`` are skipped;
            labels ``> 1`` identify dynamic clusters.
        flow_list (list[Tensor]): Predicted scene flow, length ``B``.
            ``flow_list[i]`` has shape ``(N_i, 3)``.
        frame_keys (list[str]): Auxiliary frame IDs, length ``n``
            (e.g. ``['pc1', 'pch1', 'pch2']``).
        frames_dists (dict[str, list[Tensor]]): Nearest-neighbor distances from
            each ``pc0`` point to each auxiliary frame. Outer dict has ``n`` keys.
            ``frames_dists[frame_id]`` is a list of length ``B``;
            ``frames_dists[frame_id][i]`` has shape ``(N_i,)``.
        frames_indices (dict[str, list[Tensor]]): Local nearest-neighbor indices
            into each auxiliary frame. Same nested list structure as
            ``frames_dists``; values are local indices into
            ``res_dict[f'{frame_id}_list'][i]``.
        res_dict (dict): Dictionary assembled by ``ssl_loss_calculator``. Must
            contain ``f'{frame_id}_list'`` for every ``frame_id`` in
            ``frame_keys``; each such value is a list of length ``B`` with tensors
            of shape ``(M_i, 3)``.
        args (dict, optional): Hyper-parameters.

            - ``top_k_candidates`` (int, default 5): ``K`` in the paper.
            - ``ransac_cos_threshold`` (float, default 0.7071): Cosine threshold
              for considering two candidate flows as inliers.
            - ``time_decay_factor`` (float, default 0.9): Temporal weight
              ``TIME_DECAY^factor`` for past frames.
            - ``network_estimate_weight`` (float, default 1.0): Weight for the
              network's own average flow estimate in voting.

    Returns:
        torch.Tensor: Scalar cluster consistency loss. If no valid cluster is
        found across the batch, returns a zero scalar on the same device as
        ``flow_list[0]``.

    Shape notes:
        - ``B``: batch size.
        - ``n``: number of auxiliary frames (``len(frame_keys)``).
        - ``N_i``: number of points in ``pc0_list[i]``.
        - ``M_i``: number of points in an auxiliary frame for sample ``i``.
        - ``K`` (``TOP_K``): number of nearest-neighbor candidates kept per
          cluster per auxiliary frame.
        - For cluster ``c`` in sample ``i``: ``cluster_mask`` has ``K_c`` True
          entries; ``cluster_flows`` has shape ``(K_c, 3)``.
        - ``dist_c`` / ``idx_c`` have shape ``(K_c,)``.
        - ``topk_dists`` / ``topk_local`` have shape ``(K,)`` (only when
          ``K_c > K``).
        - ``ext_flows`` has length ``<= n`` (one entry per auxiliary frame with
          enough points). Each tensor has shape ``(K, 3)``.
        - ``all_cands`` has shape ``(C, 3)`` where
          ``C = len(ext_flows) * K + 1`` (the extra one is the network average).
        - ``all_d`` has shape ``(C,)``; ``all_tw`` has shape
          ``(len(ext_flows) * K,)``.
        - ``scores`` has shape ``(C,)``.
        - ``all_cluster_flows`` / ``all_target_flows`` are lists with length equal
          to the total number of valid clusters across the whole batch; each
          element has shape ``(K_c, 3)``.
        - ``all_avg_losses`` is a list with the same cluster count; each element
          is a scalar Tensor.
    """
    TOP_K         = int(args.get('top_k_candidates', 5))
    COS_THRESH    = args.get('ransac_cos_threshold', 0.7071)
    TIME_DECAY    = args.get('time_decay_factor', 0.9)
    NET_EST_W     = args.get('network_estimate_weight', 1.0)

    # 长度为label数量（不重复的）。
    # 所有动态簇的flow估计（网络估计）、按照距离相似度投票估计的flow(假设整个簇都是同一个速度)、网络估计和簇投票估计的loss
    all_cluster_flows, all_target_flows, all_avg_losses = [], [], []

    # 遍历 batch 内每个样本：p0 (N_i,3), lab0 (N_i,), fv (N_i,3)
    for i, (p0, lab0, fv) in enumerate(zip(pc0_list, pc0_lab_list, flow_list)):
        # 只处理 label > 1 的动态簇（0 是静态，1 通常是非聚类噪声/背景）
        for label in torch.unique(lab0):
            if label <= 1:
                continue

            cluster_mask  = (lab0 == label)          # (N_i,), bool
            cluster_flows = fv[cluster_mask]         # (K_c, 3), K_c 为该簇点数

            ext_flows, ext_dists, ext_tw = [], [], []
            # 从每个辅助帧中收集 TOP_K 个最近邻 flow 候选
            for frame_id in frame_keys:
                dist_c = frames_dists[frame_id][i][cluster_mask]   # (K_c,), pc0到frame_id对应帧的各点最近距离
                idx_c  = frames_indices[frame_id][i][cluster_mask] # (K_c,), pc0到frame_id对应帧的各点最近距离对应点id
                if dist_c.shape[0] <= TOP_K: # 凑不齐k个点就算了
                    continue
                    
                # 获取topk个距离和对应的id，这个id是在pc0中的位置索引
                topk_dists, topk_local = torch.topk(dist_c, k=TOP_K)  # 均为 (K,)
                # 用局部 idx 在辅助帧中取出对应目标点
                target_pts = res_dict[f'{frame_id}_list'][i][idx_c[topk_local]]  # (K, 3)
                src_pts    = p0[cluster_mask][topk_local]                        # (K, 3)
                time_delta, factor = get_time_delta(frame_id)
                # Eq. 3：把目标帧上的位移转换到与网络输出一致的 flow 方向/尺度
                flows = (target_pts - src_pts) / factor * (-1 if time_delta < 0 else 1)  # (K, 3)
                ext_flows.append(flows)
                ext_dists.append(topk_dists)                                       # (K,)
                ext_tw.append(torch.full((TOP_K,), pow(TIME_DECAY, factor), device=p0.device))

            if not ext_flows: # 每个帧都没k个点
                continue

            # Eq. 2：网络对该簇的平均 flow 估计
            # 这个板块主要是把平均值给塞进去
            # 对这个簇的flow预测求平均值
            net_avg = cluster_flows.mean(dim=0)      # (3,)
            net_mag = torch.linalg.norm(net_avg)     # scalar
            # Eq. 4：拼接所有候选 flow（辅助帧候选 + 网络平均）
            all_cands = torch.cat(ext_flows + [net_avg.unsqueeze(0)], dim=0)  # (C, 3)
            all_d     = torch.cat(ext_dists + [net_mag.unsqueeze(0)], dim=0)  # (C,)
            all_tw    = torch.cat(ext_tw, dim=0)                              # (C-1,)
            if all_cands.shape[0] < 2:
                continue
            
            # 所有距离在点云内部进行归一化
            d_norm  = (all_d - all_d.min()) / (all_d.max() - all_d.min() + 1e-6)  # (C,)
            # Eq. 5：候选间余弦相似度，构造 inlier mask
            cos_sim = torch.nn.functional.cosine_similarity(
                all_cands[:, None, :], all_cands[None, :, :], dim=-1)  # (C, C)
            inlier  = cos_sim > COS_THRESH
            # Eq. 6：时间衰减权重 + 距离归一化 + 网络估计权重
            weights = torch.cat([all_tw * (1 + d_norm[:-1]),
                                  (NET_EST_W * (1 + d_norm[-1])).unsqueeze(0)])  # (C,)
            # Eq. 7：每个候选的 inlier 加权得分
            scores  = torch.matmul(inlier.float(), weights.unsqueeze(1)).squeeze()  # (C,)
            # 找出得分最高的候选id
            best    = torch.argmax(scores) 

            # Eq. 8：在最佳候选的 inlier 集合里做加权平均，得到该簇的 target flow
            # 找出最高分对应的支持者，对支持者进行加权
            inlier_flows = all_cands[inlier[best]]   # (L, 3)
            inlier_w     = weights[inlier[best]]     # (L,)
            denom = inlier_w.sum()
            target_flow = (inlier_w.unsqueeze(1) * inlier_flows).sum(dim=0) / denom \
                          if denom > 1e-6 else all_cands[best]  # (3,)

            # 收集该簇所有点的预测 flow 和对应 target flow
            all_cluster_flows.append(cluster_flows)                          # (K_c, 3)
            all_target_flows.append(target_flow.expand_as(cluster_flows))    # (K_c, 3)
            all_avg_losses.append(
                torch.linalg.vector_norm(cluster_flows - target_flow, dim=-1).mean()
            )

    # 没有任何有效簇时返回 0
    if not all_cluster_flows:
        return torch.tensor(0.0, device=flow_list[0].device)
    # Eq. 9：点级 MSE + 簇级平均残差
    # NOTE(Qingwen): Point-level term
    loss  = torch.nn.functional.mse_loss(
        torch.cat(all_cluster_flows), torch.cat(all_target_flows)
    )
    # NOTE(Qingwen): Cluster-level term
    loss += torch.stack(all_avg_losses).mean()
    return loss


# ---- shared cluster loop (seflow / seflowpp) -------------------
# SeFlow Paper: https://arxiv.org/pdf/2407.01702
def _seflow_cluster_loop(pc0_list, pc1_list, pc0_lab_list, pc1_lab_list,
                          flow_list, dist0_list, idx0_list):
    """Per-sample seflow cluster loss (Eq. 6-11).

    dist0_list, idx0_list : output of batched_disid_res(pc0_list, pc1_list)
    idx0_list[i] is LOCAL into pc1_list[i].
    Returns (static_cluster_loss, moved_cluster_loss, have_any_dynamic).
    """
    dev = flow_list[0].device
    static_loss    = torch.tensor(0.0, device=dev)
    cluster_norms  = []
    fallback_dists = []
    have_any_dyn   = False

    for p0, p1, lab0, lab1, fv, dist0, idx0 in zip(
            pc0_list, pc1_list, pc0_lab_list, pc1_lab_list,
            flow_list, dist0_list, idx0_list):
        have_dyn = (lab0 > 0).sum() > 256 and (lab1 > 0).sum() > 256
        if have_dyn:
            have_any_dyn = True
            fallback_dists.append(dist0)

        for label in torch.unique(lab0):
            mask = (lab0 == label)
            if label == 0:
                # Eq. 6 in the paper
                static_loss += torch.linalg.vector_norm(fv[mask], dim=-1).mean()
            elif label > 1 and have_dyn:
                c_flow = fv[mask]
                c_idx0 = idx0[mask]
                # Eq. 8 in the paper
                sorted_local = torch.argsort(dist0[mask], descending=True)
                max_idx = torch.nonzero(lab1[c_idx0[sorted_local]] > 0).squeeze(1)
                if max_idx.shape[0] == 0:
                    continue
                best     = sorted_local[max_idx[0]]
                # Eq. 9 in the paper
                max_flow = p1[c_idx0[best]] - p0[mask][best]
                # Eq. 10 in the paper
                cluster_norms.append(torch.linalg.vector_norm(c_flow - max_flow, dim=-1))

    if cluster_norms:
        # Eq. 11
        moved_loss = torch.cat(cluster_norms).mean()
    elif have_any_dyn:
        all_d = torch.cat(fallback_dists)
        moved_loss = torch.mean(all_d[all_d <= TRUNCATED_DIST])
    else:
        moved_loss = torch.tensor(0.0, device=dev)

    return static_loss, moved_loss

# from paper: https://arxiv.org/abs/2602.19053
def teflowLoss(res_dict, timer=None):
    """Temporal seflow: chamfer over all frames + static + RANSAC cluster loss."""
    pc0_list     = res_dict['pc0_list']         # 当前帧点云
    flow_list    = res_dict['est_flow_list']    # 预测flow
    pc0_lab_list = res_dict['pc0_labels_list']  # 动/静标签

    # flow投影和辅助帧的chamfer距离、它们动态部分到辅助帧相应部分的chamfer距离、辅助帧列表
    chamfer_dis, dynamic_chamfer_dis, frame_keys = batched_chamfer_related(res_dict, timer)

    # 计算静态loss
    static_loss = torch.tensor(0.0, device=pc0_list[0].device)
    for fv, lab in zip(flow_list, pc0_lab_list): # 对每个批次的flow预测求静态loss，找出静态的点，算他们的速度大小
        if (lab == 0).any():
            static_loss += torch.linalg.vector_norm(fv[lab == 0], dim=-1).mean()
    static_loss /= max(len(pc0_list), 1)

    cluster_weight = res_dict['loss_weights_dict'].get('cluster_based_pc0pc1', 0.0)
    if cluster_weight > 0:
        # 对每个辅助帧计算：pc0 到 辅助帧各点间的最小距离，及其索引。
        frames_dists, frames_indices = {}, {}
        for frame_id in frame_keys:
            # pc0各点 到 辅助帧各点 的最小距离，以及这个点在辅助帧中的索引
            d_list, i_list = MyCUDAChamferDis.batched_disid_res(
                pc0_list, res_dict[f'{frame_id}_list'],
            )
            frames_dists[frame_id]   = d_list
            frames_indices[frame_id] = i_list

        moved_cluster_loss = multi_frames_clusterLoss(
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

    # Chamfer: both temporal directions concurrently
    chamfer_dis  = MyCUDAChamferDis(fwd_list, pc1_list,  truncate_dist=TRUNCATED_DIST)
    chamfer_dis += MyCUDAChamferDis(bwd_list, pch1_list, truncate_dist=TRUNCATED_DIST)

    # Dynamic chamfer
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
        dynamic_chamfer_dis += MyCUDAChamferDis(dyn_fwd, dyn_pc1, truncate_dist=TRUNCATED_DIST)
    if len(dyn_bwd) >= 1:
        dynamic_chamfer_dis += MyCUDAChamferDis(dyn_bwd, dyn_pch1, truncate_dist=TRUNCATED_DIST)

    dist0_list, idx0_list = MyCUDAChamferDis.batched_disid_res(pc0_list, pc1_list)
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

    chamfer_dis = MyCUDAChamferDis(fwd_list, pc1_list, truncate_dist=TRUNCATED_DIST)

    # Dynamic chamfer
    dyn_fwd, dyn_pc1 = [], []
    for fwd_i, p1_i, lab0_i, lab1_i in zip(fwd_list, pc1_list, pc0_lab_list, pc1_lab_list):
        dp1 = p1_i[lab1_i > 0]
        if (lab0_i > 0).sum() > 256 and dp1.shape[0] > 256:
            dyn_fwd.append(fwd_i[lab0_i > 0])
            dyn_pc1.append(dp1)

    dynamic_chamfer_dis = torch.tensor(0.0, device=dev)
    if len(dyn_fwd) >= 1:
        dynamic_chamfer_dis = MyCUDAChamferDis(dyn_fwd, dyn_pc1, truncate_dist=TRUNCATED_DIST)

    dist0_list, idx0_list = MyCUDAChamferDis.batched_disid_res(pc0_list, pc1_list)
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