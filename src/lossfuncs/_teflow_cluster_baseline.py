"""Original TeFlow cluster-loss baseline.

This module contains the unmodified ``multi_frames_clusterLoss`` implementation
that was previously part of ``selfsupervise.py``.  It is kept as a clean A/B
baseline for the ReFlow variant in ``_teflow_cluster.py``.
"""
from __future__ import annotations

import torch

from ._teflow_chamfer import get_time_delta


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
