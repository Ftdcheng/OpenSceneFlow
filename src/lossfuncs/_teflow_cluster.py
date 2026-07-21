"""
TeFlow cluster-level motion model selection.

Implements the straight-line vs rigid/Kabsch branch used inside
``multi_frames_clusterLoss``.
"""
from __future__ import annotations

import torch
from typing import Optional

from ._teflow_chamfer import get_time_delta, TRUNCATED_DIST, DELTA_T


# ---- straight-line motion detector -------------------------------------------

def _is_straight_line_motion(
    flows: torch.Tensor,
    chamfer_dists: Optional[torch.Tensor] = None,
    cos_linear: float = 0.90,
    min_cos: float = 0.70,
    norm_cv_thresh: float = 0.20,
    chamfer_thresh: float = TRUNCATED_DIST,
    min_candidates: int = 3,
    min_motion: float = 1e-3,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Cheap straight-line motion detector for a cluster's candidate flows.

    This is the gating function used before switching from the linear/uniform
    motion branch to the rigid/Kabsch branch in TeFlow's cluster loss.  It
    treats a cluster as "linear" only when all top-k nearest-neighbor flow
    candidates agree in both direction and magnitude, and the underlying
    Chamfer matches are trustworthy.

    Args:
        flows (Tensor): Candidate flow vectors, shape ``(C, 3)``.  ``C`` is
            typically ``n_aux_frames * top_k`` after time-direction alignment.
        chamfer_dists (Tensor, optional): Chamfer distances corresponding to
            each candidate, shape ``(C,)``.  If ``None``, the Chamfer term is
            ignored.
        cos_linear (float): Mean pairwise cosine threshold.  Higher means
            stricter direction consensus.
        min_cos (float): Minimum pairwise cosine threshold.  Catches outlier
            frames that would otherwise be averaged out by ``mean_cos``.
        norm_cv_thresh (float): Maximum allowed coefficient of variation of
            flow magnitudes, ``std(|f|) / mean(|f|)``.
        chamfer_thresh (float): Maximum allowed mean Chamfer distance.  The
            default ``TRUNCATED_DIST`` matches the squared-distance truncation
            used elsewhere (sqrt(4) = 2m at 10Hz).
        min_candidates (int): If ``C < min_candidates`` the cluster is declared
            non-linear so that the rigid branch is preferred for small samples.
        min_motion (float): If the mean flow magnitude is below this value the
            cluster is considered nearly static and declared linear (avoids
            paying for an expensive rigid branch on noise).
        eps (float): Numerical epsilon.

    Returns:
        tuple[Tensor, dict]:

        - ``is_linear`` (BoolTensor): Scalar tensor, ``True`` if the candidate
          set is consistent with straight-line uniform motion.
        - ``info`` (dict): Diagnostic scalars with keys ``mean_cos``,
          ``min_cos``, ``norm_cv``, ``mean_chamfer``, ``mean_mag``,
          ``n_candidates``, ``low_motion``.
    """
    C = flows.shape[0]
    device = flows.device
    info = {
        'mean_cos': torch.tensor(0.0, device=device),
        'min_cos': torch.tensor(0.0, device=device),
        'norm_cv': torch.tensor(float('inf'), device=device),
        'mean_chamfer': torch.tensor(0.0, device=device),
        'mean_mag': torch.tensor(0.0, device=device),
        'n_candidates': C,
        'low_motion': False,
    }

    if C < min_candidates:
        return torch.tensor(False, device=device), info

    mags = torch.linalg.vector_norm(flows, dim=-1)  # (C,)
    mean_mag = mags.mean()
    info['mean_mag'] = mean_mag

    # Nearly static: linear branch is good enough and avoids rigid-branch noise.
    if mean_mag < min_motion:
        info['low_motion'] = True
        return torch.tensor(True, device=device), info

    # Direction consistency from pairwise cosine similarity.
    unit = flows / (mags.unsqueeze(-1) + eps)  # (C, 3)
    cos_sim = torch.matmul(unit, unit.transpose(-2, -1))  # (C, C)
    triu_idx = torch.triu_indices(C, C, offset=1, device=device)
    pair_cos = cos_sim[triu_idx[0], triu_idx[1]]
    mean_cos = pair_cos.mean()
    min_cos_val = pair_cos.min()
    info['mean_cos'] = mean_cos
    info['min_cos'] = min_cos_val

    # Magnitude stability: coefficient of variation of flow norms.
    norm_cv = mags.std() / (mean_mag + eps)
    info['norm_cv'] = norm_cv

    # Match quality: average Chamfer distance (squared).
    if chamfer_dists is not None and chamfer_dists.numel() > 0:
        mean_chamfer = chamfer_dists.mean()
    else:
        mean_chamfer = torch.tensor(0.0, device=device)
    info['mean_chamfer'] = mean_chamfer

    is_linear = (
        mean_cos > cos_linear and
        min_cos_val > min_cos and
        norm_cv < norm_cv_thresh and
        mean_chamfer < chamfer_thresh
    )
    return is_linear, info

# ---- rigid/Kabsch branch helpers ---------------------------------------------

def _kabsch_transform(src: torch.Tensor, tgt: torch.Tensor, n_sample: int = 256, eps: float = 1e-6):
    """Estimate rigid ``(R, t)`` aligning ``src -> tgt`` via NN correspondences.

    Both point clouds may have different sizes.  We first down-sample ``src`` if
    it is too large, find the nearest neighbor in ``tgt`` for each sampled
    source point, and then run Kabsch/SVD on the resulting correspondences.

    Args:
        src (Tensor): Source points, shape ``(N, 3)``.
        tgt (Tensor): Target points, shape ``(M, 3)``.
        n_sample (int): Maximum number of source points used for the SVD step.
        eps (float): Numerical epsilon.

    Returns:
        tuple[Tensor, Tensor] or None: ``(R, t)`` with ``R`` shape ``(3, 3)`` and
        ``t`` shape ``(3,)``.  Returns ``None`` if the inputs are too small or
        the SVD is degenerate.
    """
    if src.shape[0] < 3 or tgt.shape[0] < 3:
        return None

    # Sub-sample the source cluster to keep SVD cost bounded.
    N = src.shape[0]
    if N > n_sample:
        idx = torch.randperm(N, device=src.device)[:n_sample]
        src_s = src[idx]
    else:
        src_s = src

    # Nearest-neighbor correspondences from sampled source to target.
    dists = torch.cdist(src_s, tgt)            # (S, M)
    nn_idx = dists.argmin(dim=1)               # (S,)
    tgt_corr = tgt[nn_idx]                     # (S, 3)

    src_c = src_s.mean(dim=0)
    tgt_c = tgt_corr.mean(dim=0)

    H = (src_s - src_c).T @ (tgt_corr - tgt_c)  # (3, 3)
    try:
        U, S, Vh = torch.linalg.svd(H)
    except RuntimeError:
        return None

    R = Vh.T @ U.T
    # Reflection correction.
    if torch.det(R) < 0:
        Vh[-1, :] *= -1
        R = Vh.T @ U.T

    t = tgt_c - R @ src_c
    return R, t


def _match_cluster_in_frame(
    frame_id: str,
    sample_idx: int,
    res_dict: dict,
    nn_idx_topk: torch.Tensor,
    tgt_labels_topk: torch.Tensor,
    c_center_pred: Optional[torch.Tensor] = None,
    vote_conf_thresh: float = 0.6,
    match_radius: float = 5.0,
):
    """Find the target-cluster label in one auxiliary frame that best matches
    the current ``pc0`` cluster.

    First tries a majority vote over the labels of the top-k nearest-neighbor
    target points.  If the vote is inconclusive, falls back to nearest dynamic
    cluster center (using ``c_center_pred``) within ``match_radius``.

    Args:
        frame_id (str): Auxiliary frame key, e.g. ``'pch1'``.
        sample_idx (int): Batch sample index.
        res_dict (dict): Assembled SSL dictionary.
        nn_idx_topk (Tensor): Local indices of the top-k target points,
            shape ``(K,)``.
        tgt_labels_topk (Tensor): Labels of those top-k target points,
            shape ``(K,)``.
        c_center_pred (Tensor, optional): Predicted cluster center in the
            auxiliary frame's coordinate system, shape ``(3,)``.
        vote_conf_thresh (float): Minimum ratio of votes for the winning label.
        match_radius (float): Maximum center-to-center distance for the
            fallback nearest-cluster search.

    Returns:
        int or None: The matched target cluster label, or ``None`` if no
        reliable match is found.
    """
    # ---- Layer 1: majority vote on the top-k nearest target labels ----
    valid = tgt_labels_topk > 1
    matched_label = None
    if valid.sum() > 0:
        labels, counts = torch.unique(tgt_labels_topk[valid], return_counts=True)
        conf = counts.float() / valid.sum()
        best = conf.argmax()
        if conf[best] >= vote_conf_thresh:
            matched_label = labels[best].item()

    # ---- Layer 2: nearest dynamic cluster center ----------------------
    if matched_label is None and c_center_pred is not None:
        tgt_pc = res_dict[f'{frame_id}_list'][sample_idx]
        tgt_lab = res_dict[f'{frame_id}_labels_list'][sample_idx]
        centers, label_ids = [], []
        for l in torch.unique(tgt_lab):
            if l <= 1:
                continue
            centers.append(tgt_pc[tgt_lab == l].mean(dim=0))
            label_ids.append(l)
        if centers:
            centers = torch.stack(centers)                      # (M, 3)
            dists = torch.linalg.vector_norm(centers - c_center_pred, dim=-1)
            if dists.min() < match_radius:
                matched_label = label_ids[dists.argmin().item()].item()

    return matched_label


def _rigid_cluster_target_flow(
    p0_cluster: torch.Tensor,
    frame_data: list,
    res_dict: dict,
    sample_idx: int,
    net_avg: torch.Tensor,
    args: dict,
):
    """Compute per-point target flow for a non-linear cluster by matching to
    auxiliary-frame clusters and running Kabsch.

    Args:
        p0_cluster (Tensor): Current cluster points, shape ``(K_c, 3)``.
        frame_data (list): One dict per auxiliary frame, as collected in
            ``multi_frames_clusterLoss``.
        res_dict (dict): Assembled SSL dictionary.
        sample_idx (int): Batch sample index.
        net_avg (Tensor): Network's average flow estimate for this cluster,
            shape ``(3,)``.
        args (dict): Hyper-parameters.  Relevant keys:

            - ``cluster_match_vote_conf`` (float, default 0.6)
            - ``cluster_match_radius`` (float, default 5.0)
            - ``kabsch_sample`` (int, default 256)
            - ``kabsch_residual_thresh`` (float, default 2.0)
            - ``time_decay_factor`` (float, default 0.9)

    Returns:
        tuple[Tensor, bool]: ``(target_flow, ok)``.  ``target_flow`` has shape
        ``(K_c, 3)`` when ``ok`` is ``True``.
    """
    VOTE_CONF = args.get('cluster_match_vote_conf', 0.6)
    MATCH_R = args.get('cluster_match_radius', 5.0)
    KABSCH_SAMPLE = args.get('kabsch_sample', 256)
    RESID_THRESH = args.get('kabsch_residual_thresh', 2.0)
    TIME_DECAY = args.get('time_decay_factor', 0.9)

    flow_list, weight_list = [], []
    cluster_center = p0_cluster.mean(dim=0)
    net_avg_det = net_avg.detach()  # only used for center prediction

    for fd in frame_data:
        frame_id = fd['frame_id']
        factor = fd['factor']
        time_delta = fd['time_delta']

        # Predict where the cluster center should be in the auxiliary frame.
        # net_avg is per DELTA_T, so scale by the signed number of delta steps.
        c_center_pred = cluster_center + net_avg_det * (time_delta / DELTA_T)

        matched_label = _match_cluster_in_frame(
            frame_id, sample_idx, res_dict,
            fd['nn_idx'], fd['tgt_labels'],
            c_center_pred=c_center_pred,
            vote_conf_thresh=VOTE_CONF,
            match_radius=MATCH_R,
        )
        if matched_label is None:
            continue

        tgt_pc = res_dict[f'{frame_id}_list'][sample_idx]
        tgt_lab = res_dict[f'{frame_id}_labels_list'][sample_idx]
        tgt_cluster_pts = tgt_pc[tgt_lab == matched_label]
        if tgt_cluster_pts.shape[0] < 3:
            continue

        Rt = _kabsch_transform(p0_cluster, tgt_cluster_pts, n_sample=KABSCH_SAMPLE)
        if Rt is None:
            continue
        R, t = Rt

        pred_tgt = p0_cluster @ R.T + t
        # Chamfer-like residual from predicted points to the matched target cluster.
        residual = torch.cdist(pred_tgt, tgt_cluster_pts).min(dim=1).values.mean()
        if residual > RESID_THRESH:
            continue

        flow = pred_tgt - p0_cluster
        tw = pow(TIME_DECAY, factor)
        # Weight by time decay and inverse residual.
        w = tw / (residual + 1.0)
        flow_list.append(flow)
        weight_list.append(w)

    if not flow_list:
        return None, False

    weights = torch.stack(weight_list)
    weights = weights / (weights.sum() + 1e-6)
    target_flow = torch.stack([f * w for f, w in zip(flow_list, weights)]).sum(dim=0)
    return target_flow, True


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
            - ``linear_cos_linear`` (float, default 0.90): Mean cosine threshold
              for the straight-line motion detector.
            - ``linear_min_cos`` (float, default 0.70): Minimum pairwise cosine
              threshold for the straight-line motion detector.
            - ``linear_norm_cv_thresh`` (float, default 0.20): Maximum flow
              magnitude coefficient of variation for linear motion.
            - ``linear_chamfer_thresh`` (float, default ``TRUNCATED_DIST``):
              Maximum mean Chamfer distance for linear motion.
            - ``cluster_match_vote_conf`` (float, default 0.6): Confidence
              threshold for top-k cluster-label voting in the rigid branch.
            - ``cluster_match_radius`` (float, default 5.0): Maximum center
              distance for the fallback nearest-cluster search.
            - ``kabsch_sample`` (int, default 256): Source point sub-sample size
              used inside the Kabsch SVD.
            - ``kabsch_residual_thresh`` (float, default 2.0): Maximum mean
              nearest-neighbor residual (in meters) for a Kabsch fit to be kept.

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

            p0_cluster = p0[cluster_mask]            # (K_c, 3)
            ext_flows, ext_dists, ext_tw = [], [], []
            frame_data = []                            # 非直线分支复用
            # 从每个辅助帧中收集 TOP_K 个最近邻 flow 候选
            for frame_id in frame_keys:
                dist_c = frames_dists[frame_id][i][cluster_mask]   # (K_c,), pc0到frame_id对应帧的各点最近距离
                idx_c  = frames_indices[frame_id][i][cluster_mask] # (K_c,), pc0到frame_id对应帧的各点最近距离对应点id
                if dist_c.shape[0] <= TOP_K: # 凑不齐k个点就算了
                    continue

                # 取距离最小的 TOP_K 个（最近邻）
                topk_dists, topk_local = torch.topk(dist_c, k=TOP_K, largest=False)  # 均为 (K,)
                # 用局部 idx 在辅助帧中取出对应目标点
                target_pts = res_dict[f'{frame_id}_list'][i][idx_c[topk_local]]  # (K, 3)
                src_pts    = p0_cluster[topk_local]                              # (K, 3)
                time_delta, factor = get_time_delta(frame_id)
                # Eq. 3：把目标帧上的位移转换到与网络输出一致的 flow 方向/尺度
                flows = (target_pts - src_pts) / factor * (-1 if time_delta < 0 else 1)  # (K, 3)
                ext_flows.append(flows)
                ext_dists.append(topk_dists)                                       # (K,)
                ext_tw.append(torch.full((TOP_K,), pow(TIME_DECAY, factor), device=p0.device))

                # 为非直线刚体分支保存匹配所需信息
                tgt_labels = res_dict[f'{frame_id}_labels_list'][i][idx_c[topk_local]]
                frame_data.append({
                    'frame_id': frame_id,
                    'factor': factor,
                    'time_delta': time_delta,
                    'nn_idx': idx_c[topk_local],
                    'tgt_labels': tgt_labels,
                })

            if not ext_flows: # 每个帧都没k个点
                continue

            # Eq. 2：网络对该簇的平均 flow 估计
            net_avg = cluster_flows.mean(dim=0)      # (3,)

            # 直线运动检测：用所有辅助帧的候选 flow 判断
            all_cand_flows = torch.cat(ext_flows, dim=0)    # (C, 3)
            all_cand_dists = torch.cat(ext_dists, dim=0)    # (C,)
            is_linear, _ = _is_straight_line_motion(
                all_cand_flows, all_cand_dists,
                cos_linear=args.get('linear_cos_linear', 0.90),
                min_cos=args.get('linear_min_cos', 0.70),
                norm_cv_thresh=args.get('linear_norm_cv_thresh', 0.20),
                chamfer_thresh=args.get('linear_chamfer_thresh', TRUNCATED_DIST),
            )

            if is_linear.item():
                # ---- 直线分支：保持原有 RANSAC 投票 ----
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
            else:
                # ---- 非直线分支：簇匹配 + Kabsch ----
                target_flow, ok = _rigid_cluster_target_flow(
                    p0_cluster, frame_data, res_dict, i, net_avg, args
                )
                if not ok:
                    continue

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
