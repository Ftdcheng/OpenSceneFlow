"""
Shared SeFlow / SeFlow++ cluster loss loop.
"""
from __future__ import annotations

import torch

from ._teflow_chamfer import TRUNCATED_DIST


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
