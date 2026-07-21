"""
Visualize TeFlow top-k Chamfer matches from a preprocessed HDF5 file.

This script does NOT require running training. It loads one frame (pc0), its
future frame (pc1) and optionally one past frame (pch1) from an HDF5 file,
removes ground points, warps all frames into the pc1 coordinate system,
computes nearest-neighbor distances/indices, and calls
``tools.vis_cluster_topk.vis_cluster_topk``.

Usage:
    conda run -n opensf python tools/vis_cluster_topk_from_h5.py \
        --h5_path /path/to/scene.h5 \
        --label 3 \
        --top_k 5 \
        --save_path debug/topk_scene_label3.png

If no CUDA Chamfer extension is available, the script falls back to
``scipy.spatial.cKDTree`` on CPU (slower but no compilation needed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

# Add repo root to path so that ``src.*`` and ``assets.*`` imports work.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.basic import wrap_batch_pcs
from tools.vis_cluster_topk import vis_cluster_topk


def _load_frame(group: h5py.Group, device: torch.device):
    """Load lidar, ground_mask, pose, and dynamic label from an HDF5 group."""
    pc = torch.from_numpy(group["lidar"][:][:, :3].astype(np.float32)).to(device)
    gm = torch.from_numpy(group["ground_mask"][:].astype(bool)).to(device)
    pose = torch.from_numpy(group["pose"][:].astype(np.float32)).to(device)

    # Dynamic cluster label: prefer dufocluster, fall back to cluster.
    if "dufocluster" in group:
        lab = torch.from_numpy(group["dufocluster"][:].astype(np.int64)).to(device)
    elif "cluster" in group:
        lab = torch.from_numpy(group["cluster"][:].astype(np.int64)).to(device)
    else:
        lab = torch.zeros(pc.shape[0], dtype=torch.long, device=device)

    return pc, gm, pose, lab


def _nearest_neighbor(
    src: torch.Tensor,
    tgt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-point nearest-neighbor distances and local indices.

    First tries the compiled CUDA Chamfer operator; falls back to
    scipy.spatial.cKDTree if the CUDA extension is unavailable.
    """
    try:
        from assets.cuda.chamfer3D import nnChamferDis
        chamfer = nnChamferDis()
        d_list, i_list = chamfer.batched_disid_res([src], [tgt])
        return d_list[0], i_list[0]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] CUDA Chamfer failed ({exc}), falling back to scipy cKDTree.")
        from scipy.spatial import cKDTree

        tgt_np = tgt.detach().cpu().numpy()
        src_np = src.detach().cpu().numpy()
        tree = cKDTree(tgt_np)
        dists, idxs = tree.query(src_np, k=1)
        return (
            torch.from_numpy(dists.astype(np.float32)).to(src.device),
            torch.from_numpy(idxs.astype(np.int64)).to(src.device),
        )


def visualize_topk_from_h5(
    h5_path: str,
    timestamp: Optional[str] = None,
    label: int = 3,
    top_k: int = 5,
    frame_keys: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Load one sample from HDF5 and visualize its cluster top-k matches."""
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        timestamps = sorted(f.keys(), key=int)
        if len(timestamps) < 2:
            raise ValueError("HDF5 must contain at least two timestamps")

        # Pick a timestamp with both previous and next frames available.
        if timestamp is None:
            idx = max(1, min(len(timestamps) - 2, len(timestamps) // 2))
            timestamp = timestamps[idx]
        else:
            timestamp = str(timestamp)
            if timestamp not in timestamps:
                raise ValueError(
                    f"Timestamp {timestamp} not found. Available: {timestamps[:5]}..."
                )
            idx = timestamps.index(timestamp)
            if idx == 0 or idx == len(timestamps) - 1:
                raise ValueError(
                    "Selected timestamp must have both a previous and a next frame"
                )

        ts_next = timestamps[idx + 1]
        ts_prev = timestamps[idx - 1]

        print(f"[INFO] Loading pc0 @ {timestamp}, pc1 @ {ts_next}, pch1 @ {ts_prev}")

        pc0, gm0, pose0, lab0 = _load_frame(f[timestamp], device)
        pc1, gm1, pose1, _ = _load_frame(f[ts_next], device)
        pch1, gmh1, poseh1, _ = _load_frame(f[ts_prev], device)

    # ------------------------------------------------------------------
    # Remove ground points (training loss also works on non-ground points).
    # ------------------------------------------------------------------
    pc0, lab0 = pc0[~gm0], lab0[~gm0]
    pc1 = pc1[~gm1]
    pch1 = pch1[~gmh1]

    print(f"[INFO] Non-ground points: pc0={pc0.shape[0]}, pc1={pc1.shape[0]}, pch1={pch1.shape[0]}")
    unique_labels = torch.unique(lab0)
    print(f"[INFO] Available labels in pc0: {unique_labels.tolist()}")
    if label not in unique_labels:
        raise ValueError(
            f"Label {label} not present in pc0. Available labels: {unique_labels.tolist()}"
        )

    # ------------------------------------------------------------------
    # Warp to pc1 coordinate system.
    # ------------------------------------------------------------------
    batch = {
        "pc0": pc0.unsqueeze(0),           # (1, N, 3)
        "pc1": pc1.unsqueeze(0),
        "pch1": pch1.unsqueeze(0),
        "pose0": pose0.unsqueeze(0),       # (1, 4, 4)
        "pose1": pose1.unsqueeze(0),
        "poseh1": poseh1.unsqueeze(0),
    }
    pcs_dict = wrap_batch_pcs(batch, num_frames=3)

    # ------------------------------------------------------------------
    # Decide which auxiliary frames to visualize.
    # ------------------------------------------------------------------
    available_targets = {
        "pc1": pcs_dict["pc1s"][0],
        "pch1": pcs_dict["pch1s"][0],
    }
    if frame_keys is None:
        frame_keys = [k for k in ["pc1", "pch1"] if k in available_targets]

    frames_dists: Dict[str, List[torch.Tensor]] = {}
    frames_indices: Dict[str, List[torch.Tensor]] = {}
    res_dict: Dict[str, List[torch.Tensor]] = {}

    pc0_warped = pcs_dict["pc0s"][0]
    for fid in frame_keys:
        tgt = available_targets[fid]
        d, idx = _nearest_neighbor(pc0_warped, tgt)
        frames_dists[fid] = [d]
        frames_indices[fid] = [idx]
        res_dict[f"{fid}_list"] = [tgt]

    # ------------------------------------------------------------------
    # Visualize.
    # ------------------------------------------------------------------
    vis_cluster_topk(
        p0=pc0_warped,
        lab0=lab0,
        flow_list=None,  # standalone: no network predicted flow to draw
        frame_keys=frame_keys,
        frames_dists=frames_dists,
        frames_indices=frames_indices,
        res_dict=res_dict,
        sample_idx=0,
        label=label,
        top_k=top_k,
        save_path=save_path,
        show=show,
    )

    print("[INFO] Visualization done.")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize TeFlow top-k Chamfer matches from an HDF5 file."
    )
    parser.add_argument("--h5_path", required=True, help="Path to preprocessed scene HDF5.")
    parser.add_argument("--timestamp", default=None, help="Timestamp to use as pc0. Auto-picked if omitted.")
    parser.add_argument("--label", type=int, default=3, help="Cluster label to visualize (must be > 1).")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top matches per frame.")
    parser.add_argument(
        "--frames",
        nargs="+",
        default=None,
        help="Auxiliary frames to visualize, e.g. 'pc1 pch1'. Default: both.",
    )
    parser.add_argument("--save_path", default=None, help="Path to save screenshot.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive Open3D window (needs display).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    visualize_topk_from_h5(
        h5_path=args.h5_path,
        timestamp=args.timestamp,
        label=args.label,
        top_k=args.top_k,
        frame_keys=args.frames,
        save_path=args.save_path,
        show=args.show,
        device=device,
    )


if __name__ == "__main__":
    main()
