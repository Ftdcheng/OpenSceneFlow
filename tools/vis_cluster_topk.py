"""
Visualize TeFlow top-k Chamfer matches for a single cluster.

This helper draws, for one sample in a batch and one dynamic cluster label:
  - the full pc0 point cloud (grey);
  - the selected cluster in pc0 (blue);
  - for each auxiliary frame, the top-k source points in the cluster (frame color)
    and their nearest-neighbor target points in that frame (same frame color);
  - lines connecting each src->target pair.

It is meant to be called from inside ``multi_frames_clusterLoss`` during debugging
or used standalone with the same tensors/lists that the loss receives.

Usage from inside ``multi_frames_clusterLoss``:

    from tools.vis_cluster_topk import vis_cluster_topk

    vis_cluster_topk(
        p0, lab0, flow_list,
        frame_keys, frames_dists, frames_indices, res_dict,
        sample_idx=0, label=3, top_k=5,
        save_path="debug/cluster_topk_sample0_label3.png",
    )

Standalone usage with numpy arrays is also supported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch


def _to_numpy(x):
    """Move tensor to CPU and convert to numpy. Pass-through for ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def vis_cluster_topk(
    p0: Union[torch.Tensor, np.ndarray],
    lab0: Union[torch.Tensor, np.ndarray],
    flow_list: Optional[List[Union[torch.Tensor, np.ndarray]]],
    frame_keys: List[str],
    frames_dists: Dict[str, List[Union[torch.Tensor, np.ndarray]]],
    frames_indices: Dict[str, List[Union[torch.Tensor, np.ndarray]]],
    res_dict: Dict[str, List[Union[torch.Tensor, np.ndarray]]],
    sample_idx: int = 0,
    label: int = 3,
    top_k: int = 5,
    frame_colors: Optional[Dict[str, List[float]]] = None,
    cluster_color: List[float] = [0.0, 0.0, 1.0],
    bg_color: List[float] = [0.7, 0.7, 0.7],
    line_width: float = 2.0,
    point_size: float = 3.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> List:
    """Visualize top-k Chamfer matches of one cluster across auxiliary frames.

    Args:
        p0: Source point cloud for the selected sample, shape ``(N_i, 3)``.
        lab0: Cluster labels for the source points, shape ``(N_i,)``.
        flow_list: Predicted flow for the selected sample, shape ``(N_i, 3)``.
            Used only to color cluster points by flow magnitude if provided;
            can be ``None``.
        frame_keys: List of auxiliary frame IDs, e.g. ``['pc1', 'pch1']``.
        frames_dists: Nearest-neighbor distances from pc0 to each auxiliary frame.
            ``frames_dists[frame_id]`` is a list of length ``B``; element ``i``
            has shape ``(N_i,)``.
        frames_indices: Nearest-neighbor local indices into each auxiliary frame.
            Same nested structure as ``frames_dists``.
        res_dict: Dictionary with ``f'{frame_id}_list'`` entries. Each entry is a
            list of length ``B``; element ``i`` has shape ``(M_i, 3)``.
        sample_idx: Which sample in the batch to visualize.
        label: Which cluster label to visualize (must be ``> 1``).
        top_k: Number of closest matches to draw per auxiliary frame.
        frame_colors: Optional mapping from ``frame_id`` to RGB color. Defaults
            to a small rotating palette.
        cluster_color: RGB color for the selected cluster points.
        bg_color: RGB color for all other pc0 points.
        line_width: Width of match lines.
        point_size: Size of rendered points.
        save_path: If given, save a screenshot to this path. Requires Open3D
            headless rendering support for off-screen saving.
        show: If ``True``, open the interactive Open3D window.

    Returns:
        List of Open3D geometry objects. Useful for chaining with other
        visualizations.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "vis_cluster_topk requires open3d. Install it with:\n"
            "  conda run -n opensf pip install open3d"
        ) from exc

    p0 = _to_numpy(p0)
    lab0 = _to_numpy(lab0)

    if frame_colors is None:
        palette = np.array([
            [1.0, 0.0, 0.0],   # pc1  -> red
            [0.0, 1.0, 0.0],   # pch1 -> green
            [1.0, 1.0, 0.0],   # pch2 -> yellow
            [1.0, 0.0, 1.0],   # pc2  -> magenta
            [0.0, 1.0, 1.0],   # pch3 -> cyan
            [1.0, 0.5, 0.0],   # extra -> orange
        ])
        frame_colors = {}
        for idx, fid in enumerate(frame_keys):
            frame_colors[fid] = palette[idx % len(palette)].tolist()

    # ------------------------------------------------------------------
    # 1. Base point cloud: grey background, blue selected cluster
    # ------------------------------------------------------------------
    cluster_mask = lab0 == label
    if cluster_mask.sum() == 0:
        raise ValueError(f"Sample {sample_idx} has no points with label {label}")

    colors = np.tile(bg_color, (p0.shape[0], 1))
    colors[cluster_mask] = cluster_color

    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(p0)
    pcd_base.colors = o3d.utility.Vector3dVector(colors)

    geometries: List[o3d.geometry.Geometry] = [pcd_base]

    # ------------------------------------------------------------------
    # 2. For each auxiliary frame, draw top-k matches
    # ------------------------------------------------------------------
    p0_cluster = p0[cluster_mask]

    for frame_id in frame_keys:
        dist_c = _to_numpy(frames_dists[frame_id][sample_idx])[cluster_mask]
        idx_c = _to_numpy(frames_indices[frame_id][sample_idx])[cluster_mask]

        if dist_c.shape[0] <= top_k:
            continue

        topk_dists, topk_local = torch.topk(
            torch.from_numpy(dist_c), k=top_k, largest=False
        )
        topk_local = topk_local.numpy()

        src_pts = p0_cluster[topk_local]
        tgt_pts = _to_numpy(res_dict[f"{frame_id}_list"][sample_idx])[idx_c[topk_local]]

        color = np.asarray(frame_colors.get(frame_id, [1.0, 0.0, 0.0]))

        # Source points for this frame (slightly smaller/darker)
        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(src_pts)
        pcd_src.colors = o3d.utility.Vector3dVector(np.tile(color * 0.7, (src_pts.shape[0], 1)))
        geometries.append(pcd_src)

        # Target points for this frame
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(tgt_pts)
        pcd_tgt.colors = o3d.utility.Vector3dVector(np.tile(color, (tgt_pts.shape[0], 1)))
        geometries.append(pcd_tgt)

        # Connecting lines
        line_points = np.vstack([src_pts, tgt_pts])
        line_indices = []
        for k in range(top_k):
            line_indices.append([k, k + top_k])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(np.tile(color, (len(line_indices), 1)))
        geometries.append(line_set)

    # ------------------------------------------------------------------
    # 3. Optional predicted-flow arrows for the cluster
    # ------------------------------------------------------------------
    if flow_list is not None:
        fv = _to_numpy(flow_list[sample_idx])[cluster_mask]
        # Subsample arrows if the cluster is large
        step = max(1, fv.shape[0] // 200)
        src_sub = p0_cluster[::step]
        flow_sub = fv[::step]

        for s, f in zip(src_sub, flow_sub):
            # Draw a short line segment to indicate flow direction.
            # Open3D does not have a one-line arrow primitive, so we use a
            # cylinder-like line plus a small endpoint sphere.
            e = s + f * 0.2  # scale for visualization
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector([s, e])
            ls.lines = o3d.utility.Vector2iVector([[0, 1]])
            ls.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])
            geometries.append(ls)

            sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sp.translate(e)
            sp.paint_uniform_color([0.0, 0.0, 0.0])
            geometries.append(sp)

    # ------------------------------------------------------------------
    # 4. Render or save
    # ------------------------------------------------------------------
    if show or save_path is not None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"sample{sample_idx}_label{label}_topk{top_k}",
            width=1600,
            height=900,
        )
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.line_width = line_width
        render_option.background_color = np.asarray([0.9, 0.9, 0.9])

        for geo in geometries:
            vis.add_geometry(geo)

        vis.poll_events()
        vis.update_renderer()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            vis.capture_screen_image(save_path, do_render=True)
            print(f"[vis_cluster_topk] Saved screenshot to {save_path}")

        if show:
            vis.run()
        vis.destroy_window()

    return geometries


def vis_cluster_topk_from_loss(
    pc0_list,
    pc0_lab_list,
    flow_list,
    frame_keys,
    frames_dists,
    frames_indices,
    res_dict,
    sample_idx: int = 0,
    label: int = 3,
    top_k: int = 5,
    **kwargs,
):
    """Convenience wrapper with the exact argument names used in ``multi_frames_clusterLoss``.

    Example inside ``multi_frames_clusterLoss``:

        from tools.vis_cluster_topk import vis_cluster_topk_from_loss
        vis_cluster_topk_from_loss(
            pc0_list, pc0_lab_list, flow_list,
            frame_keys, frames_dists, frames_indices, res_dict,
            sample_idx=0, label=3, top_k=5,
            save_path=f"debug/topk_sample0_label3.png",
        )
    """
    return vis_cluster_topk(
        p0=pc0_list[sample_idx],
        lab0=pc0_lab_list[sample_idx],
        flow_list=flow_list,
        frame_keys=frame_keys,
        frames_dists=frames_dists,
        frames_indices=frames_indices,
        res_dict=res_dict,
        sample_idx=sample_idx,
        label=label,
        top_k=top_k,
        **kwargs,
    )


if __name__ == "__main__":
    # Minimal sanity check: create two synthetic frames and visualize.
    np.random.seed(0)
    p0 = np.random.randn(500, 3).astype(np.float32) * 2.0
    lab0 = np.zeros(500, dtype=np.int16)
    lab0[100:200] = 3  # a fake cluster

    flow = np.zeros_like(p0)

    frame_keys = ["pc1", "pch1"]
    res_dict = {
        "pc1_list": [p0 + np.array([0.3, 0.0, 0.0])],
        "pch1_list": [p0 - np.array([0.2, 0.0, 0.0])],
    }

    # Synthetic nearest-neighbor data: each source point matches to itself.
    frames_dists = {
        "pc1": [np.linalg.norm(p0 - res_dict["pc1_list"][0], axis=1)],
        "pch1": [np.linalg.norm(p0 - res_dict["pch1_list"][0], axis=1)],
    }
    frames_indices = {
        "pc1": [np.arange(p0.shape[0])],
        "pch1": [np.arange(p0.shape[0])],
    }

    print("Running synthetic sanity-check visualization...")
    vis_cluster_topk(
        p0, lab0, [flow],
        frame_keys, frames_dists, frames_indices, res_dict,
        sample_idx=0, label=3, top_k=5,
        show=True,
    )
