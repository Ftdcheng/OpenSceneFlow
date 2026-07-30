"""
Interactive Open3D visualizer for Chamfer nearest-neighbor correspondences.

This tool visualizes the same top-k Chamfer matches that drive
``multi_frames_clusterLoss_reflow``. It loads a frame (pc0) and its auxiliary
frames via ``HDF5Dataset``, lets the user pick a dynamic cluster, computes the
k closest pc0 points to each auxiliary frame, and draws arrows, matched
clusters, and AV2 ground-truth BBOXes.

Usage:
    conda run -n opensf python tools/vis_chamfer_match.py \
        --h5py_dir /home/kin/data/av2/h5py/sensor/train \
        --n_frames 3

Controls:
    A / D          previous / next timestamp in current scene
    Mouse left     rotate
    Mouse right    pan
    Mouse wheel    zoom
"""

from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import h5py
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; avoid conflicts with Open3D GUI
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from loguru import logger
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

# Add repo root to path so that ``src.*`` imports work when running from tools/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.autolabel import seflow_auto
from src.dataset import HDF5Dataset
from src.utils.av2_eval import CATEGORY_TO_INDEX
from src.utils.o3d_view import color_map as O3D_COLOR_MAP

try:
    from assets.cuda.chamfer3D import nnChamferDis
    _CUDA_CHAMFER = nnChamferDis()
except Exception:
    _CUDA_CHAMFER = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_color(category: str) -> np.ndarray:
    """Return a deterministic RGB color (0-1) for an AV2 category name."""
    idx = CATEGORY_TO_INDEX.get(str(category).upper(), 0)
    return np.asarray(O3D_COLOR_MAP[idx % len(O3D_COLOR_MAP)])


def _distance_colors(dists: np.ndarray, cmap: str = "coolwarm") -> np.ndarray:
    """Map distances to RGB colors using a matplotlib colormap."""
    if dists.size == 0:
        return np.zeros((0, 3))
    d_min, d_max = dists.min(), dists.max()
    if d_max <= d_min:
        norm = np.zeros_like(dists)
    else:
        norm = (dists - d_min) / (d_max - d_min)
    return matplotlib.colormaps[cmap](norm)[:, :3]


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to (N, 3) points."""
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    return (T @ pts_h.T).T[:, :3]


def _sensor_to_pc0_transform(pose_aux: np.ndarray, pose0: np.ndarray,
                              ego_SE3_sensor: np.ndarray) -> np.ndarray:
    """Return T that maps points from sensor_aux frame to sensor_0 frame.

    Chain: sensor_aux -> ego_aux -> city -> ego_0 -> sensor_0.
    """
    E = ego_SE3_sensor
    E_inv = np.linalg.inv(E)
    pose0_inv = np.linalg.inv(pose0)
    return E_inv @ pose0_inv @ pose_aux @ E


def _bbox_corners(center: np.ndarray, extent: np.ndarray,
                  quat: np.ndarray) -> np.ndarray:
    """Return 8 corners of an oriented BBOX from center/extent/[w,x,y,z] quat."""
    l, w, h = extent
    corners = np.array([
        [-l / 2, -w / 2, -h / 2],
        [ l / 2, -w / 2, -h / 2],
        [ l / 2,  w / 2, -h / 2],
        [-l / 2,  w / 2, -h / 2],
        [-l / 2, -w / 2,  h / 2],
        [ l / 2, -w / 2,  h / 2],
        [ l / 2,  w / 2,  h / 2],
        [-l / 2,  w / 2,  h / 2],
    ])
    rot = R.from_quat(quat[[1, 2, 3, 0]]).as_matrix()  # [w,x,y,z] -> [x,y,z,w]
    return corners @ rot.T + center


def _bbox_lineset(corners: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
    """Create an Open3D LineSet from 8 BBOX corners."""
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
    return ls


def _point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """Create an Open3D point cloud."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _line_set(points0: np.ndarray, points1: np.ndarray,
              colors: np.ndarray) -> o3d.geometry.LineSet:
    """Create a LineSet connecting pairs of points, colored per line."""
    k = len(points0)
    pts = np.vstack([points0, points1])
    lines = [[i, i + k] for i in range(k)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _chamfer_nn(src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-point NN distances and indices from src to tgt.

    Uses CUDA Chamfer if available, otherwise scipy cKDTree on CPU.
    """
    if _CUDA_CHAMFER is not None and src.is_cuda:
        d_list, i_list = _CUDA_CHAMFER.batched_disid_res([src], [tgt])
        return d_list[0], i_list[0]

    src_np = src.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()
    tree = cKDTree(tgt_np)
    dists, idxs = tree.query(src_np, k=1)
    return (
        torch.from_numpy(dists.astype(np.float32)).to(src.device),
        torch.from_numpy(idxs.astype(np.int64)).to(src.device),
    )


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

class ChamferMatchVisualizer:
    # Default colors for auxiliary frames (cycle if more frames).
    FRAME_COLORS: Dict[str, np.ndarray] = {
        "pc1": np.array([1.0, 0.0, 0.0]),
        "pch1": np.array([0.0, 1.0, 0.0]),
        "pch2": np.array([1.0, 1.0, 0.0]),
        "pch3": np.array([1.0, 0.0, 1.0]),
        "pch4": np.array([0.0, 1.0, 1.0]),
        "pch5": np.array([1.0, 0.5, 0.0]),
    }

    def __init__(
        self,
        h5py_dir: str = "/home/kin/data/av2/h5py/sensor/train",
        n_frames: int = 2,
        top_k: int = 5,
        point_size: float = 2.0,
        device: str = "cuda",
        scene_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.h5py_dir = Path(h5py_dir)
        self.n_frames = n_frames
        self.top_k = top_k
        self.point_size = point_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if not (self.h5py_dir / "index_total.pkl").exists():
            raise FileNotFoundError(f"No index_total.pkl in {self.h5py_dir}")
        if not (self.h5py_dir / "bbox.h5").exists():
            raise FileNotFoundError(
                f"No bbox.h5 in {self.h5py_dir}. "
                "Run: python tools/precompute_av2_bboxes.py ..."
            )

        # Load index and open bbox file.
        with open(self.h5py_dir / "index_total.pkl", "rb") as f:
            self.index = pickle.load(f)
        self.bbox_h5 = h5py.File(self.h5py_dir / "bbox.h5", "r")

        # Build scene -> sorted timestamps mapping.
        self.scene_timestamps: Dict[str, List[str]] = defaultdict(list)
        for scene_id, ts in self.index:
            self.scene_timestamps[scene_id].append(ts)
        for scene_id in self.scene_timestamps:
            self.scene_timestamps[scene_id] = sorted(
                self.scene_timestamps[scene_id], key=int
            )
        self.scene_ids = sorted(self.scene_timestamps.keys())

        # Build (scene_id, timestamp) -> canonical index mapping.
        self.index_lookup = {
            (scene_id, ts): idx for idx, (scene_id, ts) in enumerate(self.index)
        }

        # Dataset for loading point clouds and labels.
        self.dataset = HDF5Dataset(
            directory=str(self.h5py_dir),
            n_frames=n_frames,
            ssl_label="seflow_auto",
        )

        # Current selection state.
        self.current_scene_id = self.scene_ids[0]
        self.current_timestamp = self.scene_timestamps[self.current_scene_id][0]
        if scene_id is not None and scene_id in self.scene_timestamps:
            self.current_scene_id = scene_id
            if timestamp is not None and timestamp in self.scene_timestamps[scene_id]:
                self.current_timestamp = timestamp
            else:
                self.current_timestamp = self.scene_timestamps[scene_id][0]
        self.current_label: Optional[int] = None
        self.show_bbox = True

        # Open3D GUI state.
        self.app = gui.Application.instance
        self.app.initialize()
        self.window: Optional[gui.Window] = None
        self.scene: Optional[gui.SceneWidget] = None
        self.panel: Optional[gui.Vert] = None
        self.scene_combo: Optional[gui.Combobox] = None
        self.ts_combo: Optional[gui.Combobox] = None
        self.label_combo: Optional[gui.Combobox] = None
        self.k_slider: Optional[gui.Slider] = None
        self.bbox_check: Optional[gui.Checkbox] = None
        self.info_label: Optional[gui.Label] = None
        self._panel_width = 260

    # ------------------------------------------------------------------
    # GUI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Create the main window and control panel."""
        logger.debug("[_build_ui] creating window")
        self.window = self.app.create_window(
            "Chamfer Match Visualizer", 1600, 900
        )
        logger.debug("[_build_ui] window created")

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        logger.debug("[_build_ui] scene widget created")

        # Left control panel.
        self.panel = gui.Vert(0, gui.Margins(10))

        # Scene selection.
        self.panel.add_child(gui.Label("Scene"))
        self.scene_combo = gui.Combobox()
        for scene_id in self.scene_ids:
            self.scene_combo.add_item(scene_id)
        self.scene_combo.set_on_selection_changed(self._on_scene_changed)
        self.panel.add_child(self.scene_combo)

        # Timestamp selection.
        self.panel.add_child(gui.Label("Timestamp"))
        self.ts_combo = gui.Combobox()
        self._populate_ts_combo()
        self.ts_combo.set_on_selection_changed(self._on_timestamp_changed)
        self.panel.add_child(self.ts_combo)

        # Cluster label selection.
        self.panel.add_child(gui.Label("Cluster label"))
        self.label_combo = gui.Combobox()
        self.label_combo.set_on_selection_changed(self._on_label_changed)
        self.panel.add_child(self.label_combo)

        # Top-k slider.
        self.panel.add_child(gui.Label("Top-k"))
        self.k_slider = gui.Slider(gui.Slider.INT)
        self.k_slider.set_limits(1, 50)
        self.k_slider.int_value = self.top_k
        self.k_slider.set_on_value_changed(self._on_k_changed)
        self.panel.add_child(self.k_slider)

        # BBOX toggle.
        self.bbox_check = gui.Checkbox("Show BBOXes")
        self.bbox_check.checked = self.show_bbox
        self.bbox_check.set_on_checked(self._on_bbox_toggled)
        self.panel.add_child(self.bbox_check)

        # Refresh button.
        refresh_btn = gui.Button("Refresh")
        refresh_btn.set_on_clicked(self._refresh)
        self.panel.add_child(refresh_btn)

        # Info label.
        self.panel.add_child(gui.Label("Matched clusters"))
        self.info_label = gui.Label("-")
        self.panel.add_child(self.info_label)

        # Keyboard navigation.
        self.scene.set_on_key(self._on_key)
        # Mouse event logging for debugging.
        self.scene.set_on_mouse(self._on_mouse)

        # Manual layout: panel on the left, scene fills the rest.
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        logger.debug("[_build_ui] children added, layout callback registered")

    def _on_layout(self, theme) -> None:
        """Position scene and panel inside the window."""
        r = self.window.content_rect
        self.scene.frame = gui.Rect(
            r.x + self._panel_width, r.y,
            max(1, r.width - self._panel_width), max(1, r.height)
        )
        self.panel.frame = gui.Rect(
            r.x, r.y, self._panel_width, max(1, r.height)
        )
        logger.info(
            f"[_on_layout] content_rect=({r.x},{r.y},{r.width},{r.height}) "
            f"scene_frame=({self.scene.frame.x},{self.scene.frame.y},{self.scene.frame.width},{self.scene.frame.height}) "
            f"panel_frame=({self.panel.frame.x},{self.panel.frame.y},{self.panel.frame.width},{self.panel.frame.height})"
        )

    def _on_mouse(self, event: gui.MouseEvent) -> int:
        """Log mouse events to help diagnose click-related disappearance."""
        type_names = {
            gui.MouseEvent.BUTTON_DOWN: "DOWN",
            gui.MouseEvent.BUTTON_UP: "UP",
            gui.MouseEvent.MOVE: "MOVE",
            gui.MouseEvent.WHEEL: "WHEEL",
            gui.MouseEvent.DRAG: "DRAG",
        }
        logger.info(
            f"[_on_mouse] type={type_names.get(event.type, event.type)} "
            f"buttons={event.buttons} x={event.x} y={event.y} "
            f"scene_frame=({self.scene.frame.x},{self.scene.frame.y},{self.scene.frame.width},{self.scene.frame.height})"
        )
        return gui.Widget.EventCallbackResult.IGNORED

    def _populate_ts_combo(self) -> None:
        """Fill the timestamp dropdown for the current scene."""
        self.ts_combo.clear_items()
        for ts in self.scene_timestamps[self.current_scene_id]:
            self.ts_combo.add_item(ts)
        if self.current_timestamp in self.scene_timestamps[self.current_scene_id]:
            self.ts_combo.selected_index = self.scene_timestamps[
                self.current_scene_id
            ].index(self.current_timestamp)
        else:
            self.current_timestamp = self.scene_timestamps[self.current_scene_id][0]
            self.ts_combo.selected_index = 0

    def _populate_label_combo(self, labels: List[int]) -> None:
        """Fill the cluster label dropdown."""
        self.label_combo.clear_items()
        for lab in labels:
            self.label_combo.add_item(str(lab))
        if labels:
            if self.current_label in labels:
                self.label_combo.selected_index = labels.index(self.current_label)
            else:
                self.current_label = labels[0]
                self.label_combo.selected_index = 0
        else:
            self.current_label = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_scene_changed(self, text: str, index: int) -> None:
        logger.info(f"[_on_scene_changed] text={text} index={index}")
        self.current_scene_id = text
        self.current_timestamp = self.scene_timestamps[self.current_scene_id][0]
        self._populate_ts_combo()
        self._refresh()

    def _on_timestamp_changed(self, text: str, index: int) -> None:
        logger.info(f"[_on_timestamp_changed] text={text} index={index}")
        self.current_timestamp = text
        self._refresh()

    def _on_label_changed(self, text: str, index: int) -> None:
        logger.info(f"[_on_label_changed] text={text} index={index}")
        self.current_label = int(text)
        self._refresh()

    def _on_k_changed(self, value: float) -> None:
        logger.info(f"[_on_k_changed] value={value}")
        self.top_k = int(value)
        self._refresh()

    def _on_bbox_toggled(self, checked: bool) -> None:
        logger.info(f"[_on_bbox_toggled] checked={checked}")
        self.show_bbox = checked
        self._refresh()

    def _on_key(self, event: gui.KeyEvent) -> int:
        logger.info(f"[_on_key] type={event.type} key={event.key}")
        if event.type != gui.KeyEvent.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED

        timestamps = self.scene_timestamps[self.current_scene_id]
        if event.key == ord("A") or event.key == ord("a"):
            idx = timestamps.index(self.current_timestamp)
            if idx > 0:
                self.current_timestamp = timestamps[idx - 1]
                self.ts_combo.selected_index = idx - 1
                self._refresh()
            return gui.Widget.EventCallbackResult.HANDLED
        elif event.key == ord("D") or event.key == ord("d"):
            idx = timestamps.index(self.current_timestamp)
            if idx < len(timestamps) - 1:
                self.current_timestamp = timestamps[idx + 1]
                self.ts_combo.selected_index = idx + 1
                self._refresh()
            return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED

    # ------------------------------------------------------------------
    # Data loading and transforms
    # ------------------------------------------------------------------

    def _load_data(self) -> Tuple[Dict, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load pc0, auxiliary frames, labels, poses, and warp everything to pc0 frame."""
        idx = self.index_lookup[(self.current_scene_id, self.current_timestamp)]
        logger.debug(f"[_load_data] scene={self.current_scene_id} ts={self.current_timestamp} idx={idx}")
        data = self.dataset[idx]

        E = self.bbox_h5[self.current_scene_id]["ego_SE3_sensor"][:]
        pose0 = data["pose0"]

        frames: Dict[str, np.ndarray] = {"pc0": data["pc0"][:, :3]}
        labels: Dict[str, np.ndarray] = {"pc0": data["pc0_dynamic"]}

        # Future frame pc1.
        if "pc1" in data:
            T = _sensor_to_pc0_transform(data["pose1"], pose0, E)
            frames["pc1"] = _transform_points(data["pc1"][:, :3], T)
            labels["pc1"] = data["pc1_dynamic"]

        # History frames pch1, pch2, ...
        history_frames = self.n_frames - 2
        for i in range(1, history_frames + 1):
            fid = f"pch{i}"
            if fid not in data:
                continue
            T = _sensor_to_pc0_transform(data[f"poseh{i}"], pose0, E)
            frames[fid] = _transform_points(data[fid][:, :3], T)

            if f"{fid}_dynamic" in data:
                labels[fid] = data[f"{fid}_dynamic"]
            else:
                # HDF5Dataset only preloads pch1_dynamic; compute pch2+ on the fly.
                aux_ts = self.index[idx - i][1]
                with h5py.File(self.h5py_dir / f"{self.current_scene_id}.h5", "r") as f:
                    labels[fid] = seflow_auto(f[aux_ts])

        logger.debug(f"[_load_data] frames={list(frames.keys())} shapes={ {k: v.shape for k, v in frames.items()} }")
        return data, frames, labels

    def _load_bboxes(
        self,
        idx: int,
        frame_keys: List[str],
        data: Dict,
    ) -> List[Tuple[str, np.ndarray, str, str]]:
        """Load and warp BBOXes for pc0 and auxiliary frames.

        Returns list of (frame_id, corners_in_pc0, category, track_uuid).
        """
        if not self.show_bbox:
            return []

        E = self.bbox_h5[self.current_scene_id]["ego_SE3_sensor"][:]
        pose0 = data["pose0"]

        bboxes: List[Tuple[str, np.ndarray, str, str]] = []
        timestamps = {
            "pc0": self.current_timestamp,
        }
        if self.n_frames >= 2:
            timestamps["pc1"] = self.index[idx + 1][1]
        for i in range(1, self.n_frames - 1):
            timestamps[f"pch{i}"] = self.index[idx - i][1]

        for fid in frame_keys:
            ts = timestamps.get(fid)
            if ts is None or ts not in self.bbox_h5[self.current_scene_id]:
                continue
            grp = self.bbox_h5[self.current_scene_id][ts]
            centers = grp["centers"][:]
            extents = grp["extents"][:]
            quats = grp["quaternions"][:]
            categories = grp["categories"][:]
            uuids = grp["track_uuids"][:]
            logger.debug(f"[_load_bboxes] {fid} ts={ts} count={len(centers)}")

            if fid == "pc1":
                pose_aux = data["pose1"]
            elif fid.startswith("pch"):
                pose_aux = data[f"poseh{fid[3:]}"]
            else:
                pose_aux = np.eye(4)

            if fid != "pc0":
                T = _sensor_to_pc0_transform(pose_aux, pose0, E)
            else:
                T = np.eye(4)

            for center, extent, quat, cat, uuid in zip(
                centers, extents, quats, categories, uuids
            ):
                corners = _bbox_corners(center, extent, quat)
                corners = _transform_points(corners, T)
                cat_str = cat.decode("utf-8") if isinstance(cat, bytes) else str(cat)
                uuid_str = uuid.decode("utf-8") if isinstance(uuid, bytes) else str(uuid)
                bboxes.append((fid, corners, cat_str, uuid_str))

        logger.debug(f"[_load_bboxes] total bboxes={len(bboxes)}")
        return bboxes

    # ------------------------------------------------------------------
    # Chamfer matching
    # ------------------------------------------------------------------

    def _compute_matches(
        self,
        pc0: np.ndarray,
        frames: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute Chamfer NN distances/indices from pc0 to each auxiliary frame."""
        import time
        pc0_t = torch.from_numpy(pc0).float().to(self.device)
        dists: Dict[str, torch.Tensor] = {}
        indices: Dict[str, torch.Tensor] = {}

        for fid in frames:
            if fid == "pc0":
                continue
            t0 = time.time()
            tgt_t = torch.from_numpy(frames[fid]).float().to(self.device)
            d, i = _chamfer_nn(pc0_t, tgt_t)
            dists[fid] = d
            indices[fid] = i
            logger.debug(f"[_compute_matches] {fid} dists shape={d.shape} min={d.min().item():.3f} max={d.max().item():.3f} time={time.time()-t0:.3f}s")

        return dists, indices

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Reload data and redraw the scene."""
        import time
        t_start = time.time()
        logger.info(f"[_refresh] start scene={self.current_scene_id} ts={self.current_timestamp} label={self.current_label} k={self.top_k}")
        if self.scene is None:
            logger.debug("[_refresh] scene is None, skipping")
            return

        data, frames, labels = self._load_data()
        pc0 = frames["pc0"]
        lab0 = labels["pc0"]

        # Update label dropdown if needed.
        available_labels = sorted(int(l) for l in np.unique(lab0) if l > 1)
        logger.debug(f"[_refresh] available_labels={available_labels[:20]}...")
        self._populate_label_combo(available_labels)
        if self.current_label is None or self.current_label not in available_labels:
            self.current_label = available_labels[0] if available_labels else None

        if self.current_label is None:
            self.info_label.text = "No dynamic clusters in pc0"
            self.scene.scene.clear_geometry()
            self._setup_camera(np.zeros((0, 3)))
            logger.debug("[_refresh] no dynamic clusters")
            return

        cluster_mask = lab0 == self.current_label
        pc0_cluster = pc0[cluster_mask]
        logger.debug(f"[_refresh] selected label={self.current_label} cluster_size={len(pc0_cluster)}")

        dists, indices = self._compute_matches(pc0, frames)

        # Build geometries.
        geometries: List[Tuple[str, o3d.geometry.Geometry]] = []

        # Base pc0 (grey) and selected cluster (blue).
        geometries.append(("pc0_base", _point_cloud(pc0, np.tile([0.7, 0.7, 0.7], (len(pc0), 1)))))
        geometries.append(("pc0_cluster", _point_cloud(pc0_cluster, np.tile([0.0, 0.0, 1.0], (len(pc0_cluster), 1)))))

        matched_info: List[str] = []
        for fid in sorted(frames.keys()):
            if fid == "pc0" or fid not in dists:
                continue

            frame_color = self.FRAME_COLORS.get(
                fid, np.array([0.5, 0.5, 0.5])
            )

            # Distances/indices restricted to the selected cluster.
            d_c = dists[fid][cluster_mask].cpu().numpy()
            i_c = indices[fid][cluster_mask].cpu().numpy()

            if len(d_c) == 0:
                continue

            k = min(self.top_k, len(d_c))
            topk_local = np.argpartition(d_c, k - 1)[:k]
            topk_local = topk_local[np.argsort(d_c[topk_local])]

            src_pts = pc0_cluster[topk_local]
            tgt_pts = frames[fid][i_c[topk_local]]
            topk_dists = d_c[topk_local]

            # Source/target points.
            geometries.append((
                f"{fid}_src",
                _point_cloud(src_pts, np.tile(frame_color * 0.7, (k, 1))),
            ))
            geometries.append((
                f"{fid}_tgt",
                _point_cloud(tgt_pts, np.tile(frame_color, (k, 1))),
            ))

            # Connection lines colored by distance.
            line_colors = _distance_colors(topk_dists)
            geometries.append((f"{fid}_lines", _line_set(src_pts, tgt_pts, line_colors)))

            # Highlight matched clusters in the auxiliary frame.
            tgt_labels = labels[fid]
            matched_labels = set(int(tgt_labels[idx]) for idx in i_c[topk_local])
            matched_labels = {l for l in matched_labels if l > 1}
            for ml in matched_labels:
                ml_mask = tgt_labels == ml
                ml_pts = frames[fid][ml_mask]
                geometries.append((
                    f"{fid}_cluster_{ml}",
                    _point_cloud(ml_pts, np.tile(frame_color, (len(ml_pts), 1))),
                ))

            matched_info.append(
                f"{fid}: k={k}, matched clusters={sorted(matched_labels)}"
            )

        # BBOXes.
        idx = self.index_lookup[(self.current_scene_id, self.current_timestamp)]
        bboxes = self._load_bboxes(idx, list(frames.keys()), data)
        for bbox_idx, (fid, corners, cat, uuid) in enumerate(bboxes):
            color = _category_color(cat)
            geometries.append((f"bbox_{bbox_idx}", _bbox_lineset(corners, color)))

        # Update scene.
        logger.debug(f"[_refresh] adding {len(geometries)} geometries")
        self.scene.scene.clear_geometry()
        mat_pcd = rendering.MaterialRecord()
        mat_pcd.shader = "defaultUnlit"
        mat_pcd.point_size = self.point_size

        mat_line = rendering.MaterialRecord()
        mat_line.shader = "defaultUnlit"
        # Line width is not supported by the new renderer; keep default.

        for name, geo in geometries:
            if isinstance(geo, o3d.geometry.LineSet):
                self.scene.scene.add_geometry(name, geo, mat_line)
            else:
                self.scene.scene.add_geometry(name, geo, mat_pcd)

        # Update info label.
        self.info_label.text = "\n".join(matched_info) if matched_info else "No matches"

        # Reset camera on first load, otherwise keep viewpoint.
        all_pts = np.vstack([pc0] + [frames[f] for f in frames if f != "pc0"])
        self._setup_camera(all_pts)
        logger.info(f"[_refresh] done in {time.time()-t_start:.3f}s")

    def _setup_camera(self, points: np.ndarray) -> None:
        """Point camera at pc0 origin; fit to points if available."""
        logger.debug(f"[_setup_camera] points={points.shape}")
        if points.size == 0:
            self.scene.look_at([0, 0, 0], [10, 10, 10], [0, 0, 1])
            return

        center = points.mean(axis=0)
        extent = points.max(axis=0) - points.min(axis=0)
        distance = max(extent.max(), 10.0)
        eye = center + np.array([distance, distance, distance * 0.5])
        self.scene.look_at(center.tolist(), eye.tolist(), [0, 0, 1])

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the interactive visualizer."""
        self._build_ui()
        self._refresh()
        try:
            self.app.run()
        finally:
            if self.bbox_h5 is not None:
                self.bbox_h5.close()


def main(
    h5py_dir: str = "/home/kin/data/av2/h5py/sensor/train",
    n_frames: int = 2,
    top_k: int = 5,
    point_size: float = 2.0,
    device: str = "cuda",
    scene_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    log_level: str = "INFO",
) -> None:
    """Launch the Chamfer match visualizer.

    Args:
        h5py_dir: Directory containing preprocessed HDF5 files, index_total.pkl,
            and bbox.h5.
        n_frames: Number of frames per sample (2 = pc0+pc1, 3 = +pch1, etc.).
        top_k: Number of closest matches to draw per auxiliary frame.
        point_size: Open3D point size.
        device: "cuda" or "cpu" for Chamfer computation.
        scene_id: Optional initial scene ID.
        timestamp: Optional initial timestamp (must belong to scene_id).
        log_level: loguru log level (DEBUG, INFO, WARNING, ERROR).
    """
    # Configure loguru: remove default sink and add stderr sink with chosen level.
    logger.remove()
    logger.add(sys.stderr, level=log_level.upper(), enqueue=True)
    logger.info(
        "Starting ChamferMatchVisualizer: h5py_dir={} n_frames={} top_k={} "
        "device={} scene_id={} timestamp={} log_level={}",
        h5py_dir, n_frames, top_k, device, scene_id, timestamp, log_level,
    )

    vis = ChamferMatchVisualizer(
        h5py_dir=h5py_dir,
        n_frames=n_frames,
        top_k=top_k,
        point_size=point_size,
        device=device,
        scene_id=scene_id,
        timestamp=timestamp,
    )
    vis.run()


if __name__ == "__main__":
    fire.Fire(main)
