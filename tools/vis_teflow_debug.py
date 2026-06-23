"""
TeFlow Debug Visualizer (polyscope)
====================================
Interactive visualization for debugging TeFlow clustering and Chamfer matching.

Usage:
    python tools/vis_teflow_debug.py --data_dir /path/to/h5py/val

Controls:
    - Left panel buttons: next/prev frame, toggle visualizations
    - Mouse: rotate, zoom, pan the 3D view
    - Space: toggle animation playback (if implemented)
"""
import argparse
import glob
import os

import h5py
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_h5_frame(filepath):
    """Load a single frame from HDF5."""
    with h5py.File(filepath, "r") as f:
        keys = list(f.keys())
        if len(keys) == 0:
            return None
        ts = keys[0]
        g = f[ts]

        data = {
            "scene_id": os.path.basename(filepath).replace(".h5", ""),
            "timestamp": ts,
        }

        # point clouds
        for k in ["lidar", "ground_mask", "pose", "flow"]:
            if k in g:
                data[k] = g[k][:]

        # dynamic labels (if pre-computed clustering exists)
        if "label" in g:
            data["labels"] = g["label"][:].astype(np.int32)
        elif "dynamic" in g:
            data["labels"] = g["dynamic"][:].astype(np.int32)
        else:
            data["labels"] = None

    return data


def flow_to_rgb(flow):
    """Map 3D flow vectors to RGB colors (similar to Open3D style)."""
    # Normalize by max magnitude for color mapping
    mag = np.linalg.norm(flow, axis=1, keepdims=True)
    # Use direction: map (x,y,z) to (r,g,b) in [0,1]
    # Normalize flow direction
    dir_norm = flow / (mag + 1e-6)
    # Map from [-1,1] to [0,1]
    color = (dir_norm + 1.0) / 2.0
    # Dampen static points
    is_static = mag.squeeze() < 0.08
    color[is_static] = [0.9, 0.9, 0.9]  # grey for static
    return np.clip(color, 0, 1)


def label_to_colors(labels):
    """Map integer cluster labels to distinct colors."""
    if labels is None:
        return None
    uniq = np.unique(labels)
    np.random.seed(42)
    palette = np.random.rand(max(uniq.max() + 1, 1), 3)
    palette[0] = [0.5, 0.5, 0.5]   # label 0 -> grey (static/ground)
    palette[1] = [0.3, 0.3, 0.3]   # label 1 -> dark grey
    return palette[labels]


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

class TeFlowDebugger:
    def __init__(self, data_dir, start_id=0):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {data_dir}")
        print(f"[TeFlowDebug] Found {len(self.files)} scenes.")

        self.frame_idx = start_id
        self.data = None
        self.pc0_cloud = None
        self.pc1_cloud = None
        self.match_lines = None
        self.flow_vectors = None

        # UI state
        self.show_chamfer = True
        self.show_flow = True
        self.show_labels = True
        self.show_pc1 = True
        self.max_match_lines = 5000   # subsample for performance
        self.max_flow_arrows = 2000

    def load_current_frame(self):
        path = self.files[self.frame_idx]
        print(f"[TeFlowDebug] Loading: {path}")
        self.data = load_h5_frame(path)
        if self.data is None:
            print("[TeFlowDebug] Empty file, skipping.")
            return False

        # Prepare point clouds
        pc0 = self.data["lidar"][:, :3]
        gm = self.data.get("ground_mask", np.zeros(len(pc0), dtype=bool))
        self.mask = ~gm  # keep non-ground

        self.pc0_pts = pc0[self.mask]
        self.labels = self.data["labels"][self.mask] if self.data["labels"] is not None else None
        self.flow = self.data["flow"][self.mask] if "flow" in self.data else np.zeros_like(self.pc0_pts)

        # For pc1 we need the next timestamp; load from the next file if available
        self.pc1_pts = None
        if self.frame_idx + 1 < len(self.files):
            next_data = load_h5_frame(self.files[self.frame_idx + 1])
            if next_data and "lidar" in next_data:
                pc1_full = next_data["lidar"][:, :3]
                gm1 = next_data.get("ground_mask", np.zeros(len(pc1_full), dtype=bool))
                self.pc1_pts = pc1_full[~gm1]

        return True

    def register_geometries(self):
        """Register or update polyscope structures."""
        # pc0 with label colors
        if self.pc0_cloud is None:
            self.pc0_cloud = ps.register_point_cloud("pc0", self.pc0_pts, point_render_mode="sphere")
        else:
            self.pc0_cloud.update_point_positions(self.pc0_pts)

        if self.show_labels and self.labels is not None:
            colors = label_to_colors(self.labels)
            self.pc0_cloud.add_color_quantity("cluster_labels", colors, enabled=True)
        else:
            colors = flow_to_rgb(self.flow)
            self.pc0_cloud.add_color_quantity("flow_color", colors, enabled=True)

        # pc1 (optional)
        if self.pc1_pts is not None and self.show_pc1:
            if self.pc1_cloud is None:
                self.pc1_cloud = ps.register_point_cloud("pc1", self.pc1_pts, point_render_mode="sphere")
                # Tint pc1 blue-ish
                blue = np.tile([0.2, 0.4, 0.9], (len(self.pc1_pts), 1))
                self.pc1_cloud.add_color_quantity("tint", blue, enabled=True)
            else:
                self.pc1_cloud.update_point_positions(self.pc1_pts)
                self.pc1_cloud.set_enabled(True)
        elif self.pc1_cloud is not None:
            self.pc1_cloud.set_enabled(False)

        self._update_chamfer_matches()
        self._update_flow_arrows()

    def _update_chamfer_matches(self):
        """Draw lines from pc0 points to their nearest neighbours in pc1."""
        if not self.show_chamfer or self.pc1_pts is None or len(self.pc1_pts) == 0:
            if self.match_lines is not None:
                self.match_lines.set_enabled(False)
            return

        # Warp pc0 by flow
        warped = self.pc0_pts + self.flow

        # Subsample for performance
        n = len(warped)
        if n > self.max_match_lines:
            idx = np.random.choice(n, self.max_match_lines, replace=False)
            src = warped[idx]
        else:
            idx = np.arange(n)
            src = warped

        # Find nearest neighbours in pc1
        tree = cKDTree(self.pc1_pts)
        _, nn_idx = tree.query(src, k=1)
        tgt = self.pc1_pts[nn_idx]

        # Build line network: nodes = src + tgt, edges = (i, i+n)
        nodes = np.vstack([src, tgt])
        edges = np.array([[i, i + len(src)] for i in range(len(src))])

        if self.match_lines is None:
            self.match_lines = ps.register_curve_network(
                "chamfer_matches", nodes, edges, enabled=True
            )
            # Tint lines green
            green = np.tile([0.0, 0.8, 0.2], (len(nodes), 1))
            self.match_lines.add_color_quantity("match_color", green, enabled=True)
        else:
            self.match_lines.update_node_positions(nodes)
            self.match_lines.set_enabled(True)

    def _update_flow_arrows(self):
        """Draw flow vectors as arrows from pc0 points."""
        if not self.show_flow:
            if self.flow_vectors is not None:
                self.flow_vectors.set_enabled(False)
            return

        n = len(self.pc0_pts)
        if n > self.max_flow_arrows:
            idx = np.random.choice(n, self.max_flow_arrows, replace=False)
            src = self.pc0_pts[idx]
            vec = self.flow[idx]
        else:
            src = self.pc0_pts
            vec = self.flow

        tgt = src + vec
        nodes = np.vstack([src, tgt])
        edges = np.array([[i, i + len(src)] for i in range(len(src))])

        if self.flow_vectors is None:
            self.flow_vectors = ps.register_curve_network(
                "flow_vectors", nodes, edges, enabled=True
            )
            red = np.tile([0.9, 0.1, 0.1], (len(nodes), 1))
            self.flow_vectors.add_color_quantity("flow_red", red, enabled=True)
        else:
            self.flow_vectors.update_node_positions(nodes)
            self.flow_vectors.set_enabled(True)

    def ui_callback(self):
        """ImGui UI panel."""
        psim.PushItemWidth(150)

        # Frame info
        psim.Text(f"Scene: {self.data['scene_id']}")
        psim.Text(f"Timestamp: {self.data['timestamp']}")
        psim.Text(f"Points: {len(self.pc0_pts)}")
        if self.labels is not None:
            psim.Text(f"Clusters: {len(np.unique(self.labels)) - 2}")  # exclude 0,1
        psim.Separator()

        # Navigation buttons
        if psim.Button("<< Prev"):
            self.frame_idx = max(0, self.frame_idx - 1)
            if self.load_current_frame():
                self.register_geometries()
        psim.SameLine()
        if psim.Button("Next >>"):
            self.frame_idx = min(len(self.files) - 1, self.frame_idx + 1)
            if self.load_current_frame():
                self.register_geometries()
        psim.SameLine()
        psim.Text(f"  Frame {self.frame_idx + 1}/{len(self.files)}")

        psim.Separator()

        # Toggles
        changed = False
        if psim.Checkbox("Show Chamfer Matches", self.show_chamfer):
            self.show_chamfer = not self.show_chamfer
            changed = True
        if psim.Checkbox("Show Flow Vectors", self.show_flow):
            self.show_flow = not self.show_flow
            changed = True
        if psim.Checkbox("Show Labels (vs Flow Color)", self.show_labels):
            self.show_labels = not self.show_labels
            changed = True
        if psim.Checkbox("Show pc1", self.show_pc1):
            self.show_pc1 = not self.show_pc1
            changed = True

        if changed:
            self.register_geometries()

        psim.Separator()

        # Subsample sliders
        changed_ml, self.max_match_lines = psim.SliderInt(
            "Max Match Lines", self.max_match_lines, v_min=100, v_max=20000)
        if changed_ml:
            self._update_chamfer_matches()
        changed_fa, self.max_flow_arrows = psim.SliderInt(
            "Max Flow Arrows", self.max_flow_arrows, v_min=100, v_max=20000)
        if changed_fa:
            self._update_flow_arrows()

        psim.PopItemWidth()

    def run(self):
        # Initialize polyscope
        ps.init()
        ps.set_ground_plane_mode("none")
        ps.set_program_name("TeFlow Debugger")
        ps.set_window_size(1600, 1000)

        # Load first frame
        if not self.load_current_frame():
            print("[TeFlowDebug] Failed to load initial frame.")
            return
        self.register_geometries()

        # Set UI callback
        ps.set_user_callback(self.ui_callback)

        print("[TeFlowDebug] Starting polyscope viewer...")
        ps.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TeFlow Debug Visualizer")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to HDF5 dataset directory (e.g., /home/kin/data/av2/h5py/sensor/val)")
    parser.add_argument("--start_id", type=int, default=0,
                        help="Starting frame index")
    args = parser.parse_args()

    dbg = TeFlowDebugger(args.data_dir, start_id=args.start_id)
    dbg.run()


if __name__ == "__main__":
    main()
