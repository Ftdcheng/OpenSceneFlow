"""
Precompute AV2 3D bounding boxes into a single HDF5 file.

This script reads the raw AV2 `annotations.feather` files, transforms every
cuboid from the ego-vehicle frame into the up_lidar (sensor) frame, and writes
the results to `<h5py_dir>/bbox.h5`. The output is organized as:

    bbox.h5
    └── {scene_id}/
        └── {timestamp}/
            ├── centers      (M, 3) float32
            ├── extents      (M, 3) float32   # length, width, height
            ├── quaternions  (M, 4) float32   # [w, x, y, z]
            ├── categories   (M,)   UTF-8 string
            └── track_uuids  (M,)   UTF-8 string

The sensor-frame BBOXes can then be warped into any chosen pc0 frame at
visualization time using the per-frame ego poses.

Usage:
    conda run -n opensf python tools/precompute_av2_bboxes.py \
        --av2_dir /home/kin/data/av2/sensor/train \
        --h5py_dir /home/kin/data/av2/h5py/sensor/train
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections import defaultdict
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Add repo root to path so that ``src.*`` imports work when running from tools/.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from av2.structures.cuboid import Cuboid, CuboidList
from av2.utils.io import read_feather
from src.utils.av2_eval import read_ego_SE3_sensor


def _rotation_matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a [w, x, y, z] quaternion."""
    return R.from_matrix(rot.astype(np.float64)).as_quat()[[3, 0, 1, 2]]


def _process_one_scene(
    args: Tuple[Path, Path, Path],
) -> Path:
    """Process a single AV2 scene and write sensor-frame BBOXes to a temp HDF5.

    Args:
        args: (av2_scene_dir, temp_output_path, h5py_scene_file)

    Returns:
        Path to the written temp HDF5 file.
    """
    av2_scene_dir, temp_output_path, h5py_scene_file = args
    log_id = av2_scene_dir.name
    ann_path = av2_scene_dir / "annotations.feather"

    # If there is no annotation file, create an empty HDF5 so the merge step
    # knows this scene exists but has no BBOXes.
    if not ann_path.exists():
        with h5py.File(temp_output_path, "w") as f:
            pass
        return temp_output_path

    # Read all cuboids for this scene and index them by timestamp.
    cuboid_list = CuboidList.from_feather(ann_path)
    raw_data = read_feather(ann_path)
    track_uuids = raw_data.track_uuid.to_numpy()

    timestamp_cuboid_index: Dict[int, Dict[str, Cuboid]] = defaultdict(dict)
    for track_uuid, cuboid in zip(track_uuids, cuboid_list.cuboids):
        timestamp_cuboid_index[cuboid.timestamp_ns][track_uuid] = cuboid

    # Sensor (up_lidar) calibration: E = ego_SE3_sensor, so E.inverse() maps
    # ego -> sensor.
    try:
        ego2sensor = read_ego_SE3_sensor(av2_scene_dir)["up_lidar"]
        sensor2ego = ego2sensor.inverse()
    except Exception as exc:
        print(f"[{log_id}] Failed to read ego_SE3_sensor calibration: {exc}")
        with h5py.File(temp_output_path, "w") as f:
            pass
        return temp_output_path

    # Determine which timestamps we need from the already preprocessed HDF5.
    try:
        with h5py.File(h5py_scene_file, "r") as f:
            target_timestamps = sorted(int(k) for k in f.keys() if k.isdigit())
    except Exception as exc:
        print(f"[{log_id}] Failed to read preprocessed HDF5 {h5py_scene_file}: {exc}")
        with h5py.File(temp_output_path, "w") as f:
            pass
        return temp_output_path

    str_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(temp_output_path, "w") as f:
        scene_group = f.create_group(log_id)
        # Store the up_lidar extrinsic so the visualizer can warp BBOXes/points
        # exactly from sensor_i to sensor_0 without re-reading raw AV2 files.
        scene_group.create_dataset(
            "ego_SE3_sensor",
            data=ego2sensor.transform_matrix.astype(np.float32),
        )

        for ts in target_timestamps:
            cuboids = timestamp_cuboid_index.get(ts, {})
            if not cuboids:
                continue

            centers, extents, quats, categories, uuids = [], [], [], [], []
            for track_uuid, cuboid in cuboids.items():
                # Transform cuboid from ego frame to sensor frame.
                cuboid_sensor = cuboid.transform(sensor2ego)
                centers.append(cuboid_sensor.xyz_center_m)
                extents.append(cuboid_sensor.dims_lwh_m)
                quats.append(
                    _rotation_matrix_to_quat(cuboid_sensor.dst_SE3_object.rotation)
                )
                categories.append(str(cuboid_sensor.category))
                uuids.append(track_uuid)

            centers = np.asarray(centers, dtype=np.float32)
            extents = np.asarray(extents, dtype=np.float32)
            quats = np.asarray(quats, dtype=np.float32)
            categories_arr = np.asarray(categories, dtype=str_dtype)
            uuids_arr = np.asarray(uuids, dtype=str_dtype)

            ts_group = scene_group.create_group(str(ts))
            ts_group.create_dataset("centers", data=centers)
            ts_group.create_dataset("extents", data=extents)
            ts_group.create_dataset("quaternions", data=quats)
            ts_group.create_dataset("categories", data=categories_arr)
            ts_group.create_dataset("track_uuids", data=uuids_arr)

    return temp_output_path


def _merge_temp_files(temp_files: List[Path], output_path: Path) -> None:
    """Merge per-scene temp HDF5 files into a single bbox.h5."""
    if output_path.exists():
        output_path.unlink()

    with h5py.File(output_path, "w") as out_f:
        for temp_file in tqdm(temp_files, desc="Merging scenes", ncols=100):
            if not temp_file.exists():
                continue
            with h5py.File(temp_file, "r") as in_f:
                for scene_id in in_f.keys():
                    in_f.copy(in_f[scene_id], out_f, name=scene_id)


def main(
    av2_dir: str = "/home/kin/data/av2/sensor/train",
    h5py_dir: str = "/home/kin/data/av2/h5py/sensor/train",
    nproc: int = 1,
):
    """Precompute AV2 BBOXes into a single HDF5 file.

    Args:
        av2_dir: Path to raw AV2 sensor data (contains scene directories).
        h5py_dir: Path to preprocessed HDF5 files; `bbox.h5` will be written here.
        nproc: Number of parallel workers. Default 1 (serial).
    """
    av2_dir = Path(av2_dir)
    h5py_dir = Path(h5py_dir)
    output_path = h5py_dir / "bbox.h5"

    if not av2_dir.exists():
        raise FileNotFoundError(f"AV2 directory not found: {av2_dir}")
    if not h5py_dir.exists():
        raise FileNotFoundError(f"HDF5 directory not found: {h5py_dir}")

    # Build list of scenes that have both raw data and a preprocessed HDF5.
    scene_dirs = sorted(d for d in av2_dir.iterdir() if d.is_dir())
    tasks: List[Tuple[Path, Path, Path]] = []
    for scene_dir in scene_dirs:
        h5py_scene_file = h5py_dir / f"{scene_dir.name}.h5"
        if not h5py_scene_file.exists():
            print(f"Skipping {scene_dir.name}: no preprocessed HDF5 found.")
            continue
        tasks.append((scene_dir, None, h5py_scene_file))

    if not tasks:
        raise RuntimeError("No scenes to process. Check --av2_dir and --h5py_dir.")

    # Create a temporary directory for per-scene outputs.
    temp_dir = Path(tempfile.mkdtemp(prefix="bbox_precompute_"))
    temp_files: List[Path] = []

    try:
        # Assign concrete temp output paths.
        tasks = [
            (scene_dir, temp_dir / f"{scene_dir.name}_bbox.h5", h5py_file)
            for scene_dir, _, h5py_file in tasks
        ]
        temp_files = [t[1] for t in tasks]

        if nproc <= 1:
            for task in tqdm(tasks, desc="Processing scenes", ncols=100):
                _process_one_scene(task)
        else:
            with Pool(processes=min(nproc, os.cpu_count() or 1)) as pool:
                list(
                    tqdm(
                        pool.imap_unordered(_process_one_scene, tasks),
                        total=len(tasks),
                        desc="Processing scenes",
                        ncols=100,
                    )
                )

        _merge_temp_files(temp_files, output_path)
        print(f"BBOX precomputation complete: {output_path}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    fire.Fire(main)
