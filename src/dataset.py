"""
# Created: 2023-11-04 15:52
# Updated: 2024-07-12 23:16
# 
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), Jaeyeul Kim (jykim94@dgist.ac.kr)
#
# Change Logs:
# 2024-11-06: Added Data Augmentation transform for RandomHeight, RandomFlip, RandomJitter from DeltaFlow project.
# 2024-07-12: Merged num_frame based on Flow4D model from Jaeyeul Kim.
# 
# Description: Torch dataloader for the dataset we preprocessed.
# 
# This file is part of 
# * OpenSceneFlow (https://github.com/KTH-RPL/OpenSceneFlow)
# 
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""

import torch, re
from torch.utils.data import Dataset, DataLoader
import h5py, pickle, argparse
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils import import_func

def extract_flow_number(key):
    digits = re.findall(r'\d+$', key)
    if digits:
        return digits[0]
    return '0'

# FIXME(Qingwen 2025-08-20): update more pretty here afterward!
def collate_fn_pad(batch):
    """将变长点云/flow样本拼成固定形状的batch tensor。

    对batch中每个样本，先按ground_mask去掉地面点，再将同一帧（pc0/pc1/pch*）
    或同一flow按最大点数padding。点云padding值为NaN，flow和动态标签padding值为0。
    """
    batch_size_ = len(batch)
    pcs_after_mask_ground = defaultdict(list)
    flows_after_mask_ground = defaultdict(list)
    poses_dict = defaultdict(list)

    # 点云及flow去地面；位姿直接收集（无需去地面）
    for i in range(batch_size_):
        single_data = batch[i]  # 单帧点云及其邻近帧
        for key in single_data.keys():
            if key.startswith('pc') and f'gm{key[2:]}' in single_data and not key.endswith("dynamic"):
                gm_key = f'gm{key[2:]}'
                # key: pc0, pc1, pch1, pch2, ...; value: 去掉地面后的点云
                pcs_after_mask_ground[key].append(single_data[key][~single_data[gm_key]])
            elif key.startswith('flow'):
                id_flow = extract_flow_number(key)
                gm_key = f'gm{id_flow}'
                flows_after_mask_ground[key].append(single_data[key][~single_data[gm_key]])
            elif key.startswith('pose'):
                poses_dict[key].append(single_data[key])

    # 按batch内最大点数padding；点云用NaN，便于后续用torch.isnan过滤padding
    for key in pcs_after_mask_ground:
        pcs_after_mask_ground[key] = torch.nn.utils.rnn.pad_sequence(
            pcs_after_mask_ground[key], batch_first=True, padding_value=torch.nan
        )
    for key in flows_after_mask_ground:
        flows_after_mask_ground[key] = torch.nn.utils.rnn.pad_sequence(
            flows_after_mask_ground[key], batch_first=True, padding_value=0
        )

    # 把点云、flow GT、位姿汇总进res_dict
    res_dict = {key: pcs_after_mask_ground[key] for key in pcs_after_mask_ground}
    res_dict.update({key: flows_after_mask_ground[key] for key in flows_after_mask_ground})
    res_dict.update({key: [poses_dict[key][i] for i in range(batch_size_)] for key in poses_dict})

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(batch_size_)]

    if 'pc0_dynamic' in batch[0]:
        pc0_dynamic_after_mask_ground, pc1_dynamic_after_mask_ground = [], []
        for i in range(batch_size_):
            pc0_dynamic_after_mask_ground.append(batch[i]['pc0_dynamic'][~batch[i]['gm0']])
            pc1_dynamic_after_mask_ground.append(batch[i]['pc1_dynamic'][~batch[i]['gm1']])
        res_dict['pc0_dynamic'] = torch.nn.utils.rnn.pad_sequence(
            pc0_dynamic_after_mask_ground, batch_first=True, padding_value=0
        )
        res_dict['pc1_dynamic'] = torch.nn.utils.rnn.pad_sequence(
            pc1_dynamic_after_mask_ground, batch_first=True, padding_value=0
        )
    if 'pch1_dynamic' in batch[0]:
        pch_dynamic_after_mask_ground = [
            batch[i]['pch1_dynamic'][~batch[i]['gmh1']] for i in range(batch_size_)
        ]
        res_dict['pch1_dynamic'] = torch.nn.utils.rnn.pad_sequence(
            pch_dynamic_after_mask_ground, batch_first=True, padding_value=0
        )

    res_dict['scene_id'] = [batch[i]['scene_id'] for i in range(batch_size_)]
    return res_dict

# transform, augment
class RandomJitter(object):
    "Randomly add small noise to the point cloud."
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        for key in data_dict.keys():
            if key.startswith("pc") and not key.endswith("dynamic"):
                jitter = np.clip(
                    self.sigma * np.random.randn(data_dict[key].shape[0], 3),
                    -self.clip,
                    self.clip,
                )
                data_dict[key] += jitter
        return data_dict

class RandomFlip(object):
    def __init__(self, p=0.5, verbose=False):
        """p: probability of flipping"""
        self.p = p
        self.verbose = verbose

    def __call__(self, data_dict):
        flip_x = np.random.rand() < self.p
        flip_y = np.random.rand() < self.p

        # If no flip, return directly
        if not (flip_x or flip_y):
            return data_dict
        
        for key in data_dict.keys():
            if (key.startswith("pc") or (key.startswith("flow") and data_dict[key].dtype == np.float32)) and not key.endswith("dynamic"):
                if flip_x:
                    data_dict[key][:, 0] = -data_dict[key][:, 0]
                if flip_y:
                    data_dict[key][:, 1] = -data_dict[key][:, 1]
            if key.startswith("pose"):
                if flip_x:
                    pose = data_dict[key].copy()
                    pose[:, 0] *= -1
                    data_dict[key] = pose
                if flip_y:
                    pose = data_dict[key].copy()
                    pose[:, 1] *= -1
                    data_dict[key] = pose

        if "ego_motion" in data_dict:
            # need recalculate the ego_motion
            data_dict["ego_motion"] = np.linalg.inv(data_dict['pose1']) @ data_dict['pose0']
        if self.verbose:
            print(f"RandomFlip: flip_x={flip_x}, flip_y={flip_y}")
        return data_dict

class RandomHeight(object):
    def __init__(self, p=0.5, verbose=False):
        """p: probability of changing height"""
        self.p = p
        self.verbose = verbose

    def __call__(self, data_dict):
        # NOTE(Qingwen): The reason set -0.5 to 2.0 is because some dataset axis origin is around the ground level. (vehicle base etc.)
        random_height = np.random.uniform(-0.5, 2.0)
        if np.random.rand() < self.p:
            for key in data_dict.keys():
                if key.startswith("pc") and not key.endswith("dynamic"):
                    data_dict[key][:, 2] += random_height
            if self.verbose:
                print(f"RandomHeight: {random_height}")
        return data_dict

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, data_dict):
        for key in data_dict.keys():
            # skip the scene_id, timestamp, eval_flag to tensor conversion
            if key in ['scene_id', 'timestamp', 'eval_flag']:
                continue
            elif isinstance(data_dict[key], np.ndarray):
                data_dict[key] = torch.tensor(data_dict[key])
            else:
                print(f"Warning: {key} is not a numpy array. Type: {type(data_dict[key])}")
        return data_dict

class HDF5Dataset(Dataset):
    """Torch Dataset backed by preprocessed HDF5 files.

    Each sample returned by ``__getitem__`` is centered on one frame (the
    current frame ``pc0``) and includes its neighboring frames: one future
    frame (``pc1``) and zero or more history frames (``pch1``, ``pch2``, ...).
    The total number of frames is controlled by ``n_frames``:

    - ``n_frames=2``: current frame ``pc0`` + next frame ``pc1``.
    - ``n_frames>2``: current frame + next frame + ``n_frames - 2`` past frames.

    The dataset directory must contain:
      - ``*.h5`` files, one per scene. Each file is keyed by timestamp strings.
      - ``index_total.pkl``: a ``List[List[str]]`` where each item is
        ``[scene_id, timestamp]``. This is the canonical full index.
      - ``index_eval.pkl`` (optional): same shape as ``index_total.pkl``,
        subset used for leaderboard evaluation.
      - ``index_flow.pkl`` (optional): same shape as ``index_total.pkl``,
        subset containing only frames with GT flow annotations.

    HDF5 file structure (one file per scene, e.g. ``{scene_id}.h5``):

    .. code-block:: text

        {scene_id}.h5
        └── {timestamp}/                # Group, one per frame
            ├── lidar                   # (N, 3) float32, point cloud xyz in sensor (up_lidar) frame
            ├── ground_mask             # (N,) bool, True for ground points
            ├── pose                    # (4, 4) float32, ego -> city transform (city_SE3_ego)
            ├── ego_motion              # (4, 4) float32, ego0 -> ego1 transform (ego1_SE3_ego0)
            ├── lidar_dt                # (N,) float32, per-point time delta
            ├── lidar_id                # (N,) uint8, LiDAR beam/ring id
            ├── flow                    # (N, 3) float32, GT scene flow in source ego frame (optional)
            ├── flow_is_valid           # (N,) bool, GT validity mask (optional)
            ├── flow_category_indices   # (N,) uint8, per-point category (optional)
            └── flow_instance_id        # (N,) int16, per-point instance id (optional)

    Notes:
      - ``lidar`` is read as ``f[timestamp]['lidar'][:][:, :3]`` to keep only
        xyz coordinates even if the stored array has extra channels.
      - Coordinate frames follow the AV2 ``dst_SE3_src`` convention:
        ``lidar`` is in the sensor (up_lidar) frame; ``pose`` transforms points
        from ego to city; ``ego_motion`` transforms points from ego0 to ego1;
        ``flow`` is defined in the source ego frame.
      - Fields marked ``(optional)`` may be absent depending on the dataset
        (e.g. test sets usually lack ``flow``; some datasets lack
        ``flow_category_indices``).
      - Additional visualization keys can be loaded via ``vis_name``.

    Data layout assumptions (verified for AV2 / nuScenes / demo):
      1. ``index_total.pkl`` lists frames grouped by ``scene_id``. All frames
         belonging to the same scene appear consecutively and are sorted by
         ``timestamp`` in ascending order.
      2. ``__getitem__`` relies on the next frame being at
         ``data_index[index_ + 1]`` (``pc1``) and history frames at
         ``data_index[index_ - i]`` (``pch1``, ``pch2``, ...). Therefore the
         caller must not request the last frame of a scene or frames too close
         to the beginning of a scene.
      3. Eval / train subsets (``index_eval.pkl`` / ``index_flow.pkl``) do NOT
         contain a separate data copy. They only list ``[scene_id, timestamp]``
         entries that exist in ``index_total.pkl``; the actual point cloud data
         is still read from the same ``*.h5`` files in ``directory``.
    """

    # Type aliases for readability (Python 3.8 compatible)
    IndexEntry = List[str]                # [scene_id: str, timestamp: str]
    DataIndex = List[IndexEntry]
    SceneBounds = Dict[str, Dict[str, Any]]

    def __init__(self,
                 directory: str,
                 transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
                 n_frames: int = 2,
                 ssl_label: Optional[str] = None,
                 eval: bool = False,
                 leaderboard_version: int = 1,
                 vis_name: str = '',
                 index_flow: bool = False) -> None:
        """
        Args:
            directory: Path to the dataset folder containing ``.h5`` files and
                ``index_total.pkl``.
            transform: Optional data-augmentation callable applied to a sample dict.
            n_frames: Number of frames to load per sample. ``n_frames=2`` loads
                the current frame (``pc0``) and the next frame (``pc1``); values
                >2 additionally load past frames (``pch1``, ``pch2``, ...).
            ssl_label: Name of the auto-label function in ``src.autolabel`` used
                to load dynamic cluster labels. ``None`` means no cluster labels.
            eval: If True, load the eval subset for leaderboard submission.
            leaderboard_version: 1 or 2; version 2 uses ``index_eval_v2.pkl``.
            vis_name: Extra HDF5 key(s) to load for visualization.
            index_flow: If True, use ``index_flow.pkl`` to skip frames without GT flow.
        """
        super(HDF5Dataset, self).__init__()
        self.directory: str = directory
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) \
                or not torch.distributed.is_initialized():
            print(f"----[Debug] Loading data with num_frames={n_frames}, "
                  f"ssl_label={ssl_label}, eval={eval}, leaderboard_version={leaderboard_version}")

        # Canonical full index: List[[scene_id, timestamp], ...].
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index: HDF5Dataset.DataIndex = pickle.load(f)

        self.eval_index: bool = False
        self.ssl_label: Optional[Callable[[h5py.Group], np.ndarray]] = \
            import_func(f"src.autolabel.{ssl_label}") if ssl_label is not None else None
        self.history_frames: int = n_frames - 2
        self.vis_name: List[str] = vis_name if isinstance(vis_name, list) else [vis_name]
        self.transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = transform

        # Eval index fallback path: eval -> eval_v2 -> flow -> panic.
        if eval:
            eval_index_file = os.path.join(self.directory, 'index_eval.pkl')
            if leaderboard_version == 2:
                print("Using index to leaderboard version 2!!")
                eval_index_file = os.path.join(BASE_DIR, 'assets/docs/index_eval_v2.pkl')

            if not os.path.exists(eval_index_file):
                print(f"Warning: No {eval_index_file} file found! We will try {'index_flow.pkl'}")
                eval_index_file = os.path.join(self.directory, 'index_flow.pkl')
                if not os.path.exists(eval_index_file):
                    raise Exception(f"No any eval index file found! Please check {self.directory}")

            self.eval_index = eval
            with open(eval_index_file, 'rb') as f:
                self.eval_data_index: HDF5Dataset.DataIndex = pickle.load(f)

        # Per-scene statistics: maps scene_id -> {min_timestamp, max_timestamp,
        # min_index, max_index} in the canonical data_index.
        self.scene_id_bounds: HDF5Dataset.SceneBounds = {}
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp, "max_timestamp": timestamp,
                    "min_index": idx, "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

        # Optional training subset used when not every frame has GT flow
        # (e.g., truckscene or nuscene with different annotation rates).
        self.train_index: Optional[HDF5Dataset.DataIndex] = None
        if (not eval and ssl_label is None and transform is not None) or index_flow:
            one_scene_id = list(self.scene_id_bounds.keys())[0]
            check_flow_exist = True
            with h5py.File(os.path.join(self.directory, f'{one_scene_id}.h5'), 'r') as f:
                for i in range(self.scene_id_bounds[one_scene_id]["min_index"],
                               self.scene_id_bounds[one_scene_id]["max_index"]):
                    scene_id, timestamp = self.data_index[i]
                    key = str(timestamp)
                    if 'flow' not in f[key]:
                        check_flow_exist = False
                        break
            if not check_flow_exist:
                print("----- [Warning]: Not all frames have flow data, "
                      "we will instead use the index_flow.pkl to train.")
                self.train_index = pickle.load(
                    open(os.path.join(self.directory, 'index_flow.pkl'), 'rb'))
                
    def __len__(self) -> int:
        """Return the number of samples for the active mode."""
        if self.eval_index:
            return len(self.eval_data_index)
        elif not self.eval_index and self.train_index is not None:
            return len(self.train_index)
        return len(self.data_index)

    def valid_index(self, index_: int) -> int:
        """Map an external sample index to the canonical ``data_index`` position.

        For eval/train subsets, this resolves the subset entry back to its
        position in ``self.data_index``. For the full index, it clamps the
        index so that history frames and the next frame are available.

        The clamping / recursion is necessary because ``__getitem__`` reads
        ``data_index[index_ + 1]`` as the next frame (``pc1``) and
        ``data_index[index_ - i]`` as history frames (``pch1``, ...). Those
        offsets are only valid when ``index_`` lies inside the interior of a
        scene's contiguous block in ``data_index`` (see class docstring).

        In eval/train subset mode, if the requested frame happens to be the
        last frame of its scene, we recursively fall back to the previous
        subset entry. This preserves the subset length while avoiding an
        out-of-bounds access to ``data_index[index_ + 1]``.

        Args:
            index_: Sample index requested by the DataLoader.

        Returns:
            The corresponding canonical index in ``self.data_index``.
        """
        subset_index = None
        if self.eval_index:
            subset_index = self.eval_data_index
        elif self.train_index is not None:
            subset_index = self.train_index

        if subset_index is not None:
            subset_index_ = index_
            scene_id, timestamp = subset_index[subset_index_]
            index_ = self.data_index.index([scene_id, timestamp])
            max_idx = self.scene_id_bounds[scene_id]["max_index"]
            if index_ >= max_idx:
                index_ = self.valid_index(subset_index_ - 1)
        else:
            scene_id, timestamp = self.data_index[index_]
            max_idx = self.scene_id_bounds[scene_id]["max_index"]
            min_idx = self.scene_id_bounds[scene_id]["min_index"]

            max_valid_index_for_flow = max_idx - 1
            min_valid_index_for_flow = min_idx + self.history_frames
            index_ = max(min_valid_index_for_flow, min(max_valid_index_for_flow, index_))
        return index_

    def __getitem__(self, index_: int) -> Dict[str, Any]:
        """Load one sample centered on a single frame.

        The returned ``data_dict`` contains the current frame ``pc0`` plus its
        neighboring frames (``pc1`` for the next frame, ``pch{i}`` for history
        frames) and all associated per-frame information.

        Core fields (always present):
          - ``scene_id`` (str): UUID of the scene.
          - ``timestamp`` (str): Current frame timestamp.
          - ``eval_flag`` (bool): Whether this sample belongs to an eval subset.
          - ``pc0`` (np.ndarray, (N, 3)): Current-frame point cloud in sensor frame.
          - ``gm0`` (np.ndarray, (N,) bool): Ground mask for ``pc0``.
          - ``pose0`` (np.ndarray, (4, 4)): Ego pose of the current frame,
            i.e. ``ego -> city`` transform matrix.

        Future frame (``history_frames >= -1``, i.e. always when ``n_frames >= 2``):
          - ``pc1`` (np.ndarray, (M, 3)): Next-frame point cloud in sensor frame.
          - ``gm1`` (np.ndarray, (M,) bool): Ground mask for ``pc1``.
          - ``pose1`` (np.ndarray, (4, 4)): Ego pose of the next frame,
            i.e. ``ego -> city`` transform matrix.

        History frames (``history_frames > 0``, i.e. ``n_frames > 2``):
          - ``pch{i+1}`` (np.ndarray): Point cloud of the i-th past frame in sensor frame.
          - ``gmh{i+1}`` (np.ndarray): Ground mask of the i-th past frame.
          - ``poseh{i+1}`` (np.ndarray): Ego pose of the i-th past frame,
            i.e. ``ego -> city`` transform matrix.

        Dynamic cluster labels (only if ``ssl_label`` is provided):
          - ``pc0_dynamic`` (np.ndarray): SSL cluster labels for ``pc0``.
              - ``0``: background / static points (including ground).
              - ``1``: dynamic points without a cluster id (unclustered dynamic).
              - ``2+``: dynamic cluster instance IDs.
          - ``pc1_dynamic`` (np.ndarray): SSL cluster labels for ``pc1``,
            same semantics as ``pc0_dynamic``.
          - ``pch1_dynamic`` (np.ndarray): SSL cluster labels for ``pch1``,
            same semantics as ``pc0_dynamic``.

        Optional HDF5 fields (present only when stored in the file):
          - ``ego_motion`` (np.ndarray, (4, 4)): Ego-motion transform from the
            current frame to the next frame, i.e. ``ego0 -> ego1``.
          - ``lidar_dt`` (float): Time delta between frames.
          - ``lidar_center`` (np.ndarray): LiDAR sensor center transform(s).
          - ``flow`` (np.ndarray, (N, 3)): Ground-truth scene flow, defined in the
            source ego frame.
          - ``flow_is_valid`` (np.ndarray, (N,) bool): Per-point flow validity.
          - ``flow_category_indices`` (np.ndarray): Per-point category labels.
          - ``flow_instance_id`` (np.ndarray): Per-point instance IDs.
          - ``dufo``: DUFO-related dynamic/static labels.

        Eval-only fields (only when ``eval=True``):
          - ``eval_mask`` (np.ndarray, (N,) bool): Points to include in leaderboard
            evaluation (ground points removed).
        """
        index_ = self.valid_index(index_) # 如果index_不小心在整个场景的前几帧或者最后一帧，偏移index_使得刚好能有历史帧和未来帧
        eval_flag = self.eval_index  # eval模式由构造参数决定，不需要valid_index返回
        scene_id, timestamp = self.data_index[index_]

        key = str(timestamp)
        data_dict = {
            'scene_id': scene_id,
            'timestamp': timestamp,
            'eval_flag': eval_flag
        }
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            # original data
            data_dict['pc0'] = f[key]['lidar'][:][:,:3] # 中间的[:]是复制整个(N,3)，这会触发读入内存操作。
            data_dict['gm0'] = f[key]['ground_mask'][:]
            data_dict['pose0'] = f[key]['pose'][:]
            if self.ssl_label is not None:
                data_dict['pc0_dynamic'] = self.ssl_label(f[key])

            if self.history_frames >= 0: 
                # 未来一帧加载
                next_timestamp = str(self.data_index[index_ + 1][1])
                data_dict['pose1'] = f[next_timestamp]['pose'][:]
                data_dict['pc1'] = f[next_timestamp]['lidar'][:][:,:3]
                data_dict['gm1'] = f[next_timestamp]['ground_mask'][:]
                if self.ssl_label is not None:
                    data_dict['pc1_dynamic'] = self.ssl_label(f[next_timestamp])
                
                # 历史帧加载
                past_frames = []
                for i in range(1, self.history_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = f[past_timestamp]['lidar'][:][:,:3]
                    past_gm = f[past_timestamp]['ground_mask'][:]
                    past_pose = f[past_timestamp]['pose'][:]

                    past_frames.append((past_pc, past_gm, past_pose))
                    if i == 1 and self.ssl_label is not None: # only for history 1: t-1
                        # data_dict['pch1_dynamic'] = f[past_timestamp]['label'][:].astype('int16')
                        data_dict['pch1_dynamic'] = self.ssl_label(f[past_timestamp])

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    data_dict[f'pch{i+1}'] = past_pc
                    data_dict[f'gmh{i+1}'] = past_gm
                    data_dict[f'poseh{i+1}'] = past_pose

            for data_key in self.vis_name + ['ego_motion', 'lidar_dt', 'lidar_center',
                             # ground truth information:
                             'flow', 'flow_is_valid', 'flow_category_indices', 'flow_instance_id', 'dufo']:
                if data_key in f[key]:
                    data_dict[data_key] = f[key][data_key][:]

            if self.eval_index:
                # looks like v2 not follow the same rule as v1 with eval_mask provided
                if 'eval_mask' in f[key]:
                    raw_eval = f[key]['eval_mask'][:]
                    raw_ground = f[key]['ground_mask'][:]
                    # NOTE(Qingwen): performance might be changed for av2 since some eval_mask provided by av2 didn't remove ground points.
                    data_dict['eval_mask'] = (raw_eval.reshape(-1).astype(bool) & (~raw_ground.reshape(-1).astype(bool)))
                elif 'ground_mask' in f[key]:
                    data_dict['eval_mask'] = ~f[key]['ground_mask'][:]
                else:
                    data_dict['eval_mask'] = np.ones_like(data_dict['pc0'][:, 0], dtype=np.bool_)
                    
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataLoader test")
    parser.add_argument('--data_mode', '-m', type=str, default='train', metavar='N', help='Dataset mode.')
    parser.add_argument('--data_dir', '-d', type=str, default='/home/kin/data/av2/h5py_v2/sensor', metavar='N', help='preprocess data path.')
    options = parser.parse_args()

    # testing eval mode
    dataset = HDF5Dataset(directory = options.data_dir+"/"+options.data_mode, eval = False,
                          transform = transforms.Compose([RandomHeight(), RandomFlip(), RandomJitter(), ToTensor()]))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn_pad)
    for data in tqdm(dataloader, ncols=80, desc="read data mode"):
        res_dict = data
        # print(res_dict['pc0'].shape)
        # break