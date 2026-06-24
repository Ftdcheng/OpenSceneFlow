"""
# Created: 2023-11-04 15:55
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of 
# * HiMo (https://kin-zhang.github.io/HiMo).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: follow seflow & seflow++ idea but more to run:
# (a) dufomap; (b) linefit; (c) hdbscan; (d) nnd.
#
# Default behavior when running `python process.py`:
#   - fire.Fire(main) generates dufo, ground_mask and cluster labels.
#   - fire.Fire(run_dufocluster) then generates dufocluster labels.
#   - overwrite=True means dufo labels are regenerated even if they already exist.
#   - run_gm=False means existing ground_mask in the HDF5 files is reused.
#   - scene_range=[0, 1] processes only the first scene (left-closed, right-open).
#   - interval=1 means no frame skipping when integrating into DUFOMap.
#
"""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import fire, time, h5py, os, sys
from hdbscan import HDBSCAN

from src.utils import npcal_pose0to1
from src.utils.mics import HDF5Data, transform_to_array
from dufomap import dufomap
from linefit import ground_seg

MIN_AXIS_RANGE = 2 # HARD CODED: remove ego vehicle points
MAX_AXIS_RANGE = 50 # HARD CODED: remove far away points
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ )))

def check_data_key(filekey, keyname, scene_id, ts):
    """
    Checks if a specified key exists in the given file-like object and deletes it if present.

    This is typically used to remove old or outdated data before overwriting with new data.
    
    Args:
        filekey: The file-like object (e.g., HDF5 group or dictionary) to check for the key.
        keyname: The name of the key to check and potentially delete.
        scene_id: Identifier for the scene (used for logging or warning purposes).
        ts: Timestamp or frame identifier (used for logging or warning purposes).
    """
    if keyname in filekey:
        # print(f"Warning: {scene_id} {ts} has {keyname}, old data will be removed and overwritten.")
        del filekey[keyname]

def run_dufocluster(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/train",
    scene_range: list = [0, 1],
    interval: int = 1, # interval frames to run dufomap only
    overwrite: bool = True,

    # NOTE (Qingwen): following is for ground segmentation only... at least for code now.
    tag: str = "av2", # [nus, zod, man, sca] etc
    run_gm: bool = False, # run ground segmentation
    min_nnd: float = 0.14, # min nnd distance 1.4m/s pedestrain speed; For Scania data, we set 0.32 here.
):
    """基于已有的 ``dufo`` 动态标签，用 HDBSCAN 对动态点聚类并写入 ``dufocluster``。

    该函数通常在 ``main`` 之后被 ``fire.Fire`` 自动调用。它遍历每个场景，读取
    每帧的 ``dufo`` 动态点掩码，对动态点运行 HDBSCAN 聚类，将实例级标签写入
    HDF5 的 ``dufocluster`` 数据集：

    - ``0``：非动态点或未能聚类的点；
    - 正整数：不同的动态实例聚类 ID。

    默认行为：
      - ``overwrite=True``：即使已存在 ``dufocluster`` 也会重新生成并覆盖。
      - 如果某帧缺少 ``dufo`` 标签或动态点少于 20 个，则跳过该帧。

    Args:
        data_dir: 存放预处理 ``.h5`` 文件的目录。
        scene_range: 处理的场景索引范围 ``[start, end)``；``[-1, -1]`` 表示全部。
        interval: 未在本函数中使用，保留以兼容命令行参数。
        overwrite: 是否强制重新生成 ``dufocluster`` 标签。
        tag: 地面分割配置标识（本函数中未使用）。
        run_gm: 是否重新跑地面分割（本函数中未使用）。
        min_nnd: NND 动态点阈值（本函数中未使用）。
    """
    data_path = Path(data_dir)
    dataset = HDF5Data(data_path) # single frame reading.
    all_scene_ids = list(dataset.scene_id_bounds.keys())
    for scene_in_data_index, scene_id in enumerate(all_scene_ids):
        start_time = time.time()
        # NOTE (Qingwen): so the scene id range is [start, end)
        if scene_range[0]!= -1 and scene_range[-1]!= -1 and (scene_in_data_index < scene_range[0] or scene_in_data_index >= scene_range[1]):
            continue
        bounds = dataset.scene_id_bounds[scene_id]
        flag_exist_label = True
        with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
            for ii in range(bounds["min_index"], bounds["max_index"]+1):
                key = str(dataset[ii]['timestamp'])
                if 'dufocluster' not in f[key]:
                    flag_exist_label = False
                    break
        if flag_exist_label and not overwrite:
            print(f"==> Scene {scene_id} has plus label, skip.")
            continue
        
        hdb = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon=0.7)
        for i in tqdm(range(bounds["min_index"], bounds["max_index"]+1), desc=f"Start Plus Cluster: {scene_in_data_index}/{len(all_scene_ids)}", ncols=80):
            data = dataset[i]
            pc0 = data['pc0'][:,:3]
            cluster_label = np.zeros(pc0.shape[0], dtype= np.int16)

            if "dufo" not in data:
                print(f"Warning: {scene_id} {data['timestamp']} has no dufo, will be skipped. Better to rerun dufomap again in this scene.")
                continue
            elif data["dufo"].sum() < 20:
                print(f"Warning: {scene_id} {data['timestamp']} has no dynamic points, will be skipped. Better to check this scene.")
            else:
                hdb.fit(pc0[data["dufo"]==1])
                # NOTE(Qingwen): since -1 will be assigned if no cluster. We set it to 0.
                cluster_label[data["dufo"]==1] = hdb.labels_ + 1 

            # save labels
            timestamp = data['timestamp']
            key = str(timestamp)
            with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
                if 'dufocluster' in f[key]:
                    # print(f"Warning: {scene_id} {timestamp} has label, will be overwritten.")
                    del f[key]['dufocluster']
                f[key].create_dataset('dufocluster', data=np.array(cluster_label).astype(np.int16))
        print(f"==> Scene {scene_id} finished, used: {(time.time() - start_time)/60:.2f} mins")
    print(f"Data inside {str(data_path)} finished. Check the result with tools/visulization.py if you want to visualize them.\n")

# since it's the only one need cuda to do. in case you want to run two single jobs.
def run_nnd(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/train",
    scene_range: list = [0, 1],
    interval: int = 1, # interval frames to run dufomap only
    overwrite: bool = True,

    # NOTE (Qingwen): following is for ground segmentation only... at least for code now.
    tag: str = "av2", # [nus, zod, man, sca] etc
    run_gm: bool = False, # run ground segmentation
    min_nnd: float = 0.14, # min nnd distance 1.4m/s pedestrain speed; For Scania data, we set 0.32 here.
):
    """使用 CUDA 加速的最近邻距离（NND）生成动态点标签。

    该函数目前未被 ``__main__`` 默认调用（末尾被注释掉）。它通过比较当前帧点云
    与相邻帧点云在补偿 ego motion 后的 Chamfer 距离，把距离超过 ``min_nnd`` 的
    点标记为动态。

    需要 GPU 支持。

    Args:
        data_dir: 存放预处理 ``.h5`` 文件的目录。
        scene_range: 处理的场景索引范围 ``[start, end)``。
        interval: 未在本函数中使用，保留以兼容命令行参数。
        overwrite: 是否强制重新生成 ``nnd`` 标签。
        tag: 地面分割配置标识（本函数中未使用）。
        run_gm: 是否重新跑地面分割（本函数中未使用）。
        min_nnd: 判定为动态点的最小距离阈值（单位：米）。
    """
    # nnd function
    from assets.cuda.chamfer3D import nnChamferDis
    MyCUDAChamferDis = nnChamferDis()
    import torch
    if not torch.cuda.is_available():
        raise EnvironmentError("No cuda available, please check your cuda environment.")
    # exit()
    def cuda_nnd(pc0: torch.tensor, pc1: torch.tensor, moving_threshold=0.14, truncated=4.4): # 4.4 ~= 160km/h * 0.1s
        # pc0: (N,3), already ego motion transformed; pc1: (M,3)
        pow2_dist0, _= MyCUDAChamferDis.dis_res(pc0, pc1)
        pow2_dist0 = pow2_dist0.cpu().numpy()
        label = np.zeros(pc0.shape[0], dtype=np.uint8)
        label[(pow2_dist0>=pow(moving_threshold,2)) & (pow2_dist0<pow(truncated,2))] = 1
        return label

    data_path = Path(data_dir)
    dataset = HDF5Data(data_path) # single frame reading.
    all_scene_ids = list(dataset.scene_id_bounds.keys())
    for scene_in_data_index, scene_id in enumerate(all_scene_ids):
        start_time = time.time()
        # NOTE (Qingwen): so the scene id range is [start, end)
        if scene_range[0]!= -1 and scene_range[-1]!= -1 and (scene_in_data_index < scene_range[0] or scene_in_data_index >= scene_range[1]):
            continue
        bounds = dataset.scene_id_bounds[scene_id]
        exist_dict = {"nnd": True}
        with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r') as f:
            for ii in range(bounds["min_index"], bounds["max_index"]+1):
                key = str(dataset[ii]['timestamp'])
                for datakey in exist_dict.keys():
                    if datakey not in f[key]:
                        exist_dict[datakey] = False
                if not all(exist_dict.values()):
                    break
        
        if all(exist_dict.values()) and not overwrite:
            print(f"==> Scene {scene_id} already processed, skip.")
            continue
        
        ego_pose_norm = dataset[bounds["min_index"]]['pose0']
        for data_id in tqdm(range(bounds["min_index"], bounds["max_index"]+1), desc=f"CUDA nnd run: {scene_in_data_index+1}/{len(all_scene_ids)}", ncols=80):
            data = dataset[data_id]
            pc0 = data['pc0'][:,:3]
            pose0 = npcal_pose0to1(data['pose0'], ego_pose_norm)
            timestamp = data['timestamp']
            key = str(timestamp)

            with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
                if not exist_dict["nnd"]:
                    nnd_labels = np.zeros(pc0.shape[0], dtype= np.uint8)
                    data_t1 = dataset[data_id-1] if data_id == bounds["max_index"] else dataset[data_id+1]
                    ego_motion = npcal_pose0to1(pose0, npcal_pose0to1(data_t1['pose0'], ego_pose_norm))
                    transform_pc0 = pc0 @ ego_motion[:3,:3].T + ego_motion[:3,3]
                    nnd_labels = cuda_nnd(torch.tensor(transform_pc0).cuda(), torch.tensor(data_t1['pc0'][:,:3]).cuda(), moving_threshold=min_nnd)
                    check_data_key(f[key], "nnd", scene_id, timestamp)
                    f[key].create_dataset("nnd", data=np.array(nnd_labels).astype(np.uint8))

# No Need GPU, CPU-only for following process.
def main(
    data_dir: str ="/home/kin/data/av2/preprocess/sensor/train",
    scene_range: list = [0, 1],
    interval: int = 1, # interval frames to run dufomap only
    overwrite: bool = True,

    # NOTE (Qingwen): following is for ground segmentation only... at least for code now.
    tag: str = "av2", # [zod, nuscenes, truckscenes, sca] etc
    run_gm: bool = False, # run ground segmentation
    min_nnd: float = 0.14, # min nnd distance 1.4m/s pedestrain speed; For Scania data, we set 0.32 here.
):
    """为每个场景生成 DUFOMap 动态标签、地面掩码和 HDBSCAN 聚类标签。

    默认行为（直接运行 ``python process.py`` 或只改 ``data_dir`` /
    ``scene_range``）：

    - ``overwrite=True``：即使 HDF5 里已有 ``dufo`` 标签，也会重新生成并覆盖。
    - ``run_gm=False``：如果 HDF5 里已有 ``ground_mask``，就直接复用，不重新跑
      地面分割；只有缺失时才会临时初始化 ``ground_seg``。
    - ``scene_range=[0, 1]``：只处理第 0 个场景（左闭右开）。
    - ``interval=1``：每一帧都参与 DUFOMap 建图，不跳帧。
    - ``cluster`` 标签只有在缺失时才会生成，**不受 ``overwrite`` 控制**。

    处理完 ``main`` 后，脚本末尾会继续调用 ``run_dufocluster`` 生成
    ``dufocluster`` 实例聚类标签。

    Args:
        data_dir: 存放预处理 ``.h5`` 文件的目录。
        scene_range: 处理的场景索引范围 ``[start, end)``；``[-1, -1]`` 表示全部场景。
        interval: DUFOMap 建图时的跳帧间隔，``1`` 表示不跳帧。
        overwrite: 是否强制重新生成 ``dufo`` 标签并覆盖已有结果。
        tag: 地面分割配置文件名，会去 ``conf/ground/{tag}.toml`` 读取配置，
            仅在 ``run_gm=True`` 时生效。
        run_gm: 是否强制重新运行地面分割。默认 ``False``，即优先使用 HDF5 中
            已有的 ``ground_mask``。
        min_nnd: NND 动态标签阈值（仅在 ``run_nnd`` 中使用，``main`` 中未使用）。
    """
    gm_config_path = f"{BASE_DIR}/conf/ground/{tag}.toml"
    if not os.path.exists(gm_config_path) and run_gm:
        raise FileNotFoundError(f"Ground segmentation config file not found: {gm_config_path}. Please check folder")
    
    data_path = Path(data_dir)
    dataset = HDF5Data(data_path) # single frame reading.
    all_scene_ids = list(dataset.scene_id_bounds.keys())
    for scene_in_data_index, scene_id in enumerate(all_scene_ids): # 遍历所有场景
        start_time = time.time()
        # NOTE (Qingwen): so the scene id range is [start, end)
        if scene_range[0]!= -1 and scene_range[-1]!= -1 and (scene_in_data_index < scene_range[0] or scene_in_data_index >= scene_range[1]):
            continue
        bounds = dataset.scene_id_bounds[scene_id]
        # 检查当前场景的 h5 中是否已经存在三类标签
        # exist_dict 初始假设全部存在，一旦某帧缺失某个 key 就置为 False
        exist_dict = {"dufo": True, "ground_mask": True, "cluster": True}
        with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r') as f:
            for ii in range(bounds["min_index"], bounds["max_index"]+1):
                key = str(dataset[ii]['timestamp'])
                for datakey in exist_dict.keys():
                    if datakey not in f[key]:
                        exist_dict[datakey] = False
                if not all(exist_dict.values()):
                    break

        # 如果三类标签全都有，且 overwrite=False，则跳过本场景
        if all(exist_dict.values()) and not overwrite:
            print(f"==> Scene {scene_id} already processed, skip.")
            continue

        # 地面分割：只有缺失 ground_mask 或用户强制 run_gm=True 时才实例化 ground_seg
        if not exist_dict["ground_mask"] or run_gm:
            mygroundseg = ground_seg(gm_config_path)
        elif not exist_dict["ground_mask"] and not run_gm: # 这条分支实际上不可达
            raise ValueError("You set run_gm=False, but ground segmentation is not done. Please check the code.")
        # run_gm=True 时强制把 exist_dict["ground_mask"] 置为 False，使其重新写入
        exist_dict["ground_mask"] = exist_dict["ground_mask"] and not run_gm

        # overwrite=True 时强制重新生成 dufo 标签（注意：cluster 不受 overwrite 控制）
        if overwrite:
            exist_dict["dufo"] = False

        # 只有缺失 cluster 标签时才初始化 HDBSCAN
        if 'cluster' in exist_dict and not exist_dict["cluster"]:
            hdbscan_cluster = HDBSCAN(min_cluster_size=20, cluster_selection_epsilon=0.7, alpha=1.1)
        
        # for each scene, we normalize the pose to the first frame to avoid large values.
        ego_pose_norm = dataset[bounds["min_index"]]['pose0']
        # 如果没有dufo标签，那么开始生成
        if not exist_dict["dufo"]:
            mydufo = dufomap(0.15, 0.2, 1, num_threads=12) # resolution, d_s, d_p, hit_extension
            mydufo.setCluster(0, 20, 0.2) # depth=0, min_points=20, max_dist=0.2

            print(f"==> Scene {scene_id} start, data path: {data_path}")
            for i in tqdm(range(bounds["min_index"], bounds["max_index"]+1), desc=f"Dufo run: {scene_in_data_index+1}/{len(all_scene_ids)}", ncols=80):
                if interval != 1 and i % interval != 0 and (i + interval//2 < bounds["max_index"] or i - interval//2 > bounds["min_index"]):
                    continue
                data = dataset[i]
                assert data['scene_id'] == scene_id, f"Check the data, scene_id {scene_id} is not consistent in {i}th data in {scene_in_data_index}th scene."
                # HARD CODED: remove points outside the range
                norm_pc0 = np.linalg.norm(data['pc0'][:, :3], axis=1)
                range_mask = (
                        (norm_pc0>MIN_AXIS_RANGE) & 
                        (norm_pc0<MAX_AXIS_RANGE)
                )
                pose0 = npcal_pose0to1(data['pose0'], ego_pose_norm) # ego_i -> ego_0
                if 'lidar_center' not in data:
                    # single lidar
                    pose_array = transform_to_array(pose0)
                    mydufo.run(data['pc0'][range_mask], pose_array, cloud_transform = True)
                else:
                    # multi-lidar
                    for lid in range(data['lidar_center'].shape[0]):
                        pose_lidar = pose0 @ np.linalg.inv(data['lidar_center'][lid])
                        lidar_mask = data['lidar_id']==lid
                        points_xyz_ego = data['pc0'][lidar_mask & range_mask][:,:3]
                        # to lidar frame
                        T_ego_to_lidar = data['lidar_center'][lid]
                        R_ego_to_lidar = T_ego_to_lidar[:3, :3]
                        t_ego_to_lidar = T_ego_to_lidar[:3, 3]
                        
                        points_xyz_lidar = points_xyz_ego @ R_ego_to_lidar.T + t_ego_to_lidar

                        # to world frame
                        T_lidar_to_ego = np.linalg.inv(T_ego_to_lidar)
                        T_lidar_to_world = pose0 @ T_lidar_to_ego
                        pose_array_lidar = transform_to_array(T_lidar_to_world)

                        pose_array = transform_to_array(pose_lidar)
                        mydufo.run(points_xyz_lidar, pose_array_lidar, cloud_transform = True)
            
            # finished integrate, start segment, needed since we have map.label inside dufo
            mydufo.oncePropagateCluster(if_cluster = True, if_propagate=True)
            # NOTE(Qingwen): Just for Qingwen to check the voxel map is correct, for dufomap outputMap if voxel_map=True, there is no need to input points.
            # mydufo.outputMap(np.zeros_like(data['pc0'][:,:3]), voxel_map=True)
            # return

        for data_id in tqdm(range(bounds["min_index"], bounds["max_index"]+1), desc=f"Auto-labels run: {scene_in_data_index+1}/{len(all_scene_ids)}", ncols=80):
            data = dataset[data_id]
            pc0 = data['pc0'][:,:3]
            pose0 = npcal_pose0to1(data['pose0'], ego_pose_norm)
            timestamp = data['timestamp']
            key = str(timestamp)
            
            gm0 = data['gm0'] if exist_dict["ground_mask"] else mygroundseg.run(pc0)
            if 'cluster' in exist_dict and not exist_dict["cluster"]:
                cluster_labels = np.zeros(pc0.shape[0], dtype=np.int16) # =0 no label
                # NOTE(Qingwen): since -1 will be assigned if no cluster. We set it to 0.
                cluster_labels[gm0==0] = (hdbscan_cluster.fit_predict(pc0[gm0==0]) + 1)
            if not exist_dict["dufo"]:
                pose_array = transform_to_array(pose0)
                dufo_label = mydufo.segment(pc0, pose_array, cloud_transform = True)

            with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
                if not exist_dict["ground_mask"]:
                    check_data_key(f[key], "ground_mask", scene_id, timestamp)
                    f[key].create_dataset("ground_mask", data=np.array(gm0).astype(np.uint8))
                if 'cluster' in exist_dict and not exist_dict["cluster"]:
                    check_data_key(f[key], "cluster", scene_id, timestamp)
                    f[key].create_dataset("cluster", data=np.array(cluster_labels).astype(np.int16))
                if not exist_dict["dufo"]:
                    dufo_labels = np.zeros(pc0.shape[0], dtype= np.uint8)
                    dufo_labels[~gm0] = dufo_label[~gm0]
                    check_data_key(f[key], "dufo", scene_id, timestamp)
                    f[key].create_dataset("dufo", data=np.array(dufo_labels).astype(np.uint8))
        print(f"==> Scene {scene_id} finished, used: {(time.time() - start_time)/60:.2f} mins")
        
if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    fire.Fire(run_dufocluster)
    print("\nAlready Finished the main labels: dufo, cluster, ground mask etc.\n")
    # fire.Fire(run_nnd)
    print(f"\nScript Time used: {(time.time() - start_time)/60:.2f} mins")