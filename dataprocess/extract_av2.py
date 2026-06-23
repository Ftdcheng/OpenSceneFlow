"""
# Created: 2023-11-01 17:02
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: Preprocess Data, save as h5df format for faster loading
# Reference: 
#   * ZeroFlow data preprocessing work: https://github.com/kylevedder/argoverse2-sf
#   * Argoverse API source code: https://github.com/argoverse/av2-api
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from av2.datasets.sensor.av2_sensor_dataloader import convert_pose_dataframe_to_SE3
from av2.structures.sweep import Sweep
from av2.structures.cuboid import CuboidList, Cuboid
from av2.utils.io import read_feather
from av2.map.map_api import ArgoverseStaticMap
from av2.geometry.se3 import SE3
from av2.datasets.sensor.constants import AnnotationCategories

import multiprocessing
from pathlib import Path
from multiprocessing import Pool, current_process
from typing import Optional, Tuple, Dict, Union, Final
from tqdm import tqdm
import numpy as np
import fire, time, h5py
from collections import defaultdict
import pickle
from zipfile import ZipFile
import pandas as pd
from copy import deepcopy

import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from dataprocess.misc_data import create_reading_index, check_h5py_file_exists
from src.utils.av2_eval import read_ego_SE3_sensor

# ---------------------------------------------------------------------------
# AV2 scene flow 预处理说明
#
# AV2 官方并未直接提供每个点的 scene flow 真值，因为采集设备无法直接测量
# 每一点的 3D 运动。取而代之的，AV2 提供了一组带 track_uuid 的 3D BBOX
# （cuboid），同一物体在跨帧时拥有相同的追踪 ID。
#
# 本脚本的核心工作就是：利用相邻两帧中同一 track_uuid 对应的 cuboid 位姿
# 变化，近似估计该物体内部点的 scene flow；静态背景点则使用自车运动作为
# flow。处理结果（点云、位姿、地面掩码、flow 等）被写入 HDF5，方便后续
# 训练时快速读取。
# ---------------------------------------------------------------------------

BOUNDING_BOX_EXPANSION: Final = 0.2
CATEGORY_TO_INDEX: Final = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}

def create_eval_mask(data_mode: str, output_dir_: Path, mask_dir: str):
    """
    Need download the official mask file run: `s5cmd --no-sign-request cp "s3://argoverse/tasks/3d_scene_flow/zips/*" .`
    Check more in our assets/README.md
    """
    mask_file_path = Path(mask_dir) / f"{data_mode}-masks.zip"
    if not mask_file_path.exists():
        print(f'{mask_file_path} not found, please download the mask file for official evaluation.')
        return
    # extract the mask file
    with ZipFile(mask_file_path, 'r') as zipObj:
        zipObj.extractall(Path(mask_dir) / f"{data_mode}-masks")
    
    data_index = []
    # list scene ids
    scene_ids = os.listdir(Path(mask_dir) / f"{data_mode}-masks")
    for scene_id in tqdm(scene_ids, desc=f'Create {data_mode} eval mask', ncols=100):
        timestamps = sorted([int(file.replace('.feather', ''))
                        for file in os.listdir(Path(mask_dir) / f"{data_mode}-masks" / scene_id)
                        if file.endswith('.feather')])
        if not os.path.exists(output_dir_ / f'{scene_id}.h5'):
            continue
        with h5py.File(output_dir_ / f'{scene_id}.h5', 'r+') as f:
            for ts in timestamps:
                key = str(ts)
                if key not in f.keys():
                    print(f'{scene_id}/{key} not found')
                    continue
                group = f[key]
                mask = pd.read_feather(Path(mask_dir) / f"{data_mode}-masks" / scene_id / f"{key}.feather").to_numpy().astype(bool)
                group.create_dataset('eval_mask', data=mask)
                data_index.append([scene_id, key])

    with open(output_dir_/'index_eval.pkl', 'wb') as f:
        pickle.dump(data_index, f)
        print(f"Create reading index Successfully")

def read_pose_pc_ground(data_dir: Path, log_id: str, timestamp: int, avm: ArgoverseStaticMap):
    """读取某一时刻的位姿、点云、地面掩码，并将点云转换到 sensor 坐标系。

    AV2 的 sweep 点云默认在 ego vehicle 坐标系下。本函数为了后续处理
    （地面检测、部分 ray-casting 方法）会依次做以下转换：

    1. 读取 ``city_SE3_egovehicle.feather`` 中该时刻的位姿 ``pose``：
       表示 ego -> city 的变换。
    2. 读取 ``sensors/lidar/<timestamp>.feather`` 得到 sweep 点云 ``pc``，
       初始在 ego 坐标系下。
    3. 把 ``pc`` 通过 ``pose`` 转到 city 坐标系，调用
       ``avm.get_ground_points_boolean`` 得到地面点掩码 ``is_ground``。
    4. 利用 ``calibration/egovehicle_SE3_sensor.feather`` 中的 up_lidar 外参
       ``ego2sensor_pose``，将 ``pc`` 从 ego 坐标系转换到 sensor（up_lidar）
       坐标系。

    Args:
        data_dir: AV2 数据集根目录。
        log_id: 场景 ID。
        timestamp: 雷达帧时间戳（纳秒）。
        avm: 已加载的 ArgoverseStaticMap，用于地面检测。

    Returns:
        5 元组 ``(pc, lidar_id, lidar_dt, pose, is_ground)``：

        - ``pc`` (np.ndarray): 转换后的点云，形状 ``(N, 3)``，位于
          **sensor（up_lidar）坐标系**。
        - ``lidar_id`` (np.ndarray): 每个点来自哪个 LiDAR，``1=up_lidar``，
          ``2=down_lidar``，形状 ``(N,)``。
        - ``lidar_dt`` (np.ndarray): 每个点相对于 sweep 时间戳的时间偏移，
          单位秒，形状 ``(N,)``。
        - ``pose`` (SE3): 该时刻自车相对于城市坐标系的位姿，即 ego -> city。
        - ``is_ground`` (np.ndarray): 地面点布尔掩码，形状 ``(N,)``。
    """
    # 读取该场景所有时刻的自车位姿（ego -> city）
    log_poses_df = read_feather(data_dir / log_id / "city_SE3_egovehicle.feather")
    # 读取 up_lidar 外参（ego -> sensor），后续把点云转到 sensor 坐标系
    # 详见：https://argoverse.github.io/user-guide/datasets/lidar.html#sensor-suite
    ego2sensor_pose = read_ego_SE3_sensor((data_dir / log_id))['up_lidar']
    filtered_log_poses_df = log_poses_df[log_poses_df["timestamp_ns"].isin([timestamp])]
    # 提取当前时刻的 ego -> city 位姿
    pose = convert_pose_dataframe_to_SE3(filtered_log_poses_df.loc[filtered_log_poses_df["timestamp_ns"] == timestamp])

    # 读取当前时刻的 sweep，点云初始在 ego 坐标系
    av2_sweep = Sweep.from_feather(data_dir / log_id / "sensors" / "lidar" / f"{timestamp}.feather")
    pc = av2_sweep.xyz

    # 根据 laser_number 区分点来自 up_lidar 还是 down_lidar
    # ref: https://github.com/argoverse/av2-api/issues/77
    lidar_id = np.zeros(len(pc), dtype=np.uint8)
    lidar_id[av2_sweep.laser_number < 32] = 1
    lidar_id[av2_sweep.laser_number >= 32] = 2
    lidar_dt = av2_sweep.offset_ns / 1e9  # 转换为秒

    # 先转到 city 坐标系，再用静态地图判断地面点
    is_ground = avm.get_ground_points_boolean(pose.transform_point_cloud(pc))

    # 最终把点云转到 sensor（up_lidar）坐标系，因为部分下游方法需要 sensor 坐标
    # NOTE(SeFlow): transform to sensor coordinate, since some ray-casting based methods need sensor coordinate
    pc = ego2sensor_pose.inverse().transform_point_cloud(pc)
    return pc, lidar_id, lidar_dt, pose, is_ground

def compute_sceneflow(data_dir: Path, log_id: str, timestamps: Tuple[int, int], dclass) -> Dict[str, Union[np.ndarray, SE3]]:
    """计算两个 LiDAR sweep 之间的场景流 (scene flow)。

    AV2 官方并不直接提供每一点的 scene flow 真值（采集设备无法直接测得点级
    运动），而是提供带 ``track_uuid`` 的 3D 标注框（cuboid）。同一物体跨帧
    共享同一个追踪 ID。本函数通过匹配相邻两帧中相同 ``track_uuid`` 的 cuboid
    位姿变化，近似估计该物体内部点的 flow；静态背景点则使用自车运动作为 flow。

    读取 ``timestamps[0]`` 与 ``timestamps[1]`` 两个时刻的点云、3D 标注框
    (cuboid) 以及自车位姿，通过追踪实例在两个时刻的位姿变化，得到前景点
    从时刻 0 到时刻 1 的 flow。该函数只计算单向场景流（0 -> 1），不返回
    反向 flow。

    Args:
        data_dir: Argoverse 2.0 数据集目录，例如 ``/home/kin/data/av2/sensor/train``。
        log_id: 场景的唯一标识符。
        timestamps: 两个雷达帧的时间戳，格式为 ``(ts0, ts1)``，长度为 2。
        dclass: 用于给每个追踪实例分配连续整数实例 ID 的 ``defaultdict``。

    Returns:
        包含以下字段的字典：

        - ``pcl_0`` (np.ndarray): 时刻 0 的点云，形状 ``(N, 3)``。位于 ego0。
        - ``pcl_1`` (np.ndarray): 时刻 1 的点云，形状 ``(M, 3)``。位于 ego1。
        - ``flow_0_1`` (np.ndarray): 时刻 0 -> 1 的场景流，形状 ``(N, 3)``。
          表示点相对于时刻 0 的位移。
        - ``valid_0`` (np.ndarray): ``flow_0_1`` 的有效性掩码，形状 ``(N,)``；
          1 表示有效，0 表示无效（例如目标在下一时刻消失）。
        - ``classes_0`` (np.ndarray): 时刻 0 每个前景点的语义类别 ID，形状 ``(N,)``。
        - ``pose_0`` (SE3): 时刻 0 自车相对于城市坐标系的位姿，即 ego0 -> city。
        - ``pose_1`` (SE3): 时刻 1 自车相对于城市坐标系的位姿，即 ego1 -> city。
        - ``instances`` (np.ndarray): 时刻 0 每个点的实例 ID，形状 ``(N,)``；
          未分配实例的点为 0。
        - ``ego_motion`` (SE3): 自车从时刻 0 到时刻 1 的相对运动。
    """
    def compute_flow(sweeps, cuboids, poses):
        """根据两个时刻的 sweep、3D 标注框和自车位姿计算前景点流。

        AV2 没有直接的真值 flow，因此只能通过追踪框推导：对相邻两帧中相同
        ``track_uuid`` 的 cuboid，计算其刚体位姿变化，并将该变换作用于框内点，
        得到动态物体的 flow；未在下一帧匹配到同一 track_uuid 的物体点则标记为
        无效（``valid=0``）。

        ``sweeps``、``cuboids``、``poses`` 三个参数均只包含两个时刻的信息：
        索引 0 对应 ``timestamps[0]``，索引 1 对应 ``timestamps[1]``。

        Args:
            sweeps: 长度为 2 的 ``Sweep`` 列表，``sweeps[i].xyz`` 形状为 ``(N_i, 3)``。
                点云位于 ego 坐标系。
            cuboids: 长度为 2 的字典列表，``cuboids[i][track_uuid]`` 得到对应时刻
                的 ``Cuboid`` 标注框；缺失标注时该字典为空。
            poses: 长度为 2 的 ``SE3`` 位姿列表，``poses[i]`` 为时刻 i 城市 -> 自车
                的变换矩阵。

        Returns:
            5 元组 ``(flow, classes, valid, ego1_SE3_ego0, instances)``：

            - ``flow`` (np.ndarray): 每个前景点从时刻 0 到时刻 1 的 flow，形状
              ``(N_0, 3)``。
            - ``classes`` (np.ndarray): 每个前景点的类别 ID，形状 ``(N_0,)``。
            - ``valid`` (np.ndarray): flow 有效性掩码，形状 ``(N_0,)``；1 为有效，
              0 为无效。
            - ``ego1_SE3_ego0`` (SE3): 自车从时刻 0 到时刻 1 的相对位姿。
            - ``instances`` (np.ndarray): 每个前景点的实例 ID，形状 ``(N_0,)``；
              未分配实例的点为 0。
        """
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0]) # ego0 -> city -> city -> ego1
        # 将自车相对位姿转换为 float32
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)

        # 初始化 flow 为纯 ego motion 造成的位移（静态背景假设）
        flow = ego1_SE3_ego0.transform_point_cloud(sweeps[0].xyz) -  sweeps[0].xyz
        flow = flow.astype(np.float32)

        valid = np.ones(len(sweeps[0].xyz), dtype=np.bool_)
        classes = np.zeros(len(sweeps[0].xyz), dtype=np.uint8)
        instances = np.zeros(len(sweeps[0].xyz), dtype=np.int16)

        # # old version：使用固定 BOUNDING_BOX_EXPANSION 的包围盒膨胀
        # for id in cuboids[0]:
        #     c0 = cuboids[0][id]
        #     c0.length_m += BOUNDING_BOX_EXPANSION # the bounding boxes are a little too tight and some points are missed
        #     c0.width_m += BOUNDING_BOX_EXPANSION
        #     obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
        #     classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]

        #     if id in cuboids[1]:
        #         c1 = cuboids[1][id]
        #         c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
        #         obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
        #         flow[obj_mask] = obj_flow.astype(np.float32)
        #     else:
        #         valid[obj_mask] = 0

        # 基于目标速度的自适应包围盒膨胀（HiMo 方法）
        # 详见：https://kin-zhang.github.io/HiMo
        for id in cuboids[0]: # 第一帧拿出一个3D BBOX
            c0 = deepcopy(cuboids[0][id])
            obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)
            if id in cuboids[1]: # 第二帧有第一帧的同一物体的3D BBOX
                c1 = cuboids[1][id]
                # 在 ego 坐标系下计算同一实例在两个时刻的相对位移
                c1_SE3_c0_ego_frame = ego1_SE3_ego0.inverse().compose(c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse()))
                rel_obj_flow = c1_SE3_c0_ego_frame.transform_point_cloud(obj_pts) - obj_pts
                delta_move = abs(np.linalg.norm(rel_obj_flow, axis=0).mean())

                # 仅在目标运动时扩大包围盒，减少点云漏检
                if delta_move > 0.04:
                    c0 = cuboids[0][id]
                    # 运动越大膨胀越多，上限 2m；同时补偿 180/360 两个 LiDAR 朝向差异
                    c0.length_m += (BOUNDING_BOX_EXPANSION + min(delta_move/2, 2))
                    c0.width_m += BOUNDING_BOX_EXPANSION
                    c0.height_m += BOUNDING_BOX_EXPANSION
                obj_pts, obj_mask = c0.compute_interior_points(sweeps[0].xyz)

                # 包围盒膨胀后需要重新计算该实例内点的 flow
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
                classes[obj_mask] = CATEGORY_TO_INDEX[str(c0.category)]
                flow[obj_mask] = obj_flow.astype(np.float32)
                instances[obj_mask] = dclass[id]+1 # 在自增id的基础上偏离一位作为instance id，偏移一位是给没有标记的点腾一个0出来。
            else:
                # 下一时刻不存在同一实例，该点 flow 标记为无效
                valid[obj_mask] = 0
        return flow, classes, valid, ego1_SE3_ego0, instances
    
    # ---- 加载两个时刻的点云 ----
    sweeps = [Sweep.from_feather(data_dir / log_id / "sensors" / "lidar" / f"{ts}.feather") for ts in timestamps]

    # ---- 加载整条序列的 3D 标注，并建立时间戳索引 ----
    annotations_feather_path = data_dir / log_id / "annotations.feather"

    if not annotations_feather_path.exists():
        timestamp_cuboid_index = {}
    else:
        # annotations.feather 包含当前场景所有时刻的 3D BBOX
        cuboid_list = CuboidList.from_feather(annotations_feather_path)

        raw_data = read_feather(annotations_feather_path)
        ids = raw_data.track_uuid.to_numpy()
        # 构建 (时间戳, 追踪实例 ID) -> Cuboid 的索引，便于按时刻查找
        timestamp_cuboid_index = defaultdict(dict)
        for id, cuboid in zip(ids, cuboid_list.cuboids):
            timestamp_cuboid_index[cuboid.timestamp_ns][id] = cuboid
    # ---- 加载整条序列的 3D 标注，并建立时间戳索引 ----

    # 提取两个时刻各自的 追踪实例 ID -> Cuboid 映射
    cuboids = [timestamp_cuboid_index.get(ts, {}) for ts in timestamps]

    log_poses_df = read_feather(data_dir / log_id / "city_SE3_egovehicle.feather")

    # ---- 加载两个时刻的自车位姿（自车 -> 城市）----
    filtered_log_poses_df = log_poses_df[log_poses_df["timestamp_ns"].isin(timestamps)]
    poses = [convert_pose_dataframe_to_SE3(filtered_log_poses_df.loc[filtered_log_poses_df["timestamp_ns"] == ts]) for ts in timestamps]

    flow_0_1, classes_0, valid_0, ego_motion, instances = compute_flow(sweeps, cuboids, poses)

    return {'pcl_0': sweeps[0].xyz, 'pcl_1' :sweeps[1].xyz, 'flow_0_1': flow_0_1,
            'valid_0': valid_0, 'classes_0': classes_0, 
            'pose_0': poses[0], 'pose_1': poses[1], 'instances': instances,
            'ego_motion': ego_motion}

def process_log(data_dir: Path, log_id: str, output_dir: Path, n: Optional[int] = None):
    """处理 AV2 中的一个 log（即一个场景 scene），生成对应的 ``.h5`` 文件。

    在 AV2 术语中，一个 ``log`` 就是一段连续采集的驾驶数据，等价于一个
    ``scene``。本函数读取 ``<data_dir>/<log_id>`` 下的所有 LiDAR sweep，
    逐帧保存点云、位姿、地面掩码，并在有标注时计算相邻帧之间的 scene flow。

    由于 AV2 没有直接的真值 flow，这里的 flow 是通过匹配相邻帧中相同
    ``track_uuid`` 的 3D cuboid 位姿变化推导出来的近似值。

    最终写入 ``<output_dir>/<log_id>.h5``。

    Args:
        data_dir: AV2 数据集根目录，例如 ``/home/kin/data/av2/sensor/train``。
        log_id: 场景 ID（AV2 中称为 log）。
        output_dir: 输出 ``.h5`` 文件的目录。
        n: （当前未使用）原用于多进程 ``tqdm`` 进度条的 ``position`` 参数。
    """

    def create_group_data(group, pc, pc_id, pc_dt, gm, pose, flow_0to1=None, flow_valid=None, flow_category=None, flow_instance=None, ego_motion=None):
        """在当前时间戳对应的 h5py group 下创建各数据集。

        每个 group 代表一帧（以时间戳命名），写入的数据集结构如下：

        - ``lidar`` (float32, (N, 3)): 点云坐标，已在 sensor 坐标系下。
        - ``ground_mask`` (bool, (N,)): 每个点是否为地面点。
        - ``pose`` (float32, (4, 4)): 自车相对于城市坐标系的位姿矩阵 ``city_SE3_ego``。
        - ``lidar_id`` (uint8, (N,)): 每个点来自哪个 LiDAR（1=up_lidar, 2=down_lidar）。
        - ``lidar_dt`` (float32, (N,)): 每个点相对于 sweep 时间戳的偏移（秒）。

        当存在下一帧且有标注时，还会写入以下 ground truth flow 相关数据集：

        - ``flow`` (float32, (N, 3)): 当前帧到下一帧的场景流，定义在当前帧 ego 坐标系下。
        - ``flow_is_valid`` (bool, (N,)): 该点 flow 是否有效。
        - ``flow_category_indices`` (uint8, (N,)): 该点所属语义类别索引。
        - ``flow_instance_id`` (int16, (N,)): 该点所属实例 ID，0 表示未分配。
        - ``ego_motion`` (float32, (4, 4)): 自车从当前帧到下一帧的相对位姿矩阵 ``ego1_SE3_ego0``。
        """
        # 点云：sensor 坐标系下的 (N, 3) 坐标
        group.create_dataset('lidar', data=pc.astype(np.float32))
        # 地面点掩码：True 表示该点被判定为地面
        group.create_dataset('ground_mask', data=gm.astype(bool))
        # 自车位姿：4x4 变换矩阵，city_SE3_ego
        group.create_dataset('pose', data=pose.astype(np.float32))
        # LiDAR 编号与偏移时间：用于可视化和 HiMo 等后续处理
        group.create_dataset('lidar_id', data=pc_id.astype(np.uint8))  # 1=up, 2=down
        group.create_dataset('lidar_dt', data=pc_dt.astype(np.float32))  # 单位：秒
        if flow_0to1 is not None:
            # ground truth flow：当前帧到下一帧的 3D 位移
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
            group.create_dataset('flow_instance_id', data=flow_instance.astype(np.int16))
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))

    log_map_dirpath = data_dir / log_id / "map"
    if(len(os.listdir(log_map_dirpath))<3):
        print(f'{log_map_dirpath} needed by 3 to find the ground layer, check if you are using the correct *sensor* dataset')
        print("If you are using *lidar* dataset, Please run the following command to generate the map files:")
        print(f"python run_steps/0_additional_lidar_map.py --argo_dir {data_dir}")
        return
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    # 特定场景下所有雷达帧的时间戳
    timestamps = sorted([int(file.replace('.feather', ''))
                        for file in os.listdir(data_dir / log_id / "sensors/lidar")
                        if file.endswith('.feather')])

    
    gt_flow_flag = False if not (data_dir / log_id / "annotations.feather").exists() else True
    if check_h5py_file_exists(output_dir/f'{log_id}.h5', timestamps):
        return
    # if n is not None:
    #     iter_bar = tqdm(zip(timestamps, timestamps[1:]), leave=False,
    #                      total=len(timestamps) - 1, position=n,
    #                      desc=f'Log {log_id}')
    # else:
    #     iter_bar = zip(timestamps, timestamps[1:])
    dclass = defaultdict(lambda: len(dclass))
    with h5py.File(output_dir/f'{log_id}.h5', 'a') as f:
        for cnt, ts0 in enumerate(timestamps):
            group = f.create_group(str(ts0))
            pc0, lidar_id0, lidar_dt0, pose0, is_ground_0 = read_pose_pc_ground(data_dir, log_id, ts0, avm)
            if pc0.shape[0] < 256:
                print(f'{log_id}/{ts0} has less than 256 points, skip this scenarios. Please check the data if needed.')
                break
            if cnt == len(timestamps) - 1 or not gt_flow_flag:
                create_group_data(group, pc0, lidar_id0, lidar_dt0, is_ground_0.astype(np.bool_), pose0.transform_matrix.astype(np.float32))
            else:
                ts1 = timestamps[cnt + 1] # ts0 当前时刻， ts1 下一时刻
                scene_flow = compute_sceneflow(data_dir, log_id, (ts0, ts1), dclass)
                create_group_data(group, pc0, lidar_id0, lidar_dt0, is_ground_0.astype(np.bool_), pose0.transform_matrix.astype(np.float32),
                                  scene_flow['flow_0_1'], scene_flow['valid_0'], scene_flow['classes_0'], scene_flow['instances'],
                                  scene_flow['ego_motion'].transform_matrix.astype(np.float32))

def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)
    
def process_logs(data_dir: Path, output_dir: Path, nproc: int):
    """并行处理 AV2 数据集下的所有场景（log）。

    在 AV2 术语中，``log`` 等价于 ``scene``，即一段连续采集的驾驶序列。
    该函数列出 ``data_dir`` 下的所有 log 目录，并为每个 log 调用
    ``process_log`` 生成对应的 ``.h5`` 文件。

    Args:
        data_dir: AV2 数据集根目录，例如 ``/home/kin/data/av2/sensor/train``。
        output_dir: 输出 ``.h5`` 文件的目录。
        nproc: 并行进程数；``nproc <= 1`` 时串行执行。
    """
    
    if not data_dir.exists():
        print(f'{data_dir} not found')
        return
    
    # NOTE(Qingwen): if you don't want to all data_dir, then change here: logs = logs[:10] only 10 scene.
    logs = os.listdir(data_dir)
    args = sorted([(data_dir, log, output_dir) for log in logs])
    print(f'Using {nproc} processes to process data: {data_dir} to .h5 format. (#scenes: {len(args)})')
    # for debug
    # for x in tqdm(args):
    #     proc(x, ignore_current_process=True)
    #     break
    if nproc <= 1:
        for x in tqdm(args, ncols=120):
            proc(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(logs), ncols=120))

def main(
    argo_dir: str = "/home/kin/data/av2",
    output_dir: str ="/home/kin/data/av2/h5py",
    av2_type: str = "sensor",
    data_mode: str = "val",
    mask_dir: str = "/home/kin/data/av2/3d_scene_flow",
    nproc: int = (multiprocessing.cpu_count() - 1),
    only_index: bool = False,
):
    data_root_ = Path(argo_dir) / av2_type/ data_mode
    output_dir_ = Path(output_dir) / av2_type / data_mode
    if only_index:
        create_reading_index(output_dir_)
        return
    output_dir_.mkdir(exist_ok=True, parents=True)
    process_logs(data_root_, output_dir_, nproc)
    create_reading_index(output_dir_)
    if data_mode == "val" or data_mode == "test":
        create_eval_mask(data_mode, output_dir_, mask_dir)

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")