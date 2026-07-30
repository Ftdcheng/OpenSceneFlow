"""
最简单的单帧点云可视化脚本

运行方式（在 VNC 终端里）：
    conda activate opensf
    export DISPLAY=:1
    export LD_LIBRARY_PATH=/opt/conda/envs/opensf/lib:$LD_LIBRARY_PATH
    python visual_simple.py

按键：
    ESC / Q   退出
    鼠标左键  旋转
    鼠标右键  平移
    滚轮      缩放
"""

import os

# 保险起见，脚本里也设置一次（如果在 terminal 里已经设置过，这里只是重复）
os.environ.setdefault("DISPLAY", ":1")
os.environ["LD_LIBRARY_PATH"] = "/opt/conda/envs/opensf/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time

from src.dataset import HDF5Dataset


def main():
    # 1. 读取数据集
    data_dir = "/home/kin/data/av2/h5py/sensor/train"
    dataset = HDF5Dataset(data_dir, n_frames=2)
    print(f"数据集长度: {len(dataset)}")

    # 2. 读取第 0 帧
    data = dataset[0]
    print(f"scene_id: {data['scene_id']}, timestamp: {data['timestamp']}")
    pc0 = data["pc0"]
    print(f"pc0 shape: {pc0.shape}")

    # 3. 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc0[:, :3])

    # 4. 按高度 z 着色
    z = pc0[:, 2]
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z)
    colors = plt.get_cmap("viridis")(z_norm)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 5. 添加坐标系
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)

    # 6. 创建可视化窗口并设置点大小
    #    point_size 默认通常比较大，调小可以看清细节
    point_size = 1.0  # 可以改成 0.5, 1.0, 2.0, 3.0 试试

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="pc0", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(frame)

    # 设置点大小
    opt = vis.get_render_option()
    opt.point_size = point_size

    vis.reset_view_point(True)
    vis.update_renderer()

    print(f"正在打开 Open3D 窗口（point_size={point_size}），关闭窗口后脚本结束...")
    while vis.poll_events():
        vis.update_renderer()
        time.sleep(0.01)

    vis.destroy_window()
    print("窗口已关闭")


if __name__ == "__main__":
    main()
