import fire
import torch

import open3d as o3d
from torch import Tensor
import numpy as np
from .color_platte import color_map, ColorPlatte, WHITE
from jaxtyping import Float, Int, Bool

def remove_ground(pc:Float[Tensor, 'N 3'], gm:Float[Tensor, 'N']):
    assert len(pc) == len(gm)
    return pc[~gm][:, :3]

def remove_far_point(pc:Float[np.ndarray, 'N 3'], threshold:float=50):
    mask = np.linalg.norm(pc[:,:3], axis=1) > threshold
    return pc[~mask][:, :3]

# for dufo label
def color_cluster_label(
        pc:Float[Tensor, 'N 3'], cluster_labels:Int[Tensor, 'N'], 
        color_map:ColorPlatte=color_map
        ) -> Float[Tensor, 'N 3']:
    unique_labels = torch.unique(cluster_labels)

    pc_all_cluster = o3d.geometry.PointCloud()
    for label in unique_labels:
        pc_cluster = o3d.geometry.PointCloud()
        pc_cluster.points = o3d.geometry.Vector3dVector(pc[cluster_labels == label]) 

        # 给对应簇标上颜色
        if label <= 0:
            pc_cluster.paint_uniform_color(WHITE)
        else:
            pc_cluster.paint_uniform_color(color_map[label])

        pc_all_cluster += pc_cluster
    
    return pc_all_cluster

def render_point_cloud():
    pass