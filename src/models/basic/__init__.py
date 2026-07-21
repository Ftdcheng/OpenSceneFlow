import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Tuple

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        print("\nLoading... model weight from: ", ckpt_path, "\n")
        return self.load_state_dict(state_dict=state_dict, strict=False)

    
@torch.no_grad()
def cal_pose0to1(pose0: torch.Tensor, pose1: torch.Tensor):
    """
    Note(Qingwen 2023-12-05 11:09):
    Don't know why but it needed set the pose to float64 to calculate the inverse 
    otherwise it will be not expected result....
    """
    pose1_inv = torch.eye(4, dtype=torch.float64, device=pose1.device)
    pose1_inv[:3,:3] = pose1[:3,:3].T
    pose1_inv[:3,3] = (pose1[:3,:3].T * -pose1[:3,3]).sum(axis=1)
    pose_0to1 = pose1_inv @ pose0.type(torch.float64)
    return pose_0to1.type(torch.float32)

def wrap_batch_pcs(batch: Dict[str, torch.Tensor], num_frames: int = 2) -> Dict[str, torch.Tensor]:
    """Warp current and past point clouds into the next-frame coordinate system.

    For each sample in the batch, this function:
      1. Computes the ego-motion transform ``ego0 -> ego1`` (either from
         ``batch['ego_motion']`` or from ``pose0`` / ``pose1``).
      2. Warps ``pc0`` (the current frame) into the next-frame coordinate system
         to obtain ``transform_pc0``.
      3. Computes ``pose_flow = transform_pc0 - pc0``, i.e. the rigid ego-motion
         displacement field for every point in ``pc0``.
      4. When ``num_frames > 2``, also warps each history frame ``pch{i}`` into
         the next-frame coordinate system.

    Args:
        batch: A dictionary produced by the DataLoader / collate function. It
            must contain the following keys:

            - ``pc0`` (Tensor): Current-frame point cloud, shape ``(B, N0, 3)``.
            - ``pc1`` (Tensor): Next-frame point cloud, shape ``(B, N1, 3)``.
            - ``pose0`` (Tensor or list): Current-frame ego pose, shape
              ``(B, 4, 4)``. Each pose is ``ego -> city``.
            - ``pose1`` (Tensor or list): Next-frame ego pose, shape
              ``(B, 4, 4)``.
            - ``ego_motion`` (Tensor or list, optional): Pre-computed transform
              ``ego0 -> ego1``, shape ``(B, 4, 4)``. If provided, ``pose0`` /
              ``pose1`` are not used for the current-to-next transform.
            - ``pch{i}`` (Tensor, optional): i-th history-frame point cloud,
              shape ``(B, Ni, 3)``. Required when ``num_frames > 2``.
            - ``poseh{i}`` (Tensor or list, optional): i-th history-frame ego
              pose, shape ``(B, 4, 4)``. Required when ``num_frames > 2``.

            Here ``B`` is batch size, ``N0`` / ``N1`` / ``Ni`` are the numbers of
            points in the corresponding frames (usually equal after padding).

        num_frames: Total number of frames per sample. ``num_frames=2`` means
            only ``pc0`` and ``pc1``; larger values additionally include history
            frames ``pch1``, ``pch2``, ...

    Returns:
        A dictionary with the warped point clouds:

        - ``pc0s`` (Tensor): Current frame warped to next-frame coordinates,
          shape ``(B, N0, 3)``.
        - ``pc1s`` (Tensor): Next frame, unchanged, shape ``(B, N1, 3)``.
        - ``pose_flows`` (List[Tensor]): Per-sample ego-motion flow. The list
          has length ``B``; each element has shape ``(N0, 3)``.
        - ``pch{i}s`` (Tensor, optional): i-th history frame warped to
          next-frame coordinates, shape ``(B, Ni, 3)``. Present only when
          ``num_frames > 2``.

    Note:
        ``pose_flows`` is returned as a list rather than a single stacked tensor
        because downstream loss computation indexes each sample with potentially
        different valid-point subsets (``pc0_valid_point_idxes``).
    """
    batch_sizes = len(batch["pose0"])

    pose_flows = [] # ego运动
    transform_pc0s = [] # 所有ego0 -> ego1变换后的点云
    transform_pc_m_frames = [[] for _ in range(num_frames - 2)] # (pch_idx, batch_idx) -> 转换到未来帧坐标系的点云
    # print(batch)
    for batch_id in range(batch_sizes):
        selected_pc0 = batch["pc0"][batch_id] 
        with torch.no_grad():
            if 'ego_motion' in batch:
                pose_0to1 = batch['ego_motion'][batch_id].type(torch.float32)
            else:
                pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id]) 
            if num_frames > 2: 
                past_poses = []
                for i in range(1, num_frames - 1):
                    past_pose = cal_pose0to1(batch[f"poseh{i}"][batch_id], batch["pose1"][batch_id])
                    past_poses.append(past_pose)

        transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3] #t -> t+1 warping

        pose_flows.append(transform_pc0 - selected_pc0)
        transform_pc0s.append(transform_pc0)

        # 所有过去帧转到未来帧坐标系
        for i in range(1, num_frames - 1):
            selected_pc_m = batch[f"pch{i}"][batch_id]
            transform_pc_m = selected_pc_m @ past_poses[i-1][:3, :3].T + past_poses[i-1][:3, 3]
            transform_pc_m_frames[i-1].append(transform_pc_m)

    pc_m_frames = [torch.stack(transform_pc_m_frames[i], dim=0) for i in range(num_frames - 2)]

    pc0s = torch.stack(transform_pc0s, dim=0) 
    pc1s = batch["pc1"]
    pcs_dict = {
        'pc0s': pc0s,
        'pc1s': pc1s,
        'pose_flows': pose_flows
    }
    for i in range(1, num_frames - 1):
        pcs_dict[f'pch{i}s'] = pc_m_frames[i-1]
    
    return pcs_dict

class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_num_channels)
        self.nonlinearity = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.nonlinearity(batchnorm_res)

class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, lr, \
        warmup_epochs, epochs, \
        last_epoch=-1, verbose=False
    ) -> None:
        self.min_lr = min_lr
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        super(WarmupCosLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        lr = self.lr / self.warmup_epochs
        return lr

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]
