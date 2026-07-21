"""

# Created: 2023-11-05 10:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DeFlow (https://github.com/KTH-RPL/DeFlow).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: Model Wrapper for Pytorch Lightning

"""

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from typing import Any, Dict, Optional

from lightning import LightningModule
from hydra.utils import instantiate
from omegaconf import open_dict

import os, sys, time, h5py
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils import import_func
from src.lossfuncs import SSL_LOSSES_FN
from src.utils.mics import weights_init, zip_res
from src.utils.av2_eval import write_output_file
from src.models.basic import cal_pose0to1, WarmupCosLR
from src.utils.eval_metric import OfficialMetrics, evaluate_leaderboard, evaluate_leaderboard_v2, evaluate_ssf

# debugging tools
# import faulthandler
# faulthandler.enable()

torch.set_float32_matmul_precision('medium')
class ModelWrapper(LightningModule):
    def __init__(self, cfg, eval=False):
        super().__init__()

        # 从 Hydra 配置中直接读取训练相关参数；默认值统一维护在 conf/config.yaml 中，
        # 不再在代码里硬编码，避免配置与代码中的默认值不一致。
        self.batch_size:          int                        = cfg.batch_size          # 每步训练的 batch 大小
        self.epochs:              int                        = cfg.epochs              # 训练总 epoch 数
        self.loss_fn_name:        str                        = cfg.loss_fn             # 损失函数名，如 deflowLoss / teflowLoss
        self.add_seloss:          Optional[Dict[str, float]] = cfg.add_seloss          # 自监督 loss 各项权重，None 表示不使用
        self.checkpoint:          Optional[str]              = cfg.checkpoint          # checkpoint 路径，用于打印/恢复
        self.leaderboard_version: int                        = cfg.leaderboard_version # 评估提交格式版本（1/2/3）
        self.supervised_flag:     bool                       = cfg.supervised_flag     # 是否使用数据集标签（True=有监督/半监督）
        self.save_res:            bool                       = cfg.save_res            # 是否把预测结果写回 HDF5
        self.res_name:            str                        = cfg.res_name            # 写入 HDF5 的结果数据集名称
        self.num_frames:          int                        = cfg.num_frames          # 每个样本加载的帧数（当前 + 未来 + 历史）
        self.optimizer:           Dict[str, Any]             = cfg.optimizer           # 优化器与学习率调度配置
        self.dataset_path:        Optional[str]              = cfg.dataset_path        # 数据集根目录，保存结果时使用
        self.data_mode:           str                        = cfg.data_mode           # 运行模式：train / val / valid / test
        self.cluster_loss_args:   Dict[str, Any]             = cfg.cluster_loss_args   # teflow cluster loss 的额外参数

        # 向config填充体素大小
        # 使用 cfg.model.target 里的 voxel_size / point_cloud_range 计算 grid_feature_size。
        # 训练时这两个值通过 Hydra 插值来自顶层配置；评估时 cfg.model 已从 checkpoint
        # 的 hyper_parameters 更新，因此使用 target 层级的值可保证与训练时一致。
        if 'voxel_size' in cfg.model.target and 'point_cloud_range' in cfg.model.target:
            with open_dict(cfg.model.target):
                pc_range = cfg.model.target.point_cloud_range
                voxel_size = cfg.model.target.voxel_size
                cfg.model.target['grid_feature_size'] = [
                    abs(int((pc_range[0] - pc_range[3]) / voxel_size[0])),
                    abs(int((pc_range[1] - pc_range[4]) / voxel_size[1])),
                    abs(int((pc_range[2] - pc_range[5]) / voxel_size[2])),
                ]
        
        # ---> model
        self.point_cloud_range = cfg.model.target.point_cloud_range
        self.model = instantiate(cfg.model.target)
        self.model.apply(weights_init)
        if 'pretrained_weights' in cfg and cfg.pretrained_weights is not None:
            missing_keys, unexpected_keys = self.model.load_from_checkpoint(cfg.pretrained_weights)
        # print(f"Model: {self.model.__class__.__name__}, Number of Frames: {self.num_frames}")

        # ---> loss fn
        self.loss_fn = import_func("src.lossfuncs." + self.loss_fn_name) if self.loss_fn_name is not None else None
        self.cfg_loss_name = self.loss_fn_name
        
        # ---> evaluation metric
        self.metrics = OfficialMetrics()

        # ---> inference mode
        if self.save_res and self.data_mode in ['val', 'valid', 'test']:
            self.save_res_path = Path(cfg.dataset_path).parent / "results" / cfg.output
            os.makedirs(self.save_res_path, exist_ok=True)
            print(f"We are in {cfg.data_mode}, results will be saved in: {self.save_res_path} with version: {self.leaderboard_version} format for online leaderboard.")
        if self.data_mode in ['val', 'valid', 'test']:
            print(cfg)
        # self.test_total_num = 0
        self.save_hyperparameters()
        
    def ssl_loss_calculator(self, batch, res_dict, if_log=True):
        """Build dict2loss for ALL self-supervised losses (seflow, seflowpp, teflow*).

        Each frame is represented only as a List[Tensor] and a List[labels].
        No flat tensors, no offsets, no sizes — chamfer calls use list APIs only.
        """
        total_loss, bz_ = 0.0, len(batch["pose0"])

        dict2loss = {
            'pc0_list':         [res_dict['pc0_points_lst'][i] for i in range(bz_)],
            'est_flow_list':    [res_dict['flow'][i] for i in range(bz_)],
            'pc0_labels_list':  [batch['pc0_dynamic'][i][res_dict['pc0_valid_point_idxes'][i]] for i in range(bz_)],
            'batch_size':       bz_,
        }

        frame_keys = [key.replace('_points_lst', '') for key in res_dict.keys()
                      if key.startswith('pc') and key.endswith('_points_lst')]
        frame_keys.remove('pc0')

        for frame_id in frame_keys:
            points_list = [res_dict[f'{frame_id}_points_lst'][i] for i in range(bz_)]
            labels_list = [batch[f'{frame_id}_dynamic'][i][res_dict[f'{frame_id}_valid_point_idxes'][i]] for i in range(bz_)]
            dict2loss[f'{frame_id}_list']        = points_list
            dict2loss[f'{frame_id}_labels_list'] = labels_list
                    
        loss_items, weights = zip(*[(key, weight) for key, weight in self.add_seloss.items()])
        dict2loss['loss_weights_dict'] = self.add_seloss
        
        dict2loss['cluster_loss_args'] = self.cluster_loss_args

        res_loss = self.loss_fn(dict2loss)

        for i, loss_name in enumerate(loss_items):
            if not torch.isnan(res_loss[loss_name]):
                total_loss += weights[i] * res_loss[loss_name]
        
        if if_log:
            self.log("trainer/loss", total_loss, sync_dist=True, batch_size=bz_, prog_bar=True)
        for key in res_loss:
            self.log(f"trainer/{key}", res_loss[key], sync_dist=True, batch_size=bz_)
            
        return total_loss

    def loss_calculator(self, batch, res_dict, if_log=True):
        """ Calculate the loss based on the batch (gt/ssl-label) and res_dict (estimate flow)."""
        def get_batch_data(batch, key, batch_id, batch_sizes, pc0_valid_from_pc2res, pose_flow_=None):
            """NOTE(Qingwen): for gt need double check whether it exists in the batch and batch size is correct"""
            if key not in batch or batch[key].shape[0] != batch_sizes:
                return None
            data = batch[key][batch_id][pc0_valid_from_pc2res]
            if key == 'flow' and pose_flow_ is not None:
                data = data - pose_flow_
            return data
        def get_frame_keys(data_dict, suffix):
            return [key for key in data_dict.keys() if key.endswith(suffix)]
        def extract_frame_id(key, suffix):
            """Extract frame identifier from key (e.g., 'pc0_points_lst' -> 'pc0')"""
            return key.replace(suffix, '')
        
        # Supervised-only path (deflowLoss, etc.)
        # SSL losses are handled by ssl_loss_calculator.
        total_loss, loss_logger = 0.0, {}
        loss_items, weights = ['loss'], [1.0]
        for key in loss_items:
            loss_logger[key] = 0.0

        batch_sizes, pose_flows, est_flow = len(batch["pose0"]), res_dict['pose_flow'], res_dict['flow']
        for batch_id in range(batch_sizes):
            # Get pc0 valid indices (main reference frame)
            pc0_valid_from_pc2res = res_dict['pc0_valid_point_idxes'][batch_id]
            pose_flow_ = pose_flows[batch_id][pc0_valid_from_pc2res]

            dict2loss = {'est_flow': est_flow[batch_id], 
                        'gt_flow': get_batch_data(batch, 'flow', batch_id, batch_sizes, pc0_valid_from_pc2res, pose_flow_),
                        'gt_classes': get_batch_data(batch, 'flow_category_indices', batch_id, batch_sizes, pc0_valid_from_pc2res),
                        'gt_instance': get_batch_data(batch, 'flow_instance_id', batch_id, batch_sizes, pc0_valid_from_pc2res)}
            
            # Add all available point cloud frames
            for points_key in get_frame_keys(res_dict, '_points_lst'):
                frame_id = extract_frame_id(points_key, '_points_lst')
                if points_key in res_dict:
                    dict2loss[frame_id] = res_dict[points_key][batch_id]

            res_loss = self.loss_fn(dict2loss)
 
            for i, loss_name in enumerate(loss_items):
                # if torch.isnan(res_loss[loss_name]):
                #     print(f"==> Loss: {loss_name} is nan, skip this batch.")
                #     continue
                total_loss += weights[i] * res_loss[loss_name]
            for key in res_loss:
                loss_logger[key] += res_loss[key]
        if if_log:
            self.log("trainer/loss", total_loss/batch_sizes, sync_dist=True, batch_size=self.batch_size, prog_bar=True)
        return total_loss
    
    def training_step(self, batch, batch_idx):
        total_loss = 0.0
        self.model.timer[5].start("Training Step")
        self.model.timer[5][0].start("Forward")
        res_dict = self.model(batch)
        self.model.timer[5][0].stop()
        self.model.timer[5][1].start("Compute Loss")

        if self.cfg_loss_name in SSL_LOSSES_FN:
            total_loss = self.ssl_loss_calculator(batch, res_dict)
        else:
            total_loss = self.loss_calculator(batch, res_dict)
        self.model.timer[5][1].stop()
        self.model.timer[5].stop()
        
        # NOTE (Qingwen): if you want to view the detail breakdown of time cost
        # self.model.timer.print(random_colors=False, bold=False)
        return total_loss

    def train_validation_step_(self, batch, res_dict):
        # means there are ground truth flow so we can evaluate the EPE-3 Way metric
        if batch['flow'][0].shape[0] > 0:
            pose_flows = res_dict['pose_flow'] # 进入模型会进行预处理，对点云进行坐标系变换，这里使用一下处理结果
            for batch_id, gt_flow in enumerate(batch["flow"]):
                valid_from_pc2res = res_dict['pc0_valid_point_idxes'][batch_id]
                pose_flow = pose_flows[batch_id][valid_from_pc2res]

                final_flow_ = pose_flow.clone() + res_dict['flow'][batch_id]
                v1_dict = evaluate_leaderboard(final_flow_, pose_flow, batch['pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                           batch['flow_is_valid'][batch_id][valid_from_pc2res], batch['flow_category_indices'][batch_id][valid_from_pc2res])
                v2_dict = evaluate_leaderboard_v2(final_flow_, pose_flow, batch['pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                        batch['flow_is_valid'][batch_id][valid_from_pc2res], batch['flow_category_indices'][batch_id][valid_from_pc2res])
                ssf_dict = evaluate_ssf(final_flow_, pose_flow, batch['pc0'][batch_id][valid_from_pc2res], gt_flow[valid_from_pc2res], \
                                        batch['flow_is_valid'][batch_id][valid_from_pc2res], batch['flow_category_indices'][batch_id][valid_from_pc2res])
                self.metrics.step(v1_dict, v2_dict, ssf_dict)
        else:
            pass

    def configure_optimizers(self):
        optimizers_ = {}
        # default Adam
        if self.optimizer.name == "AdamW":
            optimizers_['optimizer'] = optim.AdamW(self.model.parameters(), lr=self.optimizer.lr, weight_decay=self.optimizer.get("weight_decay", 1e-4))
        else: # if self.optimizer.name == "Adam":
            optimizers_['optimizer'] = optim.Adam(self.model.parameters(), lr=self.optimizer.lr)

        if "scheduler" in self.optimizer:
            if self.optimizer.scheduler.name == "WarmupCosLR":
                optimizers_['lr_scheduler'] = WarmupCosLR(optimizers_['optimizer'], self.optimizer.scheduler.get("min_lr", self.optimizer.lr*0.1), \
                                        self.optimizer.lr, self.optimizer.scheduler.get("warmup_epochs", 1), self.epochs)
            elif self.optimizer.scheduler.name == "StepLR":
                optimizers_['lr_scheduler'] = optim.lr_scheduler.StepLR(optimizers_['optimizer'], step_size=self.optimizer.scheduler.get("step_size", self.trainer.max_epochs//3), \
                                        gamma=self.optimizer.scheduler.get("gamma", 0.1))

        return optimizers_

    def on_train_epoch_start(self):
        self.time_start_train_epoch = time.time()

    def on_train_epoch_end(self):
        self.log("pre_epoch_cost (mins)", (time.time()-self.time_start_train_epoch)/60.0, on_step=False, on_epoch=True, sync_dist=True)
        # # NOTE (Qingwen): if you want to view the detail breakdown of time cost
        # self.model.timer.print(random_colors=False, bold=False)
    
    def on_validation_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)

        if self.data_mode == 'test':
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.checkpoint}")
            print(f"Test results saved in: {self.save_res_path}, Please run submit command and upload to online leaderboard for results.")
            if self.leaderboard_version == 1:
                print(f"\nevalai challenge 2010 phase 4018 submit --file {self.save_res_path}.zip --large --private\n")
            elif self.leaderboard_version == 2:
                print(f"\nevalai challenge 2210 phase 4396 submit --file {self.save_res_path}.zip --large --private\n")
            elif self.leaderboard_version == 3:
                print(f"""
curl -X POST https://sceneflow.argoverse.org/submissions/upload \\
  -H \"X-API-Key: your_api_key\" \\
  -F \"file=@{self.save_res_path}.zip\" \\
  -F \"method_name={self.save_res_path.name}\"
""")
            else:
                print(f"Please check the leaderboard version in the config file. We only support version 1 and 2.")
            output_file = zip_res(self.save_res_path, leaderboard_version=self.leaderboard_version, is_supervised = self.supervised_flag, output_file=self.save_res_path.as_posix() + ".zip")
            # wandb.log_artifact(output_file)
            return
        
        if self.data_mode in ['val', 'valid']:
            print(f"\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.checkpoint}")
            print(f"More details parameters and training status are in checkpoints file.")        

        self.metrics.normalize()

        # wandb log things:
        for key in self.metrics.bucketed:
            for type_ in 'Static', 'Dynamic':
                self.log(f"val/{type_}/{key}", self.metrics.bucketed[key][type_], sync_dist=True)
        for key in self.metrics.epe_3way:
            self.log(f"val/{key}", self.metrics.epe_3way[key], sync_dist=True)
        
        self.metrics.print()

        self.metrics = OfficialMetrics()

        if self.save_res:
            print(f"We already write the flow_est into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
            print(f"python tools/visualization.py --res_name \"['{self.res_name}']\" --data_dir {self.dataset_path}")
            print(f"Enjoy! ^v^ ------ \n")
        
    def eval_only_step_(self, batch, res_dict):
        """Compute final scene flow and optionally evaluate / save results.

        This function is called for val/test samples (i.e. when ground has been
        removed in ``run_model_wo_ground_data``). It reconstructs the full-frame
        scene flow as ``pose_flow + predicted_residual_flow`` and then:

        1. In ``val`` / ``valid`` mode:
           - Computes leaderboard metrics (v1, v2, ssf) on ``eval_mask`` points.
           - If ``self.save_res`` is True, writes ``final_flow`` back into the
             HDF5 file under ``{scene_id}.h5/{timestamp}/{self.res_name}``.

        2. In ``test`` mode:
           - If ``self.save_res`` is True, writes the submission file for the
             online leaderboard.

        Args:
            batch: Dict containing at least ``origin_pc0``, ``gm0``, ``pose0``,
                ``pose1``, ``eval_mask``, ``scene_id``, ``timestamp`` and, in
                val/valid mode, GT ``flow`` / ``flow_is_valid`` /
                ``flow_category_indices``.
            res_dict: Model forward outputs. May contain ``flow`` (residual flow
                on non-ground points) and ``pc0_valid_point_idxes`` (indices of
                points the model actually processed).
        """
        eval_mask = batch['eval_mask'].squeeze()
        pc0 = batch['origin_pc0']
        pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0

        # Reconstruct full-frame scene flow:
        #   final_flow = ego_motion_flow + model_predicted_residual_flow
        # Ground points keep the rigid ego-motion flow; non-ground points add
        # the network's residual prediction.
        final_flow = pose_flow.clone()
        if 'pc0_valid_point_idxes' in res_dict:
            # Model produced flow only for a subset of non-ground points.
            valid_from_pc2res = res_dict['pc0_valid_point_idxes']
            pred_flow = pose_flow[~batch['gm0']].clone()
            pred_flow[valid_from_pc2res] = res_dict['flow'] + pose_flow[~batch['gm0']][valid_from_pc2res]
            final_flow[~batch['gm0']] = pred_flow
        else:
            # Model produced flow for all non-ground points.
            final_flow[~batch['gm0']] = res_dict['flow'] + pose_flow[~batch['gm0']]

        # Val / valid mode: compute metrics and optionally save predictions to HDF5.
        if self.data_mode in ['val', 'valid']:
            gt_flow = batch["flow"]
            v1_dict = evaluate_leaderboard(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                       gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], \
                                       batch['flow_category_indices'][eval_mask])
            v2_dict = evaluate_leaderboard_v2(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                    gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], batch['flow_category_indices'][eval_mask])
            ssf_dict = evaluate_ssf(final_flow[eval_mask], pose_flow[eval_mask], pc0[eval_mask], \
                                    gt_flow[eval_mask], batch['flow_is_valid'][eval_mask], batch['flow_category_indices'][eval_mask])
            
            self.metrics.step(v1_dict, v2_dict, ssf_dict)
            # Optionally persist the full-frame prediction back to the HDF5 file.
            if self.save_res:
                key = str(batch['timestamp'])
                scene_id = batch['scene_id']
                with h5py.File(os.path.join(self.dataset_path, f'{self.data_mode}/{scene_id}.h5'), 'r+') as f:
                    if self.res_name in f[key]:
                        del f[key][self.res_name]
                    f[key].create_dataset(self.res_name, data=final_flow.cpu().detach().numpy().astype(np.float32))

        # Test mode: optionally write the leaderboard submission file.
        # batch_size is forced to 1 for val/test, so each sample is written independently.
        if self.save_res and self.data_mode == 'test':
            save_pred_flow = final_flow[eval_mask, :3].cpu().detach().numpy()
            rigid_flow = pose_flow[eval_mask, :3].cpu().detach().numpy()
            is_dynamic = np.linalg.norm(save_pred_flow - rigid_flow, axis=1, ord=2) >= 0.05
            sweep_uuid = (batch['scene_id'], batch['timestamp'])
            if self.leaderboard_version in [2, 3]:
                # Leaderboard v2/v3 expects the residual flow (without ego motion).
                save_pred_flow = (final_flow - pose_flow).cpu().detach().numpy()
            write_output_file(save_pred_flow, is_dynamic, sweep_uuid, self.save_res_path, leaderboard_version=self.leaderboard_version)

    def run_model_wo_ground_data(self, batch):
        # 除去地面
        # NOTE (Qingwen): only needed when val or test mode, since train we will go through collate_fn to remove.
        batch['origin_pc0'] = batch['pc0'].clone()
        batch['pc0'] = batch['pc0'][~batch['gm0']].unsqueeze(0)
        batch['pc1'] = batch['pc1'][~batch['gm1']].unsqueeze(0)
        
        for i in range(1, self.num_frames-1):
            batch[f'pch{i}'] = batch[f'pch{i}'][~batch[f'gmh{i}']].unsqueeze(0)

        # 前向推理
        self.model.timer[12].start("One Scan")
        res_dict = self.model(batch)
        self.model.timer[12].stop()

        # NOTE (Qingwen): Since val and test, we will force set batch_size = 1 
        batch = {key: batch[key][0] for key in batch if len(batch[key])>0}
        res_dict = {key: res_dict[key][0] for key in res_dict if (res_dict[key]!=None and len(res_dict[key])>0) }
        return batch, res_dict
    
    def validation_step(self, batch, batch_idx):
        try:
            if self.data_mode in ['val', 'valid'] or self.data_mode == 'test':
                batch, res_dict = self.run_model_wo_ground_data(batch)
                if batch['eval_flag']:
                    self.eval_only_step_(batch, res_dict)
            else:
                res_dict = self.model(batch)
                self.train_validation_step_(batch, res_dict)
        except Exception as e:
            print(f"==> Exception occur during training/validation step: {e}. Skip this batch.")
            print(f"Batch info: scene_id: {batch['scene_id']}, timestamp: {batch['timestamp']}, pc0 size: {batch['pc0']}")
    
    def test_step(self, batch, batch_idx):
        batch, res_dict = self.run_model_wo_ground_data(batch)
        pc0 = batch['origin_pc0']
        pose_0to1 = cal_pose0to1(batch["pose0"], batch["pose1"])
        transform_pc0 = pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
        pose_flow = transform_pc0 - pc0

        final_flow = pose_flow.clone()
        if 'pc0_valid_point_idxes' in res_dict:
            valid_from_pc2res = res_dict['pc0_valid_point_idxes']

            # flow in the original pc0 coordinate
            pred_flow = pose_flow[~batch['gm0']].clone()
            pred_flow[valid_from_pc2res] = pose_flow[~batch['gm0']][valid_from_pc2res] + res_dict['flow']

            final_flow[~batch['gm0']] = pred_flow
        else:
            final_flow[~batch['gm0']] = res_dict['flow'] + pose_flow[~batch['gm0']]

        # write final_flow into the dataset.
        key = str(batch['timestamp'])
        scene_id = batch['scene_id']
        with h5py.File(os.path.join(self.dataset_path, f'{scene_id}.h5'), 'r+') as f:
            if self.res_name in f[key]:
                del f[key][self.res_name]
            f[key].create_dataset(self.res_name, data=final_flow.cpu().detach().numpy().astype(np.float32))

    def on_test_epoch_end(self):
        self.model.timer.print(random_colors=False, bold=False)
        print(f"\n\nModel: {self.model.__class__.__name__}, Checkpoint from: {self.checkpoint}")
        print(f"We already write the flow_est into the dataset, please run following commend to visualize the flow. Copy and paste it to your terminal:")
        print(f"python tools/visualization.py --res_name \"['{self.res_name}']\" --data_dir {self.dataset_path}")
        print(f"Enjoy! ^v^ ------ \n")
