---
title: "TeFlow Baseline 研究方向与快速验证指南"
date: 2026-04-14
category: "robot/mos"
tags: ["scene-flow", "teflow", "self-supervised", "baseline", "research-ideas", "open-scene-flow"]
status: sprout
difficulty: intermediate
prerequisites: ["TeFlow 论文", "OpenSceneFlow 代码库", "DeltaFlow 结构"]
source: "与 Claude 的对话整理"
---

# TeFlow Baseline 研究方向与快速验证指南

> 基于 TeFlow 论文（CVPR 2026 Highlight）和 OpenSceneFlow 代码库，梳理适合以该库为 baseline 发表工作的研究方向，并给出快速验证方法。

## 一句话总结

TeFlow 的核心 novelty 不在 backbone，而在**多帧自监督损失策略**（时序一致性投票）。因此最有价值的工作应该围绕「多帧自监督机制」向外扩展，而非单纯换 backbone。

---

## 第一梯队：低风险、改动小、故事清晰（首推）

### 方向 1：自适应时序共识机制（Adaptive Temporal Consensus）

**问题**：TeFlow 的投票权重是硬编码的（`TIME_DECAY=0.9`、`COS_THRESH=0.7071`、`TOP_K=5`），对所有场景一视同仁。但远距离物体应该更小 K，高速物体应允许更大角度偏差，被遮挡帧应自动降权。

**思路**：把固定参数改为基于局部上下文学习的自适应权重。例如用轻量 MLP，输入「位移大小、Chamfer 距离、时序间隔、局部点密度」，输出候选置信度。

**代码切入点**：
- `src/lossfuncs/selfsupervise.py` 中的 `multi_frames_clusterLoss`（L123）
- 替换 Eq. 5-7 的固定权重计算为可学习模块

**优势**：不改 backbone，训练成本不变，实验周期短。

---

### 方向 2：双向多帧监督（Bidirectional TeFlow / TeFlow++）

**问题**：`teflowLoss` 以 `pc0` 为锚点向其他帧做监督，而 `seflowppLoss` 已证明双向监督能显著提升一致性。TeFlow 只做了单向扩展。

**思路**：将时序投票机制扩展到双向/多锚点。让所有帧互为锚点，构建 fully-connected temporal graph 的一致性损失。或引入轻量 memory bank 保存Keyframe运动估计，处理消失-重现。

**代码切入点**：
- `src/trainer.py:104` 的 `ssl_loss_calculator` 目前只把 `pc0` 作为 anchor，可扩展为遍历所有帧对
- `teflowLoss` 里的 `frame_keys` 和 `get_time_delta` 逻辑基本不用大改

---

## 第二梯队：中等风险、跨模块结合、有强 novelty

### 方向 3：不确定性感知的自监督场景流（Uncertainty-Aware TeFlow）

**问题**：TeFlow 的 RANSAC 投票把 consensus winner 当成 pseudo-GT，但如果所有候选都彼此冲突（严重遮挡/形变物体），网络仍被强行拉向错误均值，且没有 uncertainty 输出。

**思路**：在 decoder 并联一个 `log_sigma` 或 `confidence` 头。用 consensus voting 的 **inlier ratio** 或 **score dispersion** 作为 uncertainty 的 pseudo-label：
- consensus 分数高 → 高置信度，损失权重加大
- consensus 分数低 → 降低损失权重，输出高 uncertainty

**代码切入点**：
- `src/models/basic/decoder.py`：在 `SparseGRUHead`（L281）输出层加 `log_var` 分支
- `src/lossfuncs/selfsupervise.py`：把 `multi_frames_clusterLoss` 的 MSE 换成 aleatoric loss 或加权 MSE

**发表定位**：RA-L / T-RO / ICRA（机器人领域非常重视 uncertainty）

---

### 方向 4：把 TeFlow 做到远距离/大规模场景（Long-Range TeFlow）

**问题**：TeFlow 官方 config 用 `[-38.4, 38.4]` 范围，0.15m voxel。而 Leaderboard v3 和 SSF 已推动长距离评估（100m+），远距离点云稀疏，两帧对应更不稳定，TeFlow 的多帧一致性反而更有价值。

**思路**：
- **Range-aware voxel size**：近处 0.15m，远处 0.3m/0.6m（hierarchical voxelization）
- 或参考 SSF 的 sparse attention 替换 MinkUNet backbone

**代码切入点**：
- `conf/model/deltaflow.yaml`：修改 `voxel_size` 和 `point_cloud_range`
- `src/models/deltaflow.py:42` 的 `SparseVoxelNet` 和 `MinkUNet` 需适配不同 grid size

**发表定位**：ICCV / NeurIPS / ICRA（蹭 Leaderboard v3 热点）

---

### 方向 5：联合场景流 + 时序实例跟踪（Scene Flow → Tracking）

**问题**：TeFlow 已经给动态点做了聚类，并在多帧间搜索最近邻对应（`batched_disid_res`）。这些对应关系本质上就是粗粒度的实例跟踪关联。

**思路**：训练时加入 instance association loss，让同一聚类在不同帧的对应点保持一致的 instance embedding。推理时同时输出 flow 和 track ID。

**代码切入点**：
- `src/lossfuncs/selfsupervise.py`：`frames_indices`（L291）已经是 `pc0` 到其他帧的最近邻索引。可在此基础上加 contrastive loss

---

## 第三梯队：高风险高回报（冲顶会级别）

### 方向 6：端到端可学习的动态聚类（End-to-End Learned Clustering）

**问题**：TeFlow / VoteFlow / SeFlow 都依赖预处理的动态聚类（HDBSCAN / DUFOMap）。聚类是 non-differentiable 的，且质量直接决定自监督上限。

**思路**：用 neural clustering 模块替代 HDBSCAN。例如用 Point Transformer 提取 per-point embedding，通过 Sinkhorn 或 contrastive clustering 做 soft cluster assignment；`multi_frames_clusterLoss` 对 soft mask 做加权平均。

**代码切入点**：
- `src/trainer.py:117` 目前从 `batch['pc0_dynamic']` 读预计算标签，需换成网络分支预测的 cluster assignment
- `src/lossfuncs/selfsupervise.py`：`torch.unique(lab0)` 需支持 soft mask

**挑战**：训练不稳定，需要精心设计 initialization 和正则化。

---

### 方向 7：测试时轻量优化 refinement（Feed-forward + TTA）

**问题**：TeFlow 已接近 optimization-based 方法性能，但局部细节可能仍不如 NSFP/FastNSF。

**思路**：用 TeFlow 输出作为 warm-start，在测试时只做 1-3 步轻量优化（只优化高 uncertainty 区域）。或更激进：测试时用当前帧多帧一致性做 online adaptation。

**代码切入点**：
- 本仓库已有 `src/models/fastnsf.py` 和 `src/models/nsfp.py`
- 可写新的 `eval_refine.py`，先跑 TeFlow，再对 flow 做 few-step Chamfer refinement

---

## 快速验证思路的「三板斧」

利用代码库高度模块化的优势，**几小时内就能判断一个想法是否值得做**。

### 1. Demo 数据 + 预训练权重做「零样本探针」

```bash
# 下载 demo 数据（1.3G）和 TeFlow 权重
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/demo-data-v2.zip
wget https://huggingface.co/kin-zhang/OpenSceneFlow/resolve/main/teflow/teflow-av2.ckpt

# 直接 eval 看 baseline 数字
python eval.py checkpoint=teflow-av2.ckpt data_mode=val \
  train_data=/path/to/demo/train val_data=/path/to/demo/val
```

**价值**：改 loss 函数前，先在 demo/val（仅 1 个 scene）上手动打印中间变量，确认逻辑正确。

### 2. 1-3 epoch 的「趋势验证」代替完整训练

TeFlow 完整 15 epoch 约 20-50 小时，但 SSL 方法的 loss 下降和 dynamic EPE 改善在**前 3 epoch 就能看出趋势**。

```bash
python train.py model=deltaflow epochs=3 batch_size=4 ... loss_fn=teflowLoss
```

**判断标准**（看 tensorboard `logs/`）：
- `trainer/loss` 是否下降？
- `trainer/cluster_based_pc0pc1` 是否更稳定？
- **3 epoch 后的 `val/Dynamic/Mean` 是否明显优于 baseline**？

如果 3 epoch 内 dynamic EPE 差距 < 2%，基本可以放弃。

### 3. 只改一个「开关」做 Ablation

最快速验证方式是**单一变量实验**：
1. 复制 `teflowLoss`，改名 `myteflowLoss`，放到 `src/lossfuncs/selfsupervise.py` 末尾；
2. 只改 `multi_frames_clusterLoss` 里的一个逻辑点；
3. 在 `src/lossfuncs/__init__.py` 里注册新 loss 名；
4. 跑 `epochs=3` 对比 `teflowLoss` vs `myteflowLoss`。

**如果 3 epoch 内有 3-5% 提升**，就值得补完整实验。

### 4. 用 2×3090 做「并行的想法筛选」

- **4×A6000**：主攻完整训练；
- **2×3090**：同时跑 2-3 个不同方向的 3-epoch 小实验（demo 数据或 av2 val 子集）。

只有被 3090 验证通过的 idea，才放到 A6000 上做完整训练。

---

## 算力-方向匹配建议

| 优先级 | 方向 | 推荐配置 | 预估训练时间 |
|--------|------|----------|--------------|
| **首推** | 方向 1（Adaptive Consensus） | 4×A6000, bs=4~6 | 17-25h |
| **次推** | 方向 3（Uncertainty-Aware） | 4×A6000, bs=4~6 | 20-25h |
| **蹭热点** | 方向 4（Long-Range） | 4×A6000, bs=2~4 | 30h+ |
| **筛 idea** | 任何方向 | 2×3090, bs=2, demo 数据 | 3-6h |

---

## 关键代码索引

| 文件 | 行号 | 作用 |
|------|------|------|
| `src/lossfuncs/selfsupervise.py` | L123-L212 | `multi_frames_clusterLoss`（TeFlow 核心） |
| `src/lossfuncs/selfsupervise.py` | L270-L307 | `teflowLoss` 入口 |
| `src/trainer.py` | L104-L147 | `ssl_loss_calculator`（组装自监督输入） |
| `src/models/deltaflow.py` | L22-L107 | `DeltaFlow` backbone |
| `src/models/basic/decoder.py` | L281-L315 | `SparseGRUHead` flow decoder |
| `conf/model/deltaflow.yaml` | - | DeltaFlow 配置文件 |

---

## 避坑指南

- **不要只换 backbone**：TeFlow 的 novelty 不在 backbone，而在多帧自监督策略。只换 backbone 不改 loss/mechanism 很难讲出新故事。
- **不要跨机混跑**：4×A6000 + 2×3090 跨机分布式会受网络带宽和显存不均衡拖累，反而更慢。建议分开使用。
- **先验证再投入**：任何想法先用 demo 数据或 3 epoch 验证，确认有提升后再跑完整实验。
