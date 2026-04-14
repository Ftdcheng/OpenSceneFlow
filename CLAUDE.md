# CLAUDE.md — OpenSceneFlow Session Context

## Last Updated
2026-04-14

---

## Active Research Direction

**Direction 1: Adaptive Temporal Consensus (from `suggestions.md`)**

Transform TeFlow's hardcoded multi-frame voting weights (fixed `TIME_DECAY=0.9`, `COS_THRESH=0.7071`, `NET_EST_W=1.0`) into a **learnable lightweight MLP** that predicts per-candidate confidence based on local context:
- flow magnitude
- Chamfer distance
- temporal factor
- cluster point density

---

## Git Branch

`adaptive-consensus` (branched from `main`)

---

## Code Changes Made

### 1. `src/lossfuncs/selfsupervise.py`
- **Added `AdaptiveConsensusMLP`** — 2-layer MLP (4 → 32 → 32 → 1, Sigmoid). Takes 4 normalized features per candidate and outputs a confidence score.
- **Modified `multi_frames_clusterLoss`** — added optional `adaptive_mlp` parameter.
  - When `adaptive_mlp` is provided, **Eq. 6 hardcoded weights** are replaced by the MLP output.
  - The rest of the RANSAC pipeline (cosine inlier voting, Eq. 5/7/8) remains unchanged.
- **Added `adaptiveTeflowLoss`** — identical structure to `teflowLoss`, but passes `adaptive_mlp` into `multi_frames_clusterLoss`.

### 2. `src/trainer.py`
- Imported `AdaptiveConsensusMLP`.
- In `ModelWrapper.__init__`, when `loss_fn == 'adaptiveTeflowLoss'`, auto-instantiates `AdaptiveConsensusMLP` and registers it in `cluster_loss_args['adaptive_mlp']`.
- **Hotfix**: changed exception logging in `validation_step` from `batch['timestamp']` to `batch.get('timestamp', 'N/A')` to prevent `KeyError` on demo data.

### 3. `run_adaptive_validate.sh`
- Single-script 3-epoch ablation on demo data (`/home/kin/data/demo`).
- Runs **Baseline** (`teflowLoss`) then **Adaptive** (`adaptiveTeflowLoss`), both with identical seeds and hyperparameters.
- Uses correct TeFlow geometry: `voxel_size=[0.15,0.15,0.15]`, `point_cloud_range=[-38.4,-38.4,-3,38.4,38.4,3]`.
- Disables progress bars (`PL_PROGRESS_BAR_DISABLE=1`) to keep log files small.
- Auto-extracts TensorBoard metrics and writes a summary report to `adaptive_validate/results_summary.txt`.

---

## How to Train / Validate

### Quick 3-epoch ablation (demo data)
```bash
cd /home/kin/workspace/OpenSceneFlow
bash run_adaptive_validate.sh
```

Or inside tmux:
```bash
tmux new-session -d -s adaptive_validate "bash run_adaptive_validate.sh"
tmux attach -t adaptive_validate
```

### Full training (adaptive consensus)
```bash
python train.py model=deltaflow loss_fn=adaptiveTeflowLoss num_frames=5 \
  "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
  +ssl_label=seflow_auto epochs=15 batch_size=2 \
  voxel_size="[0.15, 0.15, 0.15]" point_cloud_range="[-38.4, -38.4, -3, 38.4, 38.4, 3]" \
  train_data=/path/to/train val_data=/path/to/val \
  wandb_mode=disabled
```

### Baseline checkpoint eval (already verified)
Baseline `teflow-av2.ckpt` was evaluated on demo val and produced expected metrics:
- Three-way: ~5.48 cm (demo, higher than paper test-set 3.57 cm due to small sample size)
- Dynamic/Mean (normalized): ~0.303

---

## Current Status

- ✅ Branch `adaptive-consensus` created
- ✅ Code changes implemented and import-tested
- ✅ Validation script written and debugged
- ⏳ Waiting for 3-epoch ablation results on `adaptive_validate/results_summary.txt`

---

## Decision Criteria (from `suggestions.md`)

After reading `adaptive_validate/results_summary.txt`:
- If `val/Dynamic/Mean` improves by **> 3% relative** vs baseline → **Worth full A6000 run**.
- If gap **< 2%** → **Likely abandon** Direction 1 and try Direction 3 (Uncertainty-Aware) or Direction 2 (Bidirectional).
