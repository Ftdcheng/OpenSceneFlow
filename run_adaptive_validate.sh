#!/bin/bash
# ============================================================================
# Adaptive Consensus Quick Validation Script
# ============================================================================
# This script runs a 3-epoch ablation on demo data to validate Direction 1
# from suggestions.md: Adaptive Temporal Consensus vs baseline TeFlow.
#
# Usage:
#   bash run_adaptive_validate.sh
# Or inside tmux:
#   tmux new-session -d -s adaptive_validate "bash /home/kin/workspace/OpenSceneFlow/run_adaptive_validate.sh"
# Monitor:
#   tmux attach -t adaptive_validate
# ============================================================================

set -e

REPO_DIR="/home/kin/workspace/OpenSceneFlow"
DATA_DIR="/home/kin/data/demo"
RESULT_DIR="${REPO_DIR}/adaptive_validate"

cd "${REPO_DIR}"
mkdir -p "${RESULT_DIR}"

LOGFILE="${RESULT_DIR}/progress.log"
SUMMARY="${RESULT_DIR}/results_summary.txt"

exec > >(tee -a "${LOGFILE}")
exec 2>&1

echo "=============================================="
echo "Validation started at: $(date)"
echo "=============================================="

# ============================================================================
# Experiment A: Baseline (teflowLoss)
# ============================================================================
echo ""
echo "=== [START] Baseline: teflowLoss ==="
echo "Timestamp: $(date)"

PL_PROGRESS_BAR_DISABLE=1 python train.py model=deltaflow loss_fn=teflowLoss num_frames=5 \
  "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
  +ssl_label=seflow_auto epochs=3 batch_size=2 \
  voxel_size="[0.15, 0.15, 0.15]" point_cloud_range="[-38.4, -38.4, -3, 38.4, 38.4, 3]" \
  train_data="${DATA_DIR}/train" val_data="${DATA_DIR}/val" \
  seed=42 wandb_mode=disabled slurm_id=baseline-teflow

BASE_DIR=$(find logs/jobs -type d -name "deltaflow-baseline-teflow-*" | sort | tail -1)
echo "Baseline output dir: ${BASE_DIR}"
echo "=== [DONE] Baseline: teflowLoss ==="

# ============================================================================
# Experiment B: Adaptive Consensus (adaptiveTeflowLoss)
# ============================================================================
echo ""
echo "=== [START] Adaptive: adaptiveTeflowLoss ==="
echo "Timestamp: $(date)"

PL_PROGRESS_BAR_DISABLE=1 python train.py model=deltaflow loss_fn=adaptiveTeflowLoss num_frames=5 \
  "+add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
  +ssl_label=seflow_auto epochs=3 batch_size=2 \
  voxel_size="[0.15, 0.15, 0.15]" point_cloud_range="[-38.4, -38.4, -3, 38.4, 38.4, 3]" \
  train_data="${DATA_DIR}/train" val_data="${DATA_DIR}/val" \
  seed=42 wandb_mode=disabled slurm_id=adaptive-teflow

ADA_DIR=$(find logs/jobs -type d -name "deltaflow-adaptive-teflow-*" | sort | tail -1)
echo "Adaptive output dir: ${ADA_DIR}"
echo "=== [DONE] Adaptive: adaptiveTeflowLoss ==="

# ============================================================================
# Extract metrics from TensorBoard logs and generate summary
# ============================================================================
echo ""
echo "=== [START] Generating summary report ==="

python << PYEOF
import os, glob
from tensorboard.backend.event_processing import event_accumulator as ea

def extract(dir_path, label):
    tb_dir = os.path.join(dir_path, "logs")
    events = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
    if not events:
        print(f"[{label}] No TensorBoard events found in {tb_dir}")
        return {}
    acc = ea.EventAccumulator(events[0])
    acc.Reload()
    tags = [
        'trainer/loss',
        'trainer/cluster_based_pc0pc1',
        'val/Dynamic/Mean',
        'val/Static/Mean',
        'val/Three-way',
        'val/EPE_FD',
        'val/EPE_FS',
        'val/EPE_BS',
        'val/IoU',
    ]
    data = {}
    for tag in tags:
        try:
            scalars = acc.Scalars(tag)
            vals = [s.value for s in scalars]
            data[tag] = vals[-1] if vals else None
        except Exception:
            data[tag] = None
    return data

base_dir = "${BASE_DIR}"
ada_dir = "${ADA_DIR}"
base = extract(base_dir, 'Baseline')
ada  = extract(ada_dir,  'Adaptive')

report_path = "${SUMMARY}"
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("Adaptive Consensus Quick Validation Report\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Baseline dir:  {base_dir}\n")
    f.write(f"Adaptive dir:  {ada_dir}\n\n")

    tags = [
        'trainer/loss',
        'trainer/cluster_based_pc0pc1',
        'val/Dynamic/Mean',
        'val/Static/Mean',
        'val/Three-way',
        'val/EPE_FD',
        'val/EPE_FS',
        'val/EPE_BS',
        'val/IoU',
    ]

    f.write(f"{'Metric':<35} {'Baseline':>15} {'Adaptive':>15} {'Diff':>10}\n")
    f.write("-" * 80 + "\n")
    for tag in tags:
        b = base.get(tag)
        a = ada.get(tag)
        if b is not None and a is not None:
            diff = a - b
            f.write(f"{tag:<35} {b:>15.6f} {a:>15.6f} {diff:>+10.6f}\n")
        elif b is not None:
            f.write(f"{tag:<35} {b:>15.6f} {'N/A':>15} {'N/A':>10}\n")
        elif a is not None:
            f.write(f"{tag:<35} {'N/A':>15} {a:>15.6f} {'N/A':>10}\n")
        else:
            f.write(f"{tag:<35} {'N/A':>15} {'N/A':>15} {'N/A':>10}\n")

    f.write("\n")
    f.write("Decision Rule (from suggestions.md):\n")
    f.write("  - If val/Dynamic/Mean improves by > 3% relative: WORTH FULL RUN.\n")
    f.write("  - If gap < 2%: likely ABANDON.\n")
    f.write("  - Watch trainer/cluster_based_pc0pc1 for stability.\n")
    f.write("=" * 80 + "\n")

print(f"Report written to: {report_path}")
PYEOF

echo "=== [DONE] Summary saved to: ${SUMMARY} ==="
echo ""
echo "Validation finished at: $(date)"
echo "=============================================="
