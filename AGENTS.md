# AGENTS.md

This file records project-level findings and recommendations for future human and AI collaborators working on OpenSceneFlow, especially around TeFlow-based research directions.

## Scope

- Repository: `/home/kin/workspace/OpenSceneFlow`
- Main context reviewed on 2026-04-16
- Sources considered:
  - `CLAUDE.md`
  - `suggestions.md`
  - repository code
  - TeFlow paper: https://arxiv.org/html/2602.19053v2

## High-Level Takeaway

TeFlow's main novelty is the multi-frame self-supervised objective, not the backbone itself. In this repo, the key TeFlow logic is concentrated in `src/lossfuncs/selfsupervise.py`, especially `multi_frames_clusterLoss()` and `teflowLoss()`. As a result, the most promising research extensions are loss/mechanism changes around temporal consensus, not simple backbone swaps.

## Important Reality Check

Some information in `CLAUDE.md` is useful, but not all of it is current.

- The code changes described for `AdaptiveConsensusMLP` and `adaptiveTeflowLoss` do exist.
- However, the branch note in `CLAUDE.md` is stale. The actual branch observed on 2026-04-16 was `main`, not `adaptive-consensus`.
- `CLAUDE.md` says a 3-epoch ablation is waiting for `adaptive_validate/results_summary.txt`, but that result file was not present in the repo at review time.
- Treat `CLAUDE.md` as partial session memory, not as a guaranteed source of truth.

## What Is Confirmed in Code

### TeFlow core lives in the loss, not the model body

- `src/lossfuncs/selfsupervise.py`
  - `batched_chamfer_related()`
  - `multi_frames_clusterLoss()`
  - `teflowLoss()`
  - `adaptiveTeflowLoss()`
- `src/trainer.py`
  - `ssl_loss_calculator()` assembles the multi-frame SSL inputs
- `src/models/deltaflow.py`
  - standard model path remains fairly generic; TeFlow-specific novelty is not here

### TeFlow already uses past and future frames

Be careful with the phrase "TeFlow is only single-directional."

- The network predicts flow for `pc0`.
- But the loss already uses multiple auxiliary frames, including past frames such as `pch1`, through `frame_keys` and `get_time_delta()`.
- So TeFlow is single-anchor in output space, but not purely single-directional in supervision.

This matters because "bidirectional TeFlow" is not a tiny extension. It is a bigger design change than `suggestions.md` sometimes implies.

## Evaluation of Existing Suggestions

### Direction 1: Adaptive Temporal Consensus

Status: reasonable, low-cost to probe, but the current implementation is probably too aggressive.

What is good:

- It targets the actual TeFlow mechanism.
- It is low-cost to implement and ablate.
- It preserves most of the existing pipeline.

What is risky:

- The current implementation replaces the original Eq. 6 handcrafted weight with a learned MLP output.
- That may discard useful inductive bias from the paper instead of refining it.
- Based on the paper's ablation logic, Eq. 5 style consensus seems more central than Eq. 6 weighting alone.

Recommendation:

- Prefer a residual or gated formulation over full replacement.
- Better pattern:
  - `w = w_eq6 * g(context)`
  - or `w = w_eq6 + delta(context)`
- Keep the original weighting signal and let the learned part modulate it.

### Direction 2: Bidirectional / Multi-Anchor TeFlow

Status: conceptually interesting, but underestimates engineering cost.

What is good:

- It fits the temporal-consistency story.
- It could improve symmetry and stability.

What is accurate (and what the original notes got wrong):

- **TeFlow already supervises with both temporal directions** — `get_time_delta` handles `pch1/pch2` natively, `ssl_loss_calculator` collects all `pc*` frames as targets, and `multi_frames_clusterLoss` automatically flips flow direction for past frames. So "TeFlow is only single-directional in supervision" is false.
- **The real gap is in output space, not supervision space** — Current TeFlow is "single-anchor output + multi-frame supervision." Direction 2 would require "multi-anchor output + multi-frame supervision" (e.g., predicting `pc0->pc1`, `pc1->pc0`, `pc0->pch1`, `pch1->pc0` simultaneously).
- **Extending from one anchor to many anchors** requires deeper changes than the original notes suggested: decoder must output multiple flow fields, `multi_frames_clusterLoss` must accept arbitrary source frames (not just `p0`), and memory/compute scale with the square of frame count.

Recommendation:

- Do not treat this as a "small next step."
- Consider only if you are ready for a broader redesign.

### Direction 3: Uncertainty-Aware TeFlow

Status: strongest candidate among the proposed directions.

Why this looks promising:

- The TeFlow consensus process already exposes useful reliability signals:
  - inlier ratio
  - score gap
  - support set size
  - score dispersion
- These are natural ingredients for uncertainty-aware supervision.
- This direction has a cleaner paper story than "small MLP on top of Eq. 6."

Best practical path:

- First do a loss-only version:
  - use consensus quality to reweight cluster loss
  - no decoder change yet
- If that looks promising, then add an uncertainty head in the decoder and move toward aleatoric modeling

### Direction 4: Long-Range TeFlow

Status: potentially impactful, but much heavier than the notes suggest.

Why:

- Current voxelization and spatial assumptions are mostly fixed-grid.
- A real long-range story likely needs more than config edits.
- Hierarchical or range-aware voxelization will touch the data path and model assumptions, not just `conf/model/deltaflow.yaml`.

Recommendation:

- Good medium/large project.
- Not the best first fast-turnaround paper path unless long-range evaluation is your main goal.

### Direction 5: Scene Flow + Tracking

Status: creative but off the repo's main path.

- There is some intuition from cluster correspondences.
- But this repo does not already provide the surrounding tracking infrastructure, losses, or evaluation path.
- This likely becomes a multi-task system paper rather than a clean TeFlow extension.

Recommendation:

- Lower priority unless tracking is already a planned target.

### Direction 6: Learned Clustering

Status: high-risk and accurately labeled as such.

- Current SSL labeling heavily relies on precomputed DUFO / clustering style labels in `src/autolabel.py`.
- Replacing this with a differentiable clustering module would be a major redesign.

Recommendation:

- Only pursue with enough time for instability and debugging.

### Direction 7: Test-Time Refinement

Status: feasible, but weakens TeFlow's feed-forward positioning.

- The repo already has optimization-based methods such as NSFP/FastNSF to borrow ideas from.
- But adding TTA/refinement dilutes the simplicity and speed story.

Recommendation:

- Do not make this the main paper direction unless the goal is specifically hybrid feed-forward + optimization.

## Recommended Priority Order

If the goal is a practical and publishable TeFlow-based extension, current recommendation is:

1. Uncertainty-aware TeFlow
2. Adaptive consensus, but in residual/gated form
3. Long-range TeFlow, if willing to take on larger system changes
4. Multi-anchor / bidirectional graph supervision

## Better Ideas Than the Current Direction-1 Implementation

These ideas are especially worth considering because they stay close to the paper's mechanism while avoiding unnecessary instability.

### 1. Consensus-Quality Weighted Robust Loss

Instead of learning candidate weights first, use consensus quality to weight the cluster loss itself.

Examples of weighting signals:

- winner score
- best-vs-second-best score gap
- inlier ratio
- inlier flow dispersion

Why this is strong:

- It is directly grounded in TeFlow's voting process.
- It is easy to implement.
- It does not require changing the network output head at the first stage.

### 2. Residual Adaptive Consensus

Do not replace Eq. 6 outright.

Use:

- `w = w_eq6 * g(context)`
or
- `w = w_eq6 + delta(context)`

Why this is better:

- keeps the paper's tested prior
- lowers risk of hurting training stability
- makes ablation cleaner

### 3. Learn Cluster-Level Hyperparameters Instead of Per-Candidate Weights

Rather than predicting a weight per candidate, predict cluster-specific settings such as:

- effective `top_k`
- cosine threshold
- time decay

Why this may be better:

- These are exactly the sensitive control knobs in the mechanism.
- This is closer to "adaptive consensus policy" than a generic MLP score.
- It may be easier to explain in a paper.

### 4. Improve Candidate Generation Instead of Only Candidate Scoring

Current external candidates mainly come from nearest-neighbor style retrieval across frames.

Possible extensions:

- reciprocal nearest neighbor filtering
- cycle consistency checks across time
- lightweight rigid-fit proposal per cluster

Why this matters:

- Better candidates may help more than a more flexible scorer on weak candidates.

## Suggested Fast Validation Plan

If compute time is limited, the best next step is not a full training run. First do two small 3-epoch probes:

### Probe A: Residual Adaptive Consensus

- Keep original Eq. 6
- Add a learned gate/modulator on top
- Compare against baseline `teflowLoss`

### Probe B: Consensus-Quality Weighted Cluster Loss

- No decoder change
- No uncertainty head yet
- Use consensus reliability to weight the cluster term

Why these two first:

- Both are close to the real TeFlow mechanism
- Both are cheaper and safer than large architectural changes
- Both are easier to interpret if they fail

## Notes for Future AI Agents

- Do not assume `CLAUDE.md` is current without checking git state and result files.
- Verify branch, outputs, and experiment artifacts locally before repeating any claim.
- Be careful with statements that TeFlow "only uses forward supervision." The current implementation already incorporates past frames in supervision.
- If proposing a new TeFlow paper direction, first ask whether it strengthens the temporal consensus mechanism itself. If not, it may be a weaker story.
- Prefer mechanism-preserving changes over wholesale replacement of the paper's priors.

## Relevant Files

- `/home/kin/workspace/OpenSceneFlow/CLAUDE.md`
- `/home/kin/workspace/OpenSceneFlow/suggestions.md`
- `/home/kin/workspace/OpenSceneFlow/src/lossfuncs/selfsupervise.py`
- `/home/kin/workspace/OpenSceneFlow/src/trainer.py`
- `/home/kin/workspace/OpenSceneFlow/src/models/deltaflow.py`
- `/home/kin/workspace/OpenSceneFlow/src/models/basic/decoder.py`
- `/home/kin/workspace/OpenSceneFlow/src/autolabel.py`
- TeFlow paper: https://arxiv.org/html/2602.19053v2
