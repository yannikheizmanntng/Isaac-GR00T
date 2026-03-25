# Interpretability Pipeline: Implementation Reference

GR00T N1.5 offline attribution — theoretical foundations, what the code does, why, and what can go wrong.

---

## Architecture in one paragraph

The pipeline runs entirely offline on a saved LeRobot dataset.  For each sample it:
1. Runs the EAGLE VLM backbone with a **gradient leaf** injected at the LLM input boundary.
2. Runs the DiT denoising loop with **explicit-softmax attention processors** that capture `[B,H,Q,K]` tensors and call `retain_grad()`.
3. Fires a single backward pass that populates gradients on all captured attention tensors **and** on the EAGLE leaf in sequence.
4. Applies two attribution methods — Chefer rollout at DiT output level, gradient×input at EAGLE LLM input level — and combines them into patch heatmaps and token bar charts.
5. Averages attribution across all captured denoising steps and renders a single GIF + PNG.

---

## Theoretical foundations

### Chefer-style relevance: why gradients are included

Chefer et al. argue that **raw attention maps are not faithful explanations** because they omit key computational components and interactions; their methods incorporate gradient signals and residual-aware propagation to better capture contribution to a prediction.

- **CVPR'21** focuses on self-attention transformers and proposes a relevance propagation formulation that accounts for **skip/residual connections** and the attention operation, maintaining relevance conservation.
- **ICCV'21** extends the approach to **bi-modal and encoder–decoder architectures**, i.e., settings with **cross-/co-attention** where query tokens attend into separate conditioning tokens (the DiT action tokens attending into VL tokens).
- Official reference implementations: `Transformer-Explainability` (CVPR'21), `Transformer-MM-Explainability` (ICCV'21).

### Scalar objective and VJP

PyTorch implements reverse-mode autograd and computes gradients of scalar-valued functions efficiently; for non-scalar outputs, `grad_outputs` specifies the "vector" in the vector–Jacobian product (VJP).

Let `y0 = y_pred[:,0,:]` (shape `[B, D]`).  The L2-squared scalar objective `s = ||y0||_2^2` has VJP vector `v = ∂s/∂y0 = 2*y0` — big-magnitude joints get bigger gradient weight.  The implementation uses `s = actions[:,0,:].sum()` instead, giving equal gradient weight `1.0` to all joint dimensions.  Both are valid for relative importance; the `.sum()` variant is simpler.

### Why `retain_grad()` is required

By default, PyTorch does **not populate `.grad`** for non-leaf tensors.  Calling `tensor.retain_grad()` enables `.grad` on that intermediate during backward.  Attention probability tensors are non-leaf (they are outputs of `softmax`), so this call is required.

### Diffusers attention processors

Diffusers' `Attention` module delegates its computation to a configurable `processor` and provides methods to set/get processors; the forward path calls `self.processor(...)`.  The default processor uses PyTorch SDPA / Flash Attention and never materialises attention weights, so explainability requires swapping in a custom processor that computes explicit softmax and calls `retain_grad()`.

---

## Pipeline overview (Mermaid)

```mermaid
flowchart TD
  A[Dataset sample: video(T,V,H,W,C) + prompt + proprio] --> B[Gr00tPolicy modality transforms]
  B --> C[Normalized input: eagle_input_ids, eagle_pixel_values, state...]
  C --> D[forward_for_attribution: hook at LLM layers[0] creates gradient leaf]
  D --> E[Install CaptureAttnProcessor on all DiT attention modules]
  E --> F[DiT denoising loop: backbone_features detached, grads through DiT only]
  F --> G[s = actions[:,0,:].sum(); s.backward() → attention_probs.grad]
  G --> H[backbone_features.backward(vl_for_dit.grad) → leaf.grad]
  H --> I[Phase 1 — Chefer rollout: A_bar per layer → r_vl [B,L]]
  H --> J[Phase 2 — Gradient×Input: ReLU(leaf.grad ⊙ leaf).sum(-1) → r_input [B,L]]
  J --> K[Token mapping: r_input → text_importance + image_heatmaps per (t,camera)]
  K --> L[Save HDF5 + render GIF/PNG]
```

---

## File map

| File | Role |
|---|---|
| `gr00t/explain/_attention_store.py` | `LayerCapture` dataclass; `AttentionStore` container |
| `gr00t/explain/_attn_processor.py` | `CaptureAttnProcessor`; `install_attn_processors`; `restore_attn_processors` |
| `gr00t/explain/_relevance.py` | `compute_abar`; `rollout_relevance`; `eagle_input_attribution`; `map_vl_tokens_to_heatmaps` |
| `gr00t/model/backbone/eagle_backbone.py` | `forward_for_attribution` (additive) |
| `gr00t/model/action_head/flow_matching_action_head.py` | `get_action_with_explain` (additive) |
| `gr00t/model/policy.py` | `explain_action` (additive) |
| `extension/interpretability/_config.py` | `InterpretabilityArgs` Pydantic model |
| `extension/interpretability/_dataset_loader.py` | LeRobot → policy-compatible obs dict; episode scoping |
| `extension/interpretability/_offline_eval.py` | main loop; HDF5 writer; visualization orchestration |
| `extension/interpretability/_visualization.py` | per-frame image composition; GIF/PNG output |
| `extension/interpretability/main.py` | CLI entry point |

---

## Phase 1: DiT Chefer attention rollout

### What it computes

For each captured denoising step `t`, for each DiT transformer block:

```
A_bar[b, q, k] = mean_over_heads( ReLU( grad(A)[b,h,q,k] * A[b,h,q,k] ) )
```

where `A` is the explicit-softmax attention probability matrix, `grad(A)` its gradient w.r.t. the scalar loss `s`.

Rollout traverses layers in reverse order:
- **Cross-attn block** (SA tokens → VL tokens): `r_vl += r_sa @ A_bar_cross`
- **Self-attn block** (SA → SA): `r_sa = r_sa @ (I + A_bar_self) / norm`

Seed: `r_sa[:, action0_idx] = 1.0` where `action0_idx = n_state + n_future`.

Output: `r_vl [B, L]` — relevance at EAGLE output positions; `r_sa [B, n_sa]` — relevance at SA positions.

### DiT architecture facts (verified from `cross_attention_dit.py`)

- `BasicTransformerBlock` has exactly **one** attention module (`attn1`), no `attn2`.
- `attn1` is dual-mode: self-attention when `encoder_hidden_states=None`, cross-attention when `encoder_hidden_states=vl_embs`.
- When `interleave_self_attention=False` (likely the deployment config): **all blocks are cross-attention** — `encoder_hidden_states=vl_embs` is passed to every block. In this case `r_sa` never gets SA-redistributed; the seed flows directly and identically to `r_vl` through each cross-attn layer.
- When `interleave_self_attention=True`: odd-indexed blocks get `encoder_hidden_states=None` (SA→SA), even-indexed get VL cross-attn. Both branches of `rollout_relevance` are active.

### Scalar objective

Implemented as:
```python
s = actions[:, 0, :].sum()
s.backward()
```

This is a **sum** over the first predicted action step, not the L2-squared norm originally planned (`s = (actions[:,0,:] * actions[:,0,:]).sum()`).  Difference: with `.sum()`, all joint dimensions get equal gradient weight `1.0`; with L2², the gradient is `2 * actions[:,0,:]` — proportional to action magnitude and sign-preserving. The `.sum()` variant is simpler and sufficient for relative importance, but it gives the same gradient to a joint predicting `+5.0` and one predicting `-0.001`.

### `CaptureAttnProcessor` vs default processor

The default processor uses PyTorch SDPA / Flash Attention and never materializes `A`. The capture processor computes explicit softmax:
```
scores = Q @ K.T * scale
probs  = softmax(scores)           # [B*H, Q, K]
probs_bhqk = probs.view(B,H,Q,K)
probs_bhqk.retain_grad()
```
then routes the output through `probs_bhqk` (not `probs`) so the autograd graph passes through the retained tensor.  **Expected action drift** ≈ 1e-4 (due to different kernel path, numeric precision).  Larger drift would indicate a bug in the Q/K/V projection calls.

---

## Phase 2: EAGLE gradient×input attribution

### Motivation

`r_vl` is relevance at the **output** of EAGLE's LLM — after all LLM transformer layers have mixed image tokens and text tokens together.  A position that dominates `r_vl` might have been driven by any combination of the raw patches and prompt tokens.  Phase 2 propagates the relevance further back to get attribution **at the LLM input level**: the sequence positions where text tokens and image tokens exist separately, before any LLM mixing.

### Mechanism

```
EAGLE forward (hook) → backbone_features   [grad graph: leaf → LLM → eagle_linear]
                ↓
vl_for_dit = backbone_features.detach().requires_grad_(True)
                ↓
DiT forward + s.backward()   →   vl_for_dit.grad   [DiT graph freed]
                ↓
backbone_features.backward(gradient=vl_for_dit.grad)   →   leaf.grad   [EAGLE graph freed]
                ↓
r_input[b,l] = ReLU( leaf.grad[b,l,:] ⊙ leaf[b,l,:] ).sum(-1)   [B, L]
r_input       ← normalised to r_input.sum() == r_vl.sum()
```

### Hook mechanics (`forward_for_attribution`)

A `register_forward_pre_hook` on `eagle_model.language_model.model.layers[0]` intercepts the `hidden_states` argument before it enters the first transformer layer.  The hook:
1. Calls `h.detach()` — cuts the graph from the vision encoder and embedding lookup.
2. Creates `leaf = detached_h.requires_grad_(True)` — establishes a new differentiable leaf.
3. Returns `(leaf,) + args[1:]` — PyTorch routes the entire LLM forward through `leaf`.

The handle is always removed in a `finally` block.

**What `leaf` represents**: the combined (text + visual) hidden states at the LLM input boundary, AFTER SigLIP features have been projected by `mlp1` and injected into the text embedding sequence.  This is the pre-mixing point: tokens exist as individuals, but individual token identities (text vs image) are still recoverable via `eagle_input_ids`.

### Attribution formula

```python
r_input = (leaf_grad * leaf).clamp(min=0).sum(dim=-1)   # [B, L]
r_input = r_input * r_vl.sum(-1, keepdim=True) / r_input.sum(-1, keepdim=True)
```

Identical structure to proprioception attribution.  Budget is preserved: `r_input.sum() == r_vl.sum()` exactly.

### What changes in the outputs

| Quantity | Before Phase 2 | After Phase 2 |
|---|---|---|
| Heatmap source | `r_vl` (post-LLM-mixing) | `r_input` (pre-LLM-mixing) |
| `vl_token_importance` in HDF5 | DiT-level `r_vl` | unchanged (kept for comparison) |
| `eagle_input_importance` in HDF5 | not present | `r_input` |
| `text_importance` in visualization | from `r_vl` | from `r_input` |
| `image_importance_sum` in visualization | from `r_vl` | from `r_input` |
| Total budget | `r_vl.sum() + state_sum` | same |
| Image+text split within budget | from post-mixing | from pre-mixing — more granular |

---

## Evaluation loop details

### Episode scoping

`DatasetLoader.episode_indices(start_global_idx=0)` returns all global step indices that share the same `trajectory_id` as step 0.  The loop is restricted to this set before applying `max_samples`.  This prevents the GIF from silently crossing an episode boundary when `max_samples > episode_length`.

### Capture step validation

`capture_steps` are validated against `policy.denoising_steps` before the loop begins.  Steps outside `[0, denoising_steps - 1]` trigger a warning and are filtered out.  If no valid steps remain, the run raises immediately rather than producing empty output.

### Step aggregation

Attribution is computed independently for each step in `capture_steps`.  Before visualization, all steps for a sample are averaged into a single "mean" entry:
- `eagle_input_importance` (heatmap source): averaged — valid because the spatial pattern is fixed by `leaf.grad`; only the normalisation scale varies per step.
- `text_importance` (Chefer-based): averaged — captures mean token relevance across the denoising trajectory.
- `state_joint_attribution`: taken from the first step — it is step-independent (derived from a single `leaf.grad`).
- `raw_frames`, `token_labels`, `task_span`: taken from the first step (constant).

One GIF frame is produced per sample (not per step).

---

## Visualization

### Layout

```
┌──────────────┬───────────────────────────────────────────────┐
│              │  token attribution bar chart (50 % height)    │
│  camera      │  ─────────────────────────────────────────── │
│  heatmaps    │  token text strip                             │
│  (~30 % w)   ├───────────────────────────────────────────── │
│              │  proprioception bars  │  modality overview    │
│              │  (50 % height)        │                       │
└──────────────┴───────────────────────────────────────────────┘
```

A 24 px white gap separates the camera panel from the chart panel.

### Token bar chart

Image token positions are **always excluded** — their relevance is captured in `image_importance_sum` and shown in the modality overview only.

When `task_span` is detected (see below), three display segments are rendered:
- `<sys>` — one bar, sum of all non-image tokens before the task description
- one bar per task-description token
- `<end>` — one bar, sum of all non-image tokens after the task description

This removes the large runs of system-prompt / template special tokens from the detailed view.

The y-axis is scaled to the **individual task-token** bar heights only, so the tall aggregate `<sys>` bar does not crush the task tokens into invisibility.

### Task span detection (`_find_task_span`)

Extracts the task description from `obs["annotation.human.task_description"]`, then decodes each non-image token in `eagle_input_ids` to text, concatenates them, and finds the task description as a plain substring.  Returns `(start, end)` in full `ids_np` coordinates.

This string-based approach is robust to subword prefix characters (`▁`, `Ġ`) and leading-space tokenization differences that make re-tokenization and exact token-ID matching fragile.

Falls back to `None` (all non-image tokens shown individually) with a `WARNING` log if the search fails.

### Heatmap

- **Shared colour scale**: global `min`/`max` computed across all camera views at the display timestep before normalisation.  The same colour value means the same attribution magnitude in both cameras.
- **Colormap**: dark maroon `#3A0010` → deep crimson `#A00025` → vivid red `#E03050` → pastel pink `#FF8899` → cream `#FFE8EC`.  Low-attention areas are near-black; high-attention areas push through saturated red into pale cream.
- **Alpha blending**: `overlay = base * (1 − α·0.85) + colored * α·0.85` where `α = heat^0.6`.  A 15 % white veil is applied to the base frame first to lift dark regions.  Zero attention → original image unchanged.  High attention → strong colour overlay.

### Colour palette

All chart colours match the distribution card (`scripts/gen_distribution_card.py`):

| Element | Colour | Source |
|---|---|---|
| Token bars (text) | `#7799D3` | `_PAL["blue"][1]` |
| Proprioception bars | `#AFCA9E` | `_PAL["green"][0]` |
| Modality — image | `#D9585C` | `_PAL["red"][1]` (board rotation chart) |
| Modality — prompt | `#7799D3` | `_PAL["blue"][1]` |
| Modality — proprio | `#AFCA9E` | `_PAL["green"][0]` |
| `<sys>` / `<end>` bars | `#C8CDDA` | neutral grey |

### Joint labels

`_state_joint_labels` maps `state.robot_arm` (6 dims) to `["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]` via `_UR5_JOINT_NAMES`.  Other state keys fall through to `name_i` generic labels.

### Output files

| File | Contents |
|---|---|
| `results.h5` | Full attribution data for all samples and steps |
| `attribution.gif` | 30 fps GIF, one frame per sample (mean across steps) |
| `attribution_preview.png` | First frame of the GIF saved as a static PNG for quick review |

---

## HDF5 output schema (per sample, per step)

```
{sample_idx}/
  action_pred              float32  [H, D]
  action_gt                float32  [H, D]
  eagle_input_ids          int32    [L]
  meta/
    T, V, capture_steps, view_names
  step_{t}/
    vl_token_importance    float32  [L]    ← DiT-level (after EAGLE mixing)
    eagle_input_importance float32  [L]    ← pre-mixing (used for heatmaps)
    state_token_importance float32  [n_state]
    state_joint_attribution float32 [n_joints]
    image_heatmaps         float32  [T, V, patch_h, patch_w]
```

---

## Assumptions (final, explicit)

**A1. `attn1` is the only attention module per block.**
Verified from `cross_attention_dit.py` line 124: `BasicTransformerBlock` instantiates only `self.attn1`, no `attn2`.  `CaptureAttnProcessor` is installed on `attn1` only — correct and complete.

**A2. `interleave_self_attention` determines self/cross split.**
With `False` (default): all blocks receive `encoder_hidden_states=vl_embs` → all are cross-attn.  The rollout's SA branch is never exercised; `r_vl` accumulates from a fixed `r_sa` seed at each layer.  With `True`: alternating.  Both are handled.

**A3. LLM transformer layers form the only graph path from `leaf` to `backbone_features`.**
Verified: the hook intercepts between the embedding/projection step and `layers[0]`.  All subsequent computation (all `select_layer` LLM transformer layers + `eagle_linear`) uses `leaf` as its initial hidden state.

**A4. Flash Attention backward is differentiable through.**
Flash Attention (both `flash_attn` package and PyTorch's `F.scaled_dot_product_attention`) supports backward.  `leaf.grad` should be populated.  Verified by: `assert eagle_input_leaf.grad is not None` post-backward.

**A5. `vl_for_dit.grad` is populated by the DiT backward.**
`vl_for_dit` is a leaf with `requires_grad=True`.  It flows through `process_backbone_output` (vlln + vl_self_attention) inside `get_action_with_explain`, then through all DiT cross-attn layers.  `s.backward()` propagates to `vl_for_dit.grad` ✓.

**A6. LLM parameters have `requires_grad=False` during inference.**
Set by `set_trainable_parameters(tune_llm=False, tune_visual=False)` called from `__init__`.  EAGLE params do NOT accumulate `.grad` — only `leaf` does.  No optimizer-relevant side-effects.

**A7. Token ordering: image tokens in EAGLE input follow `(T, V)` = `(time, camera)` order.**
Set by `GR00TTransform`: `rearrange(images, "v t c h w -> (t v) c h w")`.  `map_vl_tokens_to_heatmaps` uses this ordering implicitly via `img_pos.view(T, V, num_image_tokens)`.

**A8. `num_image_tokens` is a perfect square.**
Required by `int(num_image_tokens ** 0.5)` in `map_vl_tokens_to_heatmaps`.  Asserted at runtime.

**A9. Task description appears verbatim in the tokenized sequence.**
`_find_task_span` searches by substring matching on decoded token text.  This holds unless `formalize_language=True` is set in `GR00TTransform` (lowercasing / punctuation removal).  A case-insensitive fallback is attempted.  If the search still fails, the visualization falls back to showing all non-image tokens individually with a printed warning.

---

## Critical risks and mitigations

### R1. `leaf.grad is None` after backward
**When**: if any operation between `leaf` and `backbone_features` stops gradient flow (e.g., an in-place operation or a `.detach()` inside the LLM).
**Impact**: silent fallback — `eagle_input_grad_np = None`, `r_input = r_vl` (DiT-level, no Phase 2 benefit).
**Mitigation**: the fallback is explicit in `_offline_eval.py`.  Log `eagle_input_grad is None` warnings.  First-run check: `assert result["eagle_input_grad"] is not None`.

### R2. Hook fires more than once
**When**: gradient checkpointing re-runs the forward, causing `layers[0]` to execute twice.  The second call would overwrite `leaf_holder["leaf"]` with a new leaf that is NOT connected to the first-pass backward graph.
**Impact**: `leaf.grad` is from the wrong pass; Phase 2 attribution is wrong.
**Mitigation**: gradient checkpointing is disabled for inference (`self.gradient_checkpointing = False` in DiT).  No known path for this in inference mode.

### R3. `backbone_features.backward(vl_for_dit.grad)` with wrong dtype
**When**: dtype mismatch between `vl_for_dit.grad` and the expected gradient type.
**Impact**: `leaf.grad` wrong or None.
**Mitigation**: `vl_for_dit` is created by `backbone_output["backbone_features"].detach()`, so it inherits the same dtype.  The gradient will also be bfloat16.  No mismatch.

### R4. `eagle_input_attribution` normalization blow-up
**When**: `ReLU(grad⊙leaf)` is exactly zero everywhere.
**Impact**: `r_input` is all-zero; heatmaps go blank.
**Mitigation**: `.clamp(min=1e-8)` on both `budget` and `scale`.  Uniform-zero result is visually obvious (blank heatmap).

### R5. `interleave_self_attention=False` → rollout seed never redistributed
**When**: all blocks are cross-attn.  `r_sa[:, action0_idx] = 1.0` is never modified.
**Impact**: attribution is the sum of cross-attn maps weighted by a fixed initial `r_sa`.  Valid first-order attribution but misses SA-level information flow.
**Mitigation**: acceptable for the current VL attribution use case.

### R6. `process_backbone_output` Jacobian in the attribution path
**Detail**: `get_action_with_explain` applies `vlln` (LayerNorm) + `vl_self_attention` to `vl_for_dit` before passing it to the DiT.  `vl_for_dit.grad` therefore includes the Jacobian of these operations.  The EAGLE backward then propagates this compound gradient.  **Correct by chain rule** — no bug, but the attribution path includes more than the raw VL→DiT interface.

### R7. `encoder_attention_mask` not passed to DiT in explain path
**Detail**: consistent with `get_action` (inference).  For single-sample inference (no padding) this is harmless.

---

## Verification checklist

```python
# 1. Leaf receives gradient
result = policy.explain_action(obs, capture_steps=[0])
assert result["eagle_input_grad"] is not None, "EAGLE backward failed"

# 2. Shapes consistent
assert result["eagle_input_leaf"].shape == result["eagle_input_grad"].shape
assert result["eagle_input_leaf"].ndim == 3   # [B, L, D_llm]

# 3. Budget preserved
import torch, numpy as np
leaf = torch.from_numpy(result["eagle_input_leaf"])
grad = torch.from_numpy(result["eagle_input_grad"])
r_input = (grad * leaf).clamp(min=0).sum(-1)
r_vl_sum = float(r_vl.sum())
r_input_norm_sum = float(r_input / r_input.sum() * r_vl_sum).sum()
assert abs(r_input_norm_sum - r_vl_sum) < 1e-3

# 4. Action drift small (softmax vs SDPA kernel difference)
baseline = policy.get_action(obs)["action"]
explain  = policy.explain_action(obs, [0])["action_pred"]
assert np.abs(baseline - explain).max() < 0.05

# 5. Heatmap shape
assert heatmaps.shape == (B, T_vid, V, patch_h, patch_w)

# 6. Task span detection
from extension.interpretability._offline_eval import _find_task_span
span = _find_task_span(ids_np, obs, tokenizer, image_token_index)
assert span is not None, "task span detection failed — check annotation key"

# 7. interleave_self_attention flag logged
print(policy.model.action_head.config.diffusion_model_cfg.get("interleave_self_attention"))

# 8. Faithfulness: prompt masking
# Replace prompt with neutral instruction; token importance on prompt tokens should drop.
# Camera ablation: zero out one camera's frames; heatmap mass should shift away from that camera.
# Token deletion: masking top-k important tokens should change prediction more than random deletion.
```

---

## What the attributions mean (precisely)

| Signal | What it measures | What it does NOT measure |
|---|---|---|
| `r_vl[l]` | How much EAGLE output position `l` (after full LLM mixing) drove the action | Which raw patch or token caused it |
| `r_input[l]` | How much the LLM input embedding at position `l` drove the action, flowing through all `select_layer` LLM layers | Sub-token pixel attribution; SigLIP-internal processing |
| `image_heatmaps[t,v,i,j]` | Attribution of SigLIP patch `(i,j)` for camera `v` at timestep `t`, at the LLM input level | Whether the patch was attended to by EAGLE's vision encoder internally |
| `state_joint_attribution[k]` | Gradient×input of joint `k` proprioception on the final action prediction | Joint–joint interaction effects |

The image heatmaps attribute to **already-projected SigLIP patch features** — the output of EAGLE's `mlp1` projector for each spatial patch.  Spatial structure is fully preserved (each leaf position maps to exactly one `(t, v, i, j)` patch).  The attribution does NOT decompose into raw pixel contributions.
