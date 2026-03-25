from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from gr00t.explain._relevance import eagle_input_attribution, map_vl_tokens_to_heatmaps, rollout_relevance
from gr00t.model.policy import Gr00tPolicy

from extension.interpretability._config import InterpretabilityArgs
from extension.interpretability._dataset_loader import DatasetLoader
from extension.interpretability._visualization import build_summary_frame, save_summary_gif


def _build_policy(cfg: InterpretabilityArgs) -> Gr00tPolicy:
    import importlib
    import json
    from pathlib import Path as P

    args_path = P(cfg.model_path) / "args.json"
    with open(args_path) as f:
        model_args = json.load(f)

    dataset_args = model_args.get("dataset_args", {})
    data_config_name = dataset_args.get("data_config", "UR5_Abs_Delta_4_Cfg")
    configs_path = dataset_args.get("data_configs_path", "extension/dataset/_config.py")
    embodiment_tag = dataset_args.get("embodiment_tag", cfg.embodiment_tag)

    module_name = configs_path.replace("/", ".").removesuffix(".py")
    mod = importlib.import_module(module_name)
    data_config = getattr(mod, data_config_name)()

    policy = Gr00tPolicy(
        model_path=cfg.model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=embodiment_tag,
        device=cfg.device,
    )
    return policy


def run_offline_eval(cfg: InterpretabilityArgs) -> None:
    out_root = Path(cfg.output_path)
    out_root.mkdir(parents=True, exist_ok=True)
    hdf5_path = out_root / "results.h5"

    policy = _build_policy(cfg)
    loader = DatasetLoader(cfg, policy)

    # Restrict to one episode: all steps sharing the same trajectory as step 0.
    # This prevents the GIF from silently crossing an episode boundary when max_samples
    # exceeds the length of the first episode.
    indices = loader.episode_indices(start_global_idx=0)
    if cfg.max_samples is not None:
        indices = indices[: cfg.max_samples]

    backbone = policy.model.backbone
    image_token_index = backbone.eagle_model.image_token_index
    num_image_tokens = _get_num_image_tokens(backbone)
    T_vid = len(policy.video_delta_indices)
    V = len(loader.video_keys)
    tokenizer = _get_tokenizer(policy)
    state_joint_labels = _state_joint_labels(loader._data_config)
    n_raw_state_dims = len(state_joint_labels)

    # Validate capture_steps against the model's actual denoising step count.
    n_denoise = policy.denoising_steps
    invalid = [s for s in cfg.capture_steps if s < 0 or s >= n_denoise]
    if invalid:
        print(
            f"WARNING: capture_steps {invalid} are out of range [0, {n_denoise - 1}] "
            f"(num_inference_timesteps={n_denoise}). These steps will produce no captures."
        )
    capture_steps_valid = [s for s in cfg.capture_steps if 0 <= s < n_denoise]
    if not capture_steps_valid:
        raise ValueError(
            f"No valid capture_steps remain after filtering against "
            f"num_inference_timesteps={n_denoise}. Got: {cfg.capture_steps}"
        )

    print(f"Running interpretability on {len(indices)} samples → {hdf5_path}")
    print(f"  capture_steps={capture_steps_valid}  (denoising_steps={n_denoise})")

    vis_entries: list[tuple] = []

    with h5py.File(hdf5_path, "w") as hf:
        for sample_idx in indices:
            obs, gt_action = loader.load(sample_idx)
            result = policy.explain_action(obs, capture_steps=capture_steps_valid)

            action_pred_real = _unnormalize_action(policy, result["action_pred"])
            backbone_inputs = result["backbone_inputs"]
            eagle_input_ids = backbone_inputs["eagle_input_ids"].cpu()
            n_vl = eagle_input_ids.shape[-1]

            grp = hf.require_group(str(sample_idx))
            grp.create_dataset("action_pred", data=action_pred_real.astype(np.float32))
            grp.create_dataset("action_gt", data=gt_action.astype(np.float32))
            grp.create_dataset(
                "eagle_input_ids",
                data=eagle_input_ids.squeeze(0).numpy().astype(np.int32),
            )
            meta = grp.require_group("meta")
            meta.attrs["T"] = T_vid
            meta.attrs["V"] = V
            meta.attrs["capture_steps"] = capture_steps_valid
            meta.attrs["view_names"] = loader.video_keys

            eagle_leaf = result.get("eagle_input_leaf")
            eagle_grad = result.get("eagle_input_grad")

            # Vis entries produced for this sample across all steps — used for mean aggregation.
            sample_vis_entries: list[tuple] = []

            for step in capture_steps_valid:
                captures = result["store"].captures_for_step(step)
                if not captures:
                    continue

                r_sa, r_vl = rollout_relevance(
                    captures,
                    n_sa=result["n_sa"],
                    n_vl=n_vl,
                    action0_idx=result["action0_idx"],
                )

                # Propagate relevance back through EAGLE LLM to pre-mixing patch/token level.
                # Falls back to DiT-level r_vl when the EAGLE backward pass was unavailable.
                if eagle_leaf is not None and eagle_grad is not None:
                    r_input = eagle_input_attribution(eagle_leaf, eagle_grad, r_vl)
                else:
                    r_input = r_vl

                try:
                    heatmaps = map_vl_tokens_to_heatmaps(
                        r_input,
                        eagle_input_ids,
                        image_token_index=image_token_index,
                        num_image_tokens=num_image_tokens,
                        T=T_vid,
                        V=V,
                    )
                    image_heatmaps_np = heatmaps.squeeze(0).cpu().numpy()
                except (AssertionError, RuntimeError) as exc:
                    print(f"  sample {sample_idx} step {step}: heatmap failed ({exc}), skipping")
                    image_heatmaps_np = None

                vl_token_importance = r_vl.squeeze(0).cpu().numpy()
                eagle_input_importance = r_input.squeeze(0).cpu().numpy()
                n_state_tokens = result["n_state_tokens"]
                state_token_importance = r_sa.squeeze(0).cpu().numpy()[:n_state_tokens]

                # Redistribute the Chefer state-token budget across individual joints.
                # ReLU(grad*input) gives a proportional weighting; we then rescale so that
                # sum(state_joint_attr) == sum(state_token_importance) exactly.
                # This preserves the total relevance budget — the overview proportions are
                # numerically identical to what they were with the single state token bar.
                raw_joint_attr = result["state_attribution"][:n_raw_state_dims]
                state_token_total = float(state_token_importance.sum())
                raw_sum = float(raw_joint_attr.sum())
                if raw_sum > 1e-10:
                    state_joint_attr = raw_joint_attr / raw_sum * state_token_total
                else:
                    state_joint_attr = np.full(n_raw_state_dims, state_token_total / n_raw_state_dims)

                step_grp = grp.require_group(f"step_{step}")
                # DiT-level relevance (after EAGLE mixing) — kept for analysis
                step_grp.create_dataset(
                    "vl_token_importance", data=vl_token_importance.astype(np.float32)
                )
                # Pre-mixing relevance at raw patch/token level — used for heatmaps
                step_grp.create_dataset(
                    "eagle_input_importance", data=eagle_input_importance.astype(np.float32)
                )
                step_grp.create_dataset(
                    "state_token_importance", data=state_token_importance.astype(np.float32)
                )
                step_grp.create_dataset(
                    "state_joint_attribution", data=state_joint_attr.astype(np.float32)
                )
                if image_heatmaps_np is not None:
                    step_grp.create_dataset(
                        "image_heatmaps", data=image_heatmaps_np.astype(np.float32)
                    )

                if cfg.visualize and image_heatmaps_np is not None:
                    raw_frames = {k: obs[k] for k in loader.video_keys if k in obs}
                    ids_np = eagle_input_ids.squeeze(0).numpy()
                    displayable_mask = _make_displayable_mask(ids_np, image_token_index, tokenizer)
                    text_importance = eagle_input_importance.copy()
                    text_importance[~displayable_mask] = 0.0
                    image_importance_sum = float(eagle_input_importance[ids_np == image_token_index].sum())
                    token_labels = _decode_token_labels(ids_np, image_token_index, tokenizer)
                    task_span = _find_task_span(ids_np, obs, tokenizer, image_token_index)
                    if task_span is None:
                        print(f"  WARNING: task span not found for sample {sample_idx} — showing all tokens")
                    sample_vis_entries.append((
                        image_heatmaps_np, raw_frames, text_importance, state_joint_attr,
                        sample_idx, token_labels, image_importance_sum, state_joint_labels,
                        task_span,
                    ))

            # Mean vis entry: average heatmaps, text_importance, and image_importance_sum
            # across all captured steps for this sample.  State attribution is step-independent
            # (computed once from leaf.grad), so any single step's value is representative.
            # This produces a single summary video that is free of arbitrary step-index choice.
            if cfg.visualize and len(sample_vis_entries) > 0:
                vis_entries.append(_mean_vis_entry(sample_vis_entries))

            if (sample_idx + 1) % 10 == 0:
                print(f"  {sample_idx + 1}/{len(indices)} done")

    if cfg.visualize:
        _render_gif(vis_entries, loader.video_keys, out_root)

    print(f"Saved to {hdf5_path}")


def _render_gif(
    entries: list[tuple],
    view_names: list[str],
    out_root: Path,
) -> None:
    if not entries:
        return

    totals = [
        float(e[2].sum() + e[3].sum() + (e[6] if len(e) > 6 else 0.0))
        for e in entries
    ]

    # Global y-axis limits — scale to individual (non-aggregate) bars only so the
    # <sys>/<end> totals don't crush the task-token bars into invisibility.
    def _individual_max(e: tuple, total: float) -> float:
        task_span = e[8] if len(e) > 8 else None
        imp = e[2]
        if task_span is not None:
            imp = imp[task_span[0]:task_span[1]]
        return float((imp / max(total, 1e-8) * 100).max()) if len(imp) > 0 else 0.0

    text_y_max_pct = float(max(
        _individual_max(e, totals[i]) for i, e in enumerate(entries)
    ))
    state_y_max_pct = float(max(
        (e[3] / max(totals[i], 1e-8) * 100).max()
        for i, e in enumerate(entries)
    ))

    state_joint_labels = entries[0][7] if len(entries[0]) > 7 else None

    frames = []
    for entry in entries:
        image_heatmaps_np, raw_frames, text_importance, state_joint_attr, sample_idx = entry[:5]
        token_labels = entry[5] if len(entry) > 5 else None
        image_importance_sum = entry[6] if len(entry) > 6 else 0.0
        task_span = entry[8] if len(entry) > 8 else None
        frames.append(build_summary_frame(
            heatmaps=image_heatmaps_np,
            raw_frames=raw_frames,
            text_importance=text_importance,
            state_joint_attr=state_joint_attr,
            image_importance_sum=image_importance_sum,
            view_names=view_names,
            sample_idx=sample_idx,
            text_y_max_pct=text_y_max_pct,
            state_y_max_pct=state_y_max_pct,
            token_labels=token_labels,
            state_joint_labels=state_joint_labels,
            task_span=task_span,
        ))

    save_summary_gif(frames, out_root / "attribution.gif", fps=30)

    import cv2
    cv2.imwrite(str(out_root / "attribution_preview.png"), frames[0][..., ::-1])


def _mean_vis_entry(entries: list[tuple]) -> tuple:
    """
    Average heatmaps, text_importance, and image_importance_sum across multiple step entries
    for the same sample.  All other fields (raw_frames, sample_idx, token_labels,
    state_joint_attr, state_joint_labels) are taken from the first entry — they are either
    step-independent (state attribution, raw frames) or constant (labels, sample index).

    Why this is valid: `eagle_input_importance` (the heatmap source) has the same relative
    spatial pattern at every step because it is derived from a single fixed `leaf.grad`.
    Only the absolute scale varies (normalised to match `r_vl.sum()` per step).
    Averaging therefore gives a representative spatial pattern at the mean scale.
    `text_importance` (Chefer-based) does vary per step — the mean gives an unbiased
    summary of which prompt tokens mattered across the denoising trajectory.
    """
    heatmaps_mean = np.mean([e[0] for e in entries], axis=0)
    text_importance_mean = np.mean([e[2] for e in entries], axis=0)
    image_importance_sum_mean = float(np.mean([e[6] for e in entries]))
    first = entries[0]
    return (
        heatmaps_mean,          # [T, V, ph, pw]
        first[1],               # raw_frames — same for all steps
        text_importance_mean,   # [L] — averaged over steps
        first[3],               # state_joint_attr — step-independent
        first[4],               # sample_idx
        first[5],               # token_labels
        image_importance_sum_mean,
        first[7],               # state_joint_labels
        first[8] if len(first) > 8 else None,  # task_span — constant across steps
    )


def _unnormalize_action(policy: Gr00tPolicy, action_pred_np: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(action_pred_np).unsqueeze(0)
    unnorm = policy.unapply_transforms({"action": tensor})
    result = unnorm.get("action", tensor)
    if isinstance(result, torch.Tensor):
        return result.squeeze(0).numpy()
    return np.array(result).squeeze(0)


def _get_num_image_tokens(backbone) -> int:
    try:
        return backbone.eagle_model.config.num_image_tokens
    except AttributeError:
        pass
    try:
        return backbone.eagle_model.processor.num_image_tokens
    except AttributeError:
        pass
    return 256


def _get_tokenizer(policy: Gr00tPolicy):
    try:
        from gr00t.model.transforms import GR00TTransform
        from gr00t.data.transform.base import ComposedModalityTransform

        transform = policy._modality_transform
        if isinstance(transform, ComposedModalityTransform):
            for t in transform.transforms:
                if isinstance(t, GR00TTransform):
                    return t.eagle_processor.tokenizer
    except Exception:
        pass
    return None


_UR5_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow",
    "wrist_1", "wrist_2", "wrist_3",
]


def _state_joint_labels(data_config) -> list[str]:
    """Build one label per raw state dimension from state_keys and state_slices."""
    labels = []
    for key, (start, end) in zip(data_config.state_keys, data_config.state_slices):
        name = key.removeprefix("state.")
        n = end - start
        if n == 1:
            labels.append(name)
        elif name == "robot_arm" and n == len(_UR5_JOINT_NAMES):
            labels.extend(_UR5_JOINT_NAMES)
        else:
            for i in range(n):
                labels.append(f"{name}_{i}")
    return labels


def _make_displayable_mask(
    ids_np: np.ndarray,
    image_token_index: int,
    tokenizer,
) -> np.ndarray:
    """Boolean mask: True where a VL token is a real prompt/language token worth displaying."""
    is_image = ids_np == image_token_index
    if tokenizer is not None:
        special_ids = set(tokenizer.all_special_ids)
        is_special = np.array([int(tid) in special_ids for tid in ids_np], dtype=bool)
    else:
        is_special = np.zeros(len(ids_np), dtype=bool)
    return ~is_image & ~is_special


def _decode_token_labels(
    ids_np: np.ndarray,
    image_token_index: int,
    tokenizer,
) -> list[str]:
    """
    Decode each token ID to a short human-readable label.
    Image-patch positions are replaced with '<img>'.
    When tokenizer is unavailable, falls back to 'i{idx}'.
    """
    if tokenizer is None:
        return [f"i{i}" if ids_np[i] == image_token_index else f"t{i}" for i in range(len(ids_np))]

    labels = []
    for i, tid in enumerate(ids_np):
        if tid == image_token_index:
            labels.append("<img>")
        else:
            try:
                text = tokenizer.decode([int(tid)], skip_special_tokens=False)
                text = text.replace("\n", "\\n").replace("\r", "").strip()
                labels.append(text[:12] if text else f"t{i}")
            except Exception:
                labels.append(f"t{i}")
    return labels


def _find_task_span(
    ids_np: np.ndarray,
    obs: dict,
    tokenizer,
    image_token_index: int,
) -> tuple[int, int] | None:
    """
    Find the token span of the task description within ids_np.

    Works by decoding each non-image token to text, concatenating into a string,
    then finding the task description as a substring.  This avoids re-tokenization
    fragility (subword prefix characters like Ġ, leading spaces, etc.).

    Returns (start, end) in full ids_np coordinates, or None on failure.
    """
    if tokenizer is None:
        return None

    task_desc = None
    for key, val in obs.items():
        if "annotation" in key or "task_description" in key:
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                task_desc = str(val[0])
            elif isinstance(val, (str, bytes)):
                task_desc = str(val)
            if task_desc:
                break
    if not task_desc:
        return None

    non_img = [(i, int(ids_np[i])) for i in range(len(ids_np)) if ids_np[i] != image_token_index]

    token_texts: list[str] = []
    for _, tid in non_img:
        try:
            token_texts.append(tokenizer.decode([tid], skip_special_tokens=False))
        except Exception:
            token_texts.append("")

    full_text = "".join(token_texts)
    pos = full_text.find(task_desc)
    if pos < 0:
        pos = full_text.lower().find(task_desc.lower())
    if pos < 0:
        return None

    end_char = pos + len(task_desc)
    cum = 0
    start_orig = end_orig = None
    for k, (orig_i, _) in enumerate(non_img):
        tok_end = cum + len(token_texts[k])
        if start_orig is None and tok_end > pos:
            start_orig = orig_i
        if end_orig is None and tok_end >= end_char:
            end_orig = orig_i + 1
            break
        cum = tok_end

    if start_orig is None or end_orig is None:
        return None
    return (start_orig, end_orig)


