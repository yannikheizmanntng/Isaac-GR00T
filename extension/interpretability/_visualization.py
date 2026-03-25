from __future__ import annotations

from pathlib import Path

import numpy as np


def build_summary_frame(
    heatmaps: np.ndarray,
    raw_frames: dict[str, np.ndarray],
    text_importance: np.ndarray,
    state_joint_attr: np.ndarray,
    image_importance_sum: float,
    view_names: list[str],
    sample_idx: int,
    text_y_max_pct: float,
    state_y_max_pct: float,
    cam_width: int = 480,
    token_labels: list[str] | None = None,
    state_joint_labels: list[str] | None = None,
    task_span: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Layout
    ------
    Left  : camera views with heatmap overlays, stacked vertically.
    Right :
      Top    — token attribution bar chart.
               Image token positions are always excluded (their relevance is in the
               modality overview).
               When task_span is provided:
                 · <sys>  one bar summing all pre-task non-image tokens
                 · one bar per task-description token
                 · <end>  one bar summing all post-task non-image tokens
               When task_span is None: one bar per non-image token.
      Middle — token text strip aligned to bars.
      Bottom — per-joint proprioception | modality overview (image / prompt / proprio %)
    """
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # Vivid red → orange → cream colormap.
    # Low-attention areas are blended transparently (original image shows through),
    # so the dark end of the colormap doesn't matter — only the vivid high end does.
    _HEAT_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "heat_pink_red",
        ["#3A0010", "#A00025", "#E03050", "#FF8899", "#FFE8EC"],
    )
    _C_IMAGE = "#D9585C"  # _PAL["red"][1] — board rotation chart colour from distribution card

    # ── Camera panels ──────────────────────────────────────────────────────────
    T, V, ph, pw = heatmaps.shape

    # Global min/max across all views at the display timestep so both cameras
    # share the same colour scale — the same red means the same attribution value.
    t_indices = []
    valid_views = []
    for v_idx, view_name in enumerate(view_names[:V]):
        frames = raw_frames.get(view_name)
        if frames is not None:
            t_indices.append(min(T - 1, frames.shape[0] - 1))
            valid_views.append((v_idx, view_name, frames))

    if valid_views:
        t_idx_global = t_indices[0]  # same timestep for all views
        all_heats = np.stack([heatmaps[t_idx_global, v_idx] for v_idx, _, _ in valid_views])
        g_min, g_max = float(all_heats.min()), float(all_heats.max())
    else:
        g_min, g_max = 0.0, 1.0

    cam_panels = []
    for v_idx, view_name, frames in valid_views:
        t_idx = min(T - 1, frames.shape[0] - 1)
        frame = frames[t_idx].copy()
        H, W = frame.shape[:2]
        heat = heatmaps[t_idx, v_idx]
        heat_norm = (heat - g_min) / max(g_max - g_min, 1e-8)
        heat_up = cv2.resize(heat_norm.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        colored = (_HEAT_CMAP(heat_up)[..., :3] * 255).astype(np.uint8)
        # Lighten the base image with a faint white veil so dark regions
        # don't swallow low-attention heatmap colours.
        base = np.clip(frame.astype(np.float32) * 0.85 + 255 * 0.15, 0, 255)
        # Alpha-blend: heat value controls opacity of the colour overlay.
        # gamma < 1 boosts mid-range values so they are clearly visible.
        alpha = (heat_up ** 0.6)[..., np.newaxis].astype(np.float32)
        overlay = np.clip(
            base * (1.0 - alpha * 0.85) + colored.astype(np.float32) * alpha * 0.85,
            0, 255,
        ).astype(np.uint8)
        scale = cam_width / W
        cam_panels.append(cv2.resize(overlay, (cam_width, int(H * scale)), interpolation=cv2.INTER_LINEAR))

    if not cam_panels:
        cam_panels = [np.zeros((cam_width, cam_width, 3), dtype=np.uint8)]
    left = np.concatenate(cam_panels, axis=0)
    total_h = left.shape[0]

    # ── Data ──────────────────────────────────────────────────────────────────
    total = max(float(text_importance.sum() + state_joint_attr.sum() + image_importance_sum), 1e-8)
    L = len(text_importance)
    n_joints = len(state_joint_attr)

    scores_pct      = text_importance / total * 100
    scores_state    = state_joint_attr / total * 100
    image_pct       = image_importance_sum / total * 100
    text_total_pct  = float(text_importance.sum()) / total * 100
    state_total_pct = float(state_joint_attr.sum()) / total * 100

    labels_state = state_joint_labels if state_joint_labels is not None else [f"j{i}" for i in range(n_joints)]

    is_img = np.array([
        (token_labels[i] if token_labels else "") == "<img>"
        for i in range(L)
    ], dtype=bool)

    def _label(i: int) -> str:
        lbl = token_labels[i] if token_labels is not None else ""
        return lbl.strip().lstrip("▁Ġ▾ ").strip()

    def _is_special(i: int) -> bool:
        lbl = token_labels[i] if token_labels is not None else ""
        return (lbl.startswith("<") and lbl.endswith(">")) or lbl in {"", " ", "\n", "\\n"}

    # ── Build bar segments (image positions excluded) ──────────────────────────
    # Each entry: (bar_height, label_text, color, is_aggregate)
    # x-positions are just 0, 1, 2, ... in order of segments.

    _C_TEXT = "#7799D3"    # blue  — _PAL["blue"][1]
    _C_AGG  = "#C8CDDA"   # neutral grey for <sys> and <end>
    _C_SPEC = "#D5D8DC"   # light grey for special tokens inside task span

    bars: list[tuple[float, str, str, bool]] = []

    if task_span is not None:
        pre_s, task_s, task_e, post_e = 0, task_span[0], task_span[1], L

        # Pre-task aggregate (sum over non-image positions)
        pre_mask = ~is_img
        pre_mask[task_s:] = False
        pre_sum = float(scores_pct[pre_mask].sum())
        bars.append((pre_sum, "<sys>", _C_AGG, True))

        # Individual task tokens (skip image positions inside span if any)
        for i in range(task_s, task_e):
            if is_img[i]:
                continue
            col = _C_SPEC if _is_special(i) else _C_TEXT
            bars.append((float(scores_pct[i]), _label(i), col, False))

        # Post-task aggregate
        post_mask = ~is_img
        post_mask[:task_e] = False
        post_sum = float(scores_pct[post_mask].sum())
        if post_sum > 0:
            bars.append((post_sum, "<end>", _C_AGG, True))
    else:
        # No task span — show every non-image token individually
        for i in range(L):
            if is_img[i]:
                continue
            col = _C_SPEC if _is_special(i) else _C_TEXT
            bars.append((float(scores_pct[i]), _label(i), col, False))

    n_bars = len(bars)

    # ── Figure ────────────────────────────────────────────────────────────────
    # Camera panel is ~30 % of total width → chart panel is ~60 % → gap ~10 %.
    # chart_w = 2 × cam_width gives that ratio (480 cam : 960 chart).
    dpi = 200
    chart_w_px = cam_width * 2
    fig = plt.figure(figsize=(chart_w_px / dpi, total_h / dpi), dpi=dpi,
                     layout="constrained")
    fig.get_layout_engine().set(h_pad=0.01, hspace=0.0, w_pad=0.01, wspace=0.0)

    # 50 % token bars / 50 % bottom charts.
    # wspace between proprio and modality ≈ 24 px (same as image–chart gap).
    # 24 px / (chart_w_px / 2 average subplot width) ≈ 0.05.
    gs_outer = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.45)
    gs_top = gs_outer[0].subgridspec(2, 1, height_ratios=[5, 1], hspace=0.0)
    gs_bot = gs_outer[1].subgridspec(1, 2, width_ratios=[n_joints, 3], wspace=0.05)

    ax_bars    = fig.add_subplot(gs_top[0])
    ax_tokens  = fig.add_subplot(gs_top[1])
    ax_state   = fig.add_subplot(gs_bot[0])
    ax_overview = fig.add_subplot(gs_bot[1])

    # ── Token attribution bar chart ───────────────────────────────────────────
    xs   = np.arange(n_bars)
    hs   = [b[0] for b in bars]
    cols = [b[2] for b in bars]

    ax_bars.bar(xs, hs, color=cols, width=0.85, linewidth=0)
    ax_bars.set_xlim(-0.5, n_bars - 0.5)
    ax_bars.set_ylim(0, max(text_y_max_pct * 1.1, 0.1))
    ax_bars.set_xticks([])
    ax_bars.set_ylabel("% relevance", fontsize=7)
    ax_bars.set_title(f"token attribution — sample {sample_idx:04d}", fontsize=8)
    ax_bars.tick_params(axis="y", labelsize=6)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.spines["bottom"].set_visible(False)

    # ── Token text strip ──────────────────────────────────────────────────────
    ax_tokens.set_xlim(-0.5, n_bars - 0.5)
    ax_tokens.set_ylim(0, 1)
    ax_tokens.axis("off")

    for xi, (_, lbl, color, is_agg) in enumerate(bars):
        if is_agg:
            ax_tokens.text(
                xi, 0.5, lbl, ha="center", va="center", fontsize=6,
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#E0E3E8", edgecolor="none"),
            )
        elif lbl:
            ax_tokens.text(xi, 0.5, lbl[:8], ha="center", va="center",
                           fontsize=6, color="#2C3E50")

    # ── Proprioception bars ───────────────────────────────────────────────────
    ax_state.bar(np.arange(n_joints), scores_state, color="#AFCA9E", width=0.7)  # green — _PAL["green"][0]
    ax_state.set_xlim(-0.5, n_joints - 0.5)
    ax_state.set_ylim(0, max(state_y_max_pct * 1.1, 0.1))
    ax_state.set_xticks(np.arange(n_joints))
    ax_state.set_xticklabels(labels_state, rotation=35, ha="right", fontsize=7)
    ax_state.set_ylabel("% relevance", fontsize=7)
    ax_state.set_title("proprioception", fontsize=8)
    ax_state.tick_params(axis="y", labelsize=6)
    ax_state.spines["top"].set_visible(False)
    ax_state.spines["right"].set_visible(False)

    # ── Modality overview ─────────────────────────────────────────────────────
    ax_overview.bar(range(3), [image_pct, text_total_pct, state_total_pct],
                    color=[_C_IMAGE, "#7799D3", "#AFCA9E"], width=0.6)
    ax_overview.set_xlim(-0.5, 2.5)
    ax_overview.set_ylim(0, 100)
    ax_overview.set_xticks(range(3))
    ax_overview.set_xticklabels(["image", "prompt", "proprio"],
                                 rotation=35, ha="right", fontsize=7)
    ax_overview.set_ylabel("% relevance", fontsize=7)
    ax_overview.set_title("modality", fontsize=8)
    ax_overview.tick_params(axis="y", labelsize=6)
    ax_overview.spines["top"].set_visible(False)
    ax_overview.spines["right"].set_visible(False)

    # ── Render ────────────────────────────────────────────────────────────────
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    chart_rgba = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    chart_rgb = chart_rgba[..., :3]
    plt.close(fig)

    chart_h, chart_w = chart_rgb.shape[:2]
    if chart_h != total_h:
        chart_rgb = cv2.resize(chart_rgb, (chart_w, total_h), interpolation=cv2.INTER_LINEAR)

    gap = np.full((total_h, 24, 3), 255, dtype=np.uint8)
    return np.concatenate([left, gap, chart_rgb], axis=1)


def save_summary_gif(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int = 4,
) -> None:
    try:
        from PIL import Image
    except ImportError:
        print("save_summary_gif: Pillow not available, saving individual PNGs instead")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        for i, frame in enumerate(frames):
            cv2.imwrite(str(output_path.with_suffix("")) + f"_{i:04d}.png", frame[..., ::-1])
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    print(f"Saved GIF ({len(frames)} frames, {fps} fps) → {output_path}")
