from __future__ import annotations

import argparse
import warnings

# Suppress torchvision video-backend deprecation noise (harmless; backend still works).
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision",
    category=UserWarning,
)

from extension.interpretability._config import InterpretabilityArgs
from extension.interpretability._offline_eval import run_offline_eval


def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline Chefer-style token relevance attribution for GR00T N1.5"
    )
    p.add_argument(
        "--model_path",
        default="/home/innovation-hacking/heizmany/Isaac-GR00T/models/model_7",
        help="Path to the finetuned model directory.",
    )
    p.add_argument(
        "--dataset_path",
        default="/home/innovation-hacking/heizmany/ur5_chess/datasets/dataset_7/lerobot",
        help="Path to the LeRobot dataset.",
    )
    p.add_argument(
        "--output_path",
        default=None,
        help="Output directory. Defaults to {model_path}/interpretability.",
    )
    p.add_argument(
        "--capture_steps",
        type=int,
        nargs="+",
        default=[0, 3],
        help="Denoising step indices to capture (e.g. --capture_steps 0 3).",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Stop after this many steps. Omit to run on the full dataset.",
    )
    p.add_argument(
        "--device",
        default="cuda",
    )
    p.add_argument(
        "--no_visualize",
        action="store_true",
        help="Disable GIF output.",
    )
    a = p.parse_args()

    cfg = InterpretabilityArgs(
        model_path=a.model_path,
        dataset_path=a.dataset_path,
        output_path=a.output_path,
        capture_steps=a.capture_steps,
        max_samples=a.max_samples,
        device=a.device,
        visualize=not a.no_visualize,
    )
    run_offline_eval(cfg)


if __name__ == "__main__":
    main()
