from __future__ import annotations
from pydantic import Field
from typing import List, Literal
from enum import Enum
from pathlib import Path
import shutil

from extension.utils.argparsing import AdditionalArgsBase


class DataConfigOptions(str, Enum):
        UR5_Abs_Delta_4_Cfg = "UR5_Abs_Delta_4_Cfg"
        UR5_Abs_Delta_2_Cfg = "UR5_Abs_Delta_2_Cfg"
        UR5_Abs_Delta_4_Cfg_Det = "UR5_Abs_Delta_4_Cfg_Det"
        UR5_Abs_Delta_Bin_Grp_4_Cfg = "UR5_Abs_Delta_Bin_Grp_4_Cfg"
        UR5_Abs_Delta_Abs_Grp_4_Cfg = "UR5_Abs_Delta_Abs_Grp_4_Cfg"


class DatasetArgs(AdditionalArgsBase):
    data_config: str = Field(
        description="Data configuration to use for the fine-tuning.",
        default="UR5_Abs_Delta_Abs_Grp_4_Cfg",
    )
    data_configs_path: str = Field(
        description="Path to the file containing the data configuration.",
        default="extension/dataset/_config.py",
    )
    dataset_from_hf: bool = Field(
        description=(
            """If True, treat entries of dataset_path as Hugging Face dataset repo_id(s), 
              download them and store locally before training."""
        ),
        default=False,
    )
    datasets_output_dir: str = Field(
        description=(
            "Local base directory where datasets are stored/ensured. "
            "Used for both HF downloads and local dataset caching."
        ),
        default="./datasets",
    )
    dataset_path: List[str] = Field(
        description="Path(s) to LeRobot dataset directory/directories. All datasets must share the same data config.",
        default=["/home/innovation-hacking/heizmany/ur5_chess/datasets/dataset_20260209_112625/lerobot"],
    )
    embodiment_tag: str = Field(
        description="Embodiment tag to use for training. Overrides dataset embodiment tag.",
        default="new_embodiment",
    )
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = Field(
        description="Video backend to use for training.",
        default="torchvision_av",
    )
    balance_dataset_weights: bool = Field(
        description="If True, balance dataset weights by total trajectories per dataset (mixture mode).",
        default=True,
    )
    balance_trajectory_weights: bool = Field(
        description="If True, sample trajectories weighted by length within each dataset (mixture mode).",
        default=True,
    )

    def _resolve_data_config_string(self) -> str:
        module = self.data_configs_path.replace(".py", "").replace("/", ".")
        class_name = self.data_config
        return f"{module}:{class_name}"
    
    def _safe_name(self, s: str) -> str:
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

    def _ensure_output_base(self) -> Path:
        output_path = f"{self.datasets_output_dir}/{self.data_config}"
        base = Path(output_path).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _download_hf_dataset(self, repo_id: str, dest: Path) -> Path:
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for dataset_from_hf=True. "
                "Install with: pip install huggingface_hub"
            ) from e
        if dest.exists() and any(dest.iterdir()):
            return dest
        dest.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        return dest

    def _cache_local_dataset(self, src: Path, dest: Path) -> Path:
        if dest.exists() and any(dest.iterdir()):
            return dest
        if not src.exists():
            raise FileNotFoundError(f"Local dataset path does not exist: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return dest

    def _localize_datasets(self) -> List[str]:
        if not self.dataset_path:
            raise ValueError("dataset_args.dataset_path must contain at least one dataset path")
        base = self._ensure_output_base()
        localized: List[str] = []
        if self.dataset_from_hf:
            for repo_id in self.dataset_path:
                name = self._safe_name(repo_id)
                dest = base / name
                local_path = self._download_hf_dataset(repo_id, dest)
                localized.append(str(local_path))
        else:
            for p in self.dataset_path:
                src = Path(p).expanduser().resolve()
                name = self._safe_name(src.name)
                dest = base / name
                if src == dest:
                    localized.append(str(src))
                    continue
                cached_path = self._cache_local_dataset(src, dest)
                localized.append(str(cached_path))
        return localized

    def to_cli(self) -> List[str]:
        localized_paths = self._localize_datasets()

        argv: List[str] = []
        argv += ["--dataset-path", *localized_paths]
        argv += ["--data-config", self._resolve_data_config_string()]
        argv += ["--embodiment-tag", self.embodiment_tag]
        argv += ["--video-backend", self.video_backend]

        argv += ["--balance-dataset-weights"] if self.balance_dataset_weights else ["--no-balance-dataset-weights"]
        argv += ["--balance-trajectory-weights"] if self.balance_trajectory_weights else ["--no-balance-trajectory-weights"]
        return argv