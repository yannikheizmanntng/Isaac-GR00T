from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import torch

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform

from extension.interpretability._config import InterpretabilityArgs


class DatasetLoader:
    """
    Loads a LeRobot dataset and yields raw obs dicts compatible with Gr00tPolicy.explain_action.

    The data config (video/state/action key layout) is auto-detected from the
    model's args.json if present, or defaults to all keys found in the dataset.

    Returns raw (un-transformed) obs dicts; the policy applies its own transforms internally.
    Action ground-truth is returned in raw dataset space.
    """

    def __init__(self, cfg: InterpretabilityArgs, policy):
        self._policy = policy
        self._cfg = cfg

        model_args = self._load_model_args(cfg.model_path)
        self._data_config = self._build_data_config(cfg, model_args)
        self._embodiment_tag = model_args.get("dataset_args", {}).get(
            "embodiment_tag", cfg.embodiment_tag
        )

        dataset_path = cfg.dataset_path or self._resolve_dataset_path(model_args)

        no_op = ComposedModalityTransform(transforms=[])
        self._dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=self._data_config.modality_config(),
            embodiment_tag=self._embodiment_tag,
            video_backend="torchvision_av",
            transforms=no_op,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def episode_indices(self, start_global_idx: int = 0) -> list[int]:
        """
        Return all global step indices that belong to the same episode (trajectory) as
        `start_global_idx`.  The indices are contiguous and sorted in episode order.

        Use this to restrict evaluation to a single coherent episode rather than letting
        the loop cross an episode boundary when max_samples exceeds episode length.
        """
        target_trajectory_id = self._dataset.all_steps[start_global_idx][0]
        return [
            i for i, (traj_id, _) in enumerate(self._dataset.all_steps)
            if traj_id == target_trajectory_id
        ]

    def load(self, global_step_idx: int) -> tuple[dict, np.ndarray]:
        """
        Returns (obs_dict, gt_action_np).

        obs_dict keys follow the modality.key convention expected by Gr00tPolicy
        (e.g. "video.camera_front", "state.robot_arm", "annotation.human.task_description").

        gt_action_np is float32 [H, D] in raw (un-normalized) dataset space.
        """
        trajectory_id, base_index = self._dataset.all_steps[global_step_idx]
        step = self._dataset.get_step_data(trajectory_id, base_index)

        action_keys = self._data_config.action_keys
        gt_parts = [step[k] for k in action_keys if k in step]
        gt_action = np.concatenate(gt_parts, axis=-1).astype(np.float32)

        obs = {k: v for k, v in step.items() if k not in action_keys}
        return obs, gt_action

    @property
    def video_keys(self) -> list[str]:
        return self._data_config.video_keys

    @property
    def embodiment_tag(self) -> str:
        return self._embodiment_tag

    def _resolve_dataset_path(self, model_args: dict) -> str:
        paths = model_args.get("dataset_args", {}).get("dataset_path", [])
        if isinstance(paths, list) and paths:
            path = paths[0]
        elif isinstance(paths, str):
            path = paths
        else:
            raise ValueError(
                "dataset_path not specified and could not be found in model args.json"
            )
        print(f"DatasetLoader: using dataset path from model args.json: {path}")
        return path

    def _load_model_args(self, model_path: str) -> dict:
        args_path = Path(model_path) / "args.json"
        if not args_path.exists():
            return {}
        with open(args_path) as f:
            return json.load(f)

    def _build_data_config(self, cfg: InterpretabilityArgs, model_args: dict):
        dataset_args = model_args.get("dataset_args", {})
        config_class_name = dataset_args.get("data_config")
        configs_path = dataset_args.get("data_configs_path")

        if config_class_name and configs_path:
            module_name = configs_path.replace("/", ".").removesuffix(".py")
            try:
                mod = importlib.import_module(module_name)
                if hasattr(mod, config_class_name):
                    print(f"DatasetLoader: using data config '{config_class_name}' from {module_name}")
                    return getattr(mod, config_class_name)()
            except ImportError:
                pass

        from extension.dataset._config import UR5_Abs_Delta_4_Cfg
        print("DatasetLoader: falling back to UR5_Abs_Delta_4_Cfg")
        return UR5_Abs_Delta_4_Cfg()
