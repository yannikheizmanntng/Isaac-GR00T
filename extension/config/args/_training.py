from __future__ import annotations
from pydantic import Field
from typing import Literal, List

from extension.utils.argparsing import AdditionalArgsBase


class TrainingArgs(AdditionalArgsBase):
    learning_rate: float = Field(
        description="Learning rate for training.",
        default=1e-4,
    )
    weight_decay: float = Field(
        description="Weight decay for AdamW.",
        default=1e-5,
    )
    warmup_ratio: float = Field(
        description="Warmup ratio of total training steps.",
        default=0.05,
    )
    batch_size: int = Field(
        description="Batch size per GPU for training.",
        default=4,
    )
    dataloader_num_workers: int = Field(
        description="Number of workers for data loading per GPU.",
        default=2,
    )
    dataloader_persistent_workers: bool = Field(
        description="Keep dataloader workers alive across epochs. Disable to reset worker RAM each epoch.",
        default=False,
    )
    dataloader_prefetch_factor: int = Field(
        description="Prefetch factor for data loading.",
        default=4,
    )
    gradient_accumulation_steps: int = Field(
        description="Gradient accumulation steps for training.",
        default=1,
    )
    max_steps: int = Field(
        description="Maximum number of training steps.",
        default=100_000,
    )
    save_steps: int = Field(
        description="Number of steps between saving checkpoints.",
        default=20000,
    )
    num_gpus: int = Field(
        description="Number of GPUs to use for training (script will switch to torchrun if > 1).",
        default=1,
    )
    report_to: Literal["wandb", "tensorboard", "azure_ml"] = Field(
        description="Where to report training metrics.",
        default="wandb",
    )
    eval_split: float = Field(
        description="Fraction of trajectories held out as eval set (0.0 = disable eval).",
        default=0.10,
    )

    def to_cli(self) -> List[str]:
        return [
            "--learning-rate", str(self.learning_rate),
            "--weight-decay", str(self.weight_decay),
            "--warmup-ratio", str(self.warmup_ratio),
            "--batch-size", str(self.batch_size),
            "--dataloader-num-workers", str(self.dataloader_num_workers),
            "--dataloader-persistent-workers" if self.dataloader_persistent_workers else "--no-dataloader-persistent-workers",
            "--dataloader-prefetch-factor", str(self.dataloader_prefetch_factor),
            "--gradient-accumulation-steps", str(self.gradient_accumulation_steps),
            "--max-steps", str(self.max_steps),
            "--save-steps", str(self.save_steps),
            "--num-gpus", str(self.num_gpus),
            "--report-to", self.report_to,
            "--eval-split", str(self.eval_split),
        ]