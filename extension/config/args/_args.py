from __future__ import annotations
from pydantic import Field
from typing import List, Literal

from extension.utils.argparsing import PydanticArgsBase
from ._training import TrainingArgs
from ._model import ModelArgs
from ._dataset import DatasetArgs
from ._inference import InferenceArgs


class Args(PydanticArgsBase):
    mode: Literal["run_finetune", "start_inference_server"] = Field(
        description="Mode to run the script in.",
        default="start_inference_server",
    )
    dataset_args: DatasetArgs = Field(
        description="Dataset related arguments.",
        default=DatasetArgs(),
    )
    model_args: ModelArgs = Field(
        description="Model related arguments.",
        default=ModelArgs(),
    )
    training_args: TrainingArgs = Field(
        description="Training related arguments.",
        default=TrainingArgs(),
    )
    inference_args: InferenceArgs = Field(
        description="Inference related arguments.",
        default=InferenceArgs(),
    )

    def to_gr00t_finetune_argv(self) -> List[str]:
        argv: List[str] = []
        argv += self.dataset_args.to_cli()
        argv += self.model_args.to_cli()
        argv += self.training_args.to_cli()
        return argv
    
    def to_gr00t_inference_argv(self) -> List[str]:
        argv: List[str] = []
        argv += self.inference_args.to_cli()
        argv += ["--data-config", self.dataset_args._resolve_data_config_string()]
        argv += ["--embodiment-tag", self.dataset_args.embodiment_tag]
        return argv
