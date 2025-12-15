from __future__ import annotations
from pydantic import Field
from typing import List, Optional
import time

from extension.utils.argparsing import AdditionalArgsBase
    

class LoraArgs(AdditionalArgsBase):
    lora_rank: int = Field(
        description="Rank for LoRA. If 0, LoRA is disabled.",
        default=0,
    )
    lora_alpha: int = Field(
        description="Alpha value for LoRA.",
        default=16,
    )
    lora_dropout: float = Field(
        description="Dropout rate for LoRA.",
        default=0.1,
    )
    lora_full_model: bool = Field(
        description="If False, only the action head will be trained with LoRA.",
        default=False,
    )

    def to_cli(self) -> List[str]:
        argv: List[str] = [
            "--lora-rank", str(self.lora_rank),
            "--lora-alpha", str(self.lora_alpha),
            "--lora-dropout", str(self.lora_dropout),
        ]
        if self.lora_full_model:
            argv += ["--lora-full-model"]
        return argv


class ModelArgs(AdditionalArgsBase):
    base_model_path: str = Field(
        description="Path or HuggingFace model ID for the base GR00T model.",
        default="nvidia/GR00T-N1.5-3B",
    )
    tune_llm: bool = Field(
        description="Whether to fine-tune the language model backbone.",
        default=False,
    )
    tune_visual: bool = Field(
        description="Whether to fine-tune the vision tower.",
        default=False,
    )
    tune_projector: bool = Field(
        description="Whether to fine-tune the action head projector.",
        default=True,
    )
    tune_diffusion_model: bool = Field(
        description="Whether to fine-tune the diffusion model (DiT) in the action head.",
        default=True,
    )
    resume_from: Optional[str] = Field(
        description="Whether to resume from a checkpoint and if so, which checkpoint path.",
        default=None,
    )
    model_output_dir: str = Field(
        description="Directory to save model checkpoints.",
        default="./models",
    )
    lora_args: LoraArgs = Field(
        description="LoRA (Low-Rank Adaptation) related arguments.",
        default=LoraArgs(),
    )
    push_to_hub: bool = Field(
        description="Whether to push the fine-tuned model to the HuggingFace Hub after training.",
        default=False,
    )
    hf_user: str = Field(
        description="HuggingFace user name for pushing the model to the Hub.",
        default="yannikheizmanntng",
    )

    def to_cli(self) -> List[str]:
        argv: List[str] = [
            "--base-model-path", self.base_model_path,
        ]
        argv += ["--tune-llm"] if self.tune_llm else ["--no-tune-llm"]
        argv += ["--tune-visual"] if self.tune_visual else ["--no-tune-visual"]
        argv += ["--tune-projector"] if self.tune_projector else ["--no-tune-projector"]
        argv += ["--tune-diffusion-model"] if self.tune_diffusion_model else ["--no-tune-diffusion-model"]
        
        if self.resume_from is not None:
            argv += ["--output-dir", self.resume_from]
            argv += ["--resume"]
        else:
            output_dir = f"{self.model_output_dir}/{self.base_model_path.split('/')[-1]}_{time.strftime('%Y%m%d-%H%M%S')}" 
            argv += ["--output-dir", output_dir]

        argv += self.lora_args.to_cli()
        return argv
