from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class InterpretabilityArgs(BaseModel):
    model_path: str
    dataset_path: str | None = None       # auto-detected from model args.json if omitted
    output_path: str | None = None        # defaults to {model_path}/interpretability
    embodiment_tag: str | None = None     # auto-detected from model args.json if omitted
    capture_steps: list[int] = Field(default=[0])
    max_samples: int | None = None
    device: str = "cuda"
    visualize: bool = True

    @model_validator(mode="after")
    def set_output_path_default(self) -> InterpretabilityArgs:
        if self.output_path is None:
            self.output_path = f"{self.model_path}/interpretability"
        return self
