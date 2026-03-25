from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LayerCapture:
    attention_probs: torch.Tensor   # [B, H, Q, K] — retain_grad() already called
    kind: str                        # "self" | "cross"
    layer_idx: int
    denoise_step: int


class AttentionStore:
    def __init__(self, capture_steps: list[int]):
        self._capture_steps = set(capture_steps)
        self._captures: list[LayerCapture] = []
        self._current_step: int = -1

    def set_step(self, step: int) -> None:
        self._current_step = step

    def is_capturing(self) -> bool:
        return self._current_step in self._capture_steps

    def add(self, attention_probs: torch.Tensor, kind: str, layer_idx: int) -> None:
        if self.is_capturing():
            self._captures.append(LayerCapture(attention_probs, kind, layer_idx, self._current_step))

    def captures_for_step(self, step: int) -> list[LayerCapture]:
        return [c for c in self._captures if c.denoise_step == step]

    def all_captures(self) -> list[LayerCapture]:
        return list(self._captures)

    def clear(self) -> None:
        self._captures.clear()
        self._current_step = -1
