from __future__ import annotations

from typing import Any

import torch
from diffusers.models.attention_processor import Attention

from gr00t.explain._attention_store import AttentionStore


class CaptureAttnProcessor:
    def __init__(self, store: AttentionStore, layer_idx: int):
        self._store = store
        self._layer_idx = layer_idx

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        kind = "cross" if encoder_hidden_states is not None else "self"
        kv = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        B = hidden_states.shape[0]
        H = attn.heads

        q = attn.head_to_batch_dim(attn.to_q(hidden_states))   # [B*H, Q, d]
        k = attn.head_to_batch_dim(attn.to_k(kv))
        v = attn.head_to_batch_dim(attn.to_v(kv))

        scores = torch.bmm(q, k.transpose(-1, -2)) * attn.scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = scores.softmax(dim=-1)                           # [B*H, Q, K]

        probs_bhqk = probs.view(B, H, probs.shape[-2], probs.shape[-1])
        probs_bhqk.retain_grad()
        self._store.add(probs_bhqk, kind, self._layer_idx)

        # Route through probs_bhqk so the computation graph passes through it
        # and retain_grad() actually captures a gradient on backward()
        probs_flat = probs_bhqk.view(B * H, probs_bhqk.shape[-2], probs_bhqk.shape[-1])
        out = attn.batch_to_head_dim(torch.bmm(probs_flat, v))
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


def install_attn_processors(dit_model, store: AttentionStore) -> dict[int, Any]:
    """Swap all DiT block processors; return restore map."""
    restore_map: dict[int, Any] = {}
    for idx, block in enumerate(dit_model.transformer_blocks):
        restore_map[idx] = block.attn1.processor
        block.attn1.set_processor(CaptureAttnProcessor(store, layer_idx=idx))
    return restore_map


def restore_attn_processors(dit_model, restore_map: dict[int, Any]) -> None:
    """Restore original processors — always called in finally block."""
    for idx, old_proc in restore_map.items():
        dit_model.transformer_blocks[idx].attn1.set_processor(old_proc)
