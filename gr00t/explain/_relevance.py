from __future__ import annotations

import numpy as np
import torch

from gr00t.explain._attention_store import LayerCapture


def compute_abar(probs: torch.Tensor, grad_probs: torch.Tensor) -> torch.Tensor:
    """[B,H,Q,K] → [B,Q,K]  Chefer A_bar = mean_heads(ReLU(grad * A))"""
    return (grad_probs * probs).clamp(min=0.0).mean(dim=1)


def rollout_relevance(
    captures: list[LayerCapture],
    n_sa: int,
    n_vl: int,
    action0_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Traverse layers top→bottom. Seed = first action token.
    Returns r_sa [B, n_sa], r_vl [B, n_vl].
    """
    B = captures[0].attention_probs.shape[0]
    device = captures[0].attention_probs.device
    r_sa = torch.zeros(B, n_sa, device=device)
    r_sa[:, action0_idx] = 1.0
    r_vl = torch.zeros(B, n_vl, device=device)

    for cap in sorted(captures, key=lambda c: -c.layer_idx):
        if cap.attention_probs.grad is None:
            continue
        abar = compute_abar(cap.attention_probs.detach(), cap.attention_probs.grad.detach())
        if cap.kind == "self":
            eye = torch.eye(n_sa, device=device).unsqueeze(0)
            M = eye + abar
            M = M / M.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            r_sa = torch.bmm(r_sa.unsqueeze(1), M).squeeze(1)
        else:                                    # cross: abar [B, n_sa, n_vl]
            r_vl = r_vl + torch.bmm(r_sa.unsqueeze(1), abar).squeeze(1)

    return r_sa, r_vl


def eagle_input_attribution(
    leaf: np.ndarray,
    leaf_grad: np.ndarray,
    r_vl: torch.Tensor,
) -> torch.Tensor:
    """
    Gradient × Input at EAGLE LLM input embeddings, normalised to preserve r_vl budget.

    Computes ReLU(grad ⊙ input).sum(embed_dim) per sequence position, then rescales so
    that r_input.sum(dim=-1) == r_vl.sum(dim=-1) exactly.  This is the proprioception
    attribution strategy applied at the pre-EAGLE-mixing level.

    Args:
        leaf:      [L, D] or [B, L, D] — LLM input embeddings (float32 numpy)
        leaf_grad: [L, D] or [B, L, D] — gradient w.r.t. leaf (float32 numpy)
        r_vl:      [B, L] — Chefer budget tensor to preserve

    Returns:
        [B, L] — attribution at LLM input positions, budget-normalised to r_vl.sum()
    """
    leaf_t = torch.from_numpy(leaf).to(r_vl.device)
    grad_t = torch.from_numpy(leaf_grad).to(r_vl.device)
    if leaf_t.dim() == 2:
        leaf_t = leaf_t.unsqueeze(0)
        grad_t = grad_t.unsqueeze(0)
    r_input = (grad_t * leaf_t).clamp(min=0).sum(dim=-1)   # [B, L]
    budget = r_vl.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = r_input.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return r_input * budget / scale


def map_vl_tokens_to_heatmaps(
    r_vl: torch.Tensor,
    eagle_input_ids: torch.Tensor,
    image_token_index: int,
    num_image_tokens: int,
    T: int,
    V: int,
) -> torch.Tensor:
    """r_vl [B, L] → heatmaps [B, T, V, patch_h, patch_w]"""
    B = r_vl.shape[0]
    patch_side = int(num_image_tokens ** 0.5)
    assert patch_side * patch_side == num_image_tokens
    heatmaps = torch.zeros(B, T, V, patch_side, patch_side, device=r_vl.device)
    for b in range(B):
        img_pos = (eagle_input_ids[b] == image_token_index).nonzero(as_tuple=False).squeeze(-1)
        assert img_pos.numel() == T * V * num_image_tokens
        heatmaps[b] = r_vl[b][img_pos.view(T, V, num_image_tokens)].view(T, V, patch_side, patch_side)
    return heatmaps
