"""The four A5 objectives, with sequence normalization and explicit clip units."""
from __future__ import annotations

import torch

METHODS = ("offpolicy_naive", "offpolicy_noclip", "offpolicy_clip", "offpolicy_gspo")


def policy_loss(current, old, advantages, response_mask, method, cliprange=None):
    if method not in METHODS:
        raise ValueError(method)
    if current.shape != old.shape or current.shape != response_mask.shape:
        raise ValueError("Current, old and mask shapes must match")
    mask = response_mask.bool()
    a = advantages.detach().float().reshape(-1, 1)
    if a.shape[0] != current.shape[0]:
        raise ValueError("One frozen advantage is required per response")
    lp = current.float()
    delta = torch.where(mask, lp - old.detach().float(), 0.0)
    if not torch.isfinite(delta[mask]).all():
        raise FloatingPointError("Nonfinite response log-probability ratio")
    ratio = delta.exp()
    if not torch.isfinite(ratio[mask]).all():
        raise FloatingPointError("Importance ratio overflow; do not silently clamp it")
    count = mask.sum(-1)
    length = count.clamp_min(1)
    seq_log_ratio = delta.sum(-1) / length
    seq_ratio = seq_log_ratio.exp()
    clipped = torch.zeros_like(mask)
    seq_clipped = torch.zeros_like(count, dtype=torch.bool)
    if method == "offpolicy_naive":
        token_loss = -a * lp
        per_sequence = torch.where(mask, token_loss, 0.0).sum(-1) / length
    elif method == "offpolicy_noclip":
        token_loss = -a * ratio
        per_sequence = torch.where(mask, token_loss, 0.0).sum(-1) / length
    elif method == "offpolicy_clip":
        if cliprange is None or not 0 < cliprange < 1:
            raise ValueError("Positive token cliprange below one is required")
        token_loss = -torch.minimum(a * ratio, a * ratio.clamp(1 - cliprange, 1 + cliprange))
        per_sequence = torch.where(mask, token_loss, 0.0).sum(-1) / length
        clipped = (((a > 0) & (ratio > 1 + cliprange)) |
                   ((a < 0) & (ratio < 1 - cliprange))) & mask
    else:
        if cliprange is None or not 0 < cliprange < 1:
            raise ValueError("Positive GSPO cliprange below one is required")
        seq_a = a.squeeze(-1)
        # No additional /T here: differentiating the mean log-ratio supplies it.
        per_sequence = -torch.minimum(seq_a * seq_ratio,
                                      seq_a * seq_ratio.clamp(1 - cliprange, 1 + cliprange))
        per_sequence = torch.where(count > 0, per_sequence, 0.0)
        seq_clipped = (((seq_a > 0) & (seq_ratio > 1 + cliprange)) |
                       ((seq_a < 0) & (seq_ratio < 1 - cliprange))) & (count > 0)
        clipped = seq_clipped.unsqueeze(-1) & mask
    loss = per_sequence.mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite policy loss")
    with torch.no_grad():
        valid_ratio = ratio[mask]
        denom = mask.sum().clamp_min(1)
        clip_fraction = (seq_clipped.float().mean() if method == "offpolicy_gspo"
                         else clipped.sum() / denom)
        stats = {
            "clip_fraction": float(clip_fraction),
            "token_clip_fraction": float(clipped.sum() / denom),
            "sequence_clip_fraction": float(seq_clipped.float().mean()),
            "ratio_mean": float(valid_ratio.mean()) if valid_ratio.numel() else 1.0,
            "ratio_min": float(valid_ratio.min()) if valid_ratio.numel() else 1.0,
            "ratio_max": float(valid_ratio.max()) if valid_ratio.numel() else 1.0,
            "sequence_ratio_mean": float(seq_ratio.mean()),
            "approx_kl_k3": float(torch.where(mask, ratio - 1 - delta, 0.0).sum() / denom),
            "response_tokens": int(mask.sum()),
        }
        if cliprange is not None:
            out = ((ratio < 1 - cliprange) | (ratio > 1 + cliprange)) & mask
            stats["ratio_out_of_range_fraction"] = float(out.sum() / denom)
    return loss, stats
