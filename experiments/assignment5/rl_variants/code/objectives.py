"""One-component A5 variants; denominators retain the unfiltered batch size."""
import torch
from losses import policy_loss


def advantages_for_method(rewards, group_size, settings, eps=1e-6):
    grouped = rewards.detach().float().reshape(-1, group_size)
    mean = grouped.mean(-1, keepdim=True)
    advantages = grouped - mean if settings["baseline"] == "mean" else grouped.clone()
    normalizer = settings["advantage_normalizer"]
    if normalizer == "std":
        advantages = advantages / (grouped.std(-1, keepdim=True) + eps)
    elif normalizer == "mean":
        advantages = advantages / (mean + eps)
    elif normalizer == "power of group mean":
        advantages = advantages / (mean + eps).pow(settings["advantage_alpha"])
    elif normalizer != "none":
        raise ValueError(normalizer)
    return advantages.reshape_as(rewards)


def loss_for_method(current, old, advantages, mask, settings, full_count, max_tokens=512):
    method = settings["name"]
    if method.startswith("offpolicy_"):
        return policy_loss(current, old, advantages, mask, method, settings.get("cliprange"))
    a = advantages.detach().to(current).reshape(-1, 1)
    token_loss = (-a * current).masked_fill(~mask.bool(), 0.0)
    if settings["loss_normalization"] == "sequence":
        loss = (token_loss.sum(-1) / mask.sum(-1).clamp_min(1)).mean()
    elif settings["loss_normalization"] == "constant":
        loss = token_loss.sum() / (full_count * max_tokens)
    else:
        raise ValueError(settings["loss_normalization"])
    return loss, {"clip_fraction": 0.0, "token_clip_fraction": 0.0,
                  "sequence_clip_fraction": 0.0, "ratio_mean": 1.0,
                  "ratio_min": 1.0, "ratio_max": 1.0,
                  "response_tokens": int(mask.sum())}
