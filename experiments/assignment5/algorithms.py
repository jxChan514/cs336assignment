"""Policy-gradient estimators used in the RL and DAPO experiments."""
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


def shaped_rewards(task_rewards, lengths, max_tokens=512, cache_tokens=128):
    task = task_rewards.detach().float()
    lengths = torch.as_tensor(lengths, device=task.device, dtype=torch.float32)
    penalty = -((lengths - (max_tokens-cache_tokens))/cache_tokens).clamp(0, 1)
    return 2*task-1+penalty, penalty


def normalized_advantages(rewards, group_size=8, eps=1e-6):
    grouped = rewards.detach().float().reshape(-1, group_size)
    result = (grouped-grouped.mean(-1, keepdim=True))/(grouped.std(-1, keepdim=True)+eps)
    return result.reshape_as(rewards)


def select_mixed_groups(records, group_size=8):
    if len(records) % group_size:
        raise ValueError('Incomplete prompt group')
    accepted = []
    counts = {'all_wrong': 0, 'all_correct': 0, 'mixed': 0}
    for start in range(0, len(records), group_size):
        group = records[start:start+group_size]
        if len({r['id'] for r in group}) != 1 or [r['sample_index'] for r in group] != list(range(group_size)):
            raise ValueError('Prompt groups/sample indices are misaligned')
        good = sum(r['rewards']['reward'] for r in group)
        kind = 'all_wrong' if good == 0 else 'all_correct' if good == group_size else 'mixed'
        counts[kind] += 1
        if kind == 'mixed':
            accepted.extend(group)
    return accepted, counts


def token_loss(current, old, advantages, mask, denominator, low=.2, high=.28):
    if current.shape != old.shape or current.shape != mask.shape:
        raise ValueError('Current, frozen old scores and mask must align')
    if denominator < 1:
        raise ValueError('The optimizer batch must contain response tokens')
    mask = mask.bool()
    delta = torch.where(mask, current.float()-old.detach().float(), 0.)
    ratio = delta.exp()
    if not torch.isfinite(ratio[mask]).all():
        raise FloatingPointError('Importance ratio overflow/nonfinite')
    a = advantages.detach().to(current).reshape(-1, 1)
    surrogate = torch.minimum(a*ratio, a*ratio.clamp(1-low, 1+high))
    loss = -surrogate.masked_fill(~mask, 0.).sum()/denominator
    active_clip = (((a > 0) & (ratio > 1+high)) | ((a < 0) & (ratio < 1-low))) & mask
    outside = ((ratio < 1-low) | (ratio > 1+high)) & mask
    with torch.no_grad():
        valid = ratio[mask].detach().cpu()
        meta = {'tokens': int(mask.sum()), 'clipped_tokens': int(active_clip.sum()),
                'outside_tokens': int(outside.sum()),
                'kl_k3_sum': float(((ratio-1-delta).masked_fill(~mask, 0)).sum()),
                'ratio_values': valid}
    return loss, meta


def warmup_lr(update_number, base_lr=1e-6, warmup_updates=80):
    return base_lr*min(update_number/max(warmup_updates, 1), 1.)
