"""DAPO components: asymmetric clipping, raw-accuracy filtering and token reduction."""
import torch


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
