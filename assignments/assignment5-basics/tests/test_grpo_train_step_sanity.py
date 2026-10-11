"""Training-step integration checks beyond the fixed assignment snapshot."""

import copy
from unittest.mock import patch

import pytest
import torch

from cs336_alignment.grpo import grpo_train_step
from .test_grpo import _train_step_inputs, _train_step_reward_fn


def _update(model, tokenizer, optimizer, accumulation, max_grad_norm, reward_fn):
    prompts, responses, ground_truths, _ = _train_step_inputs(tokenizer)
    return grpo_train_step(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        gradient_accumulation_steps=accumulation,
        max_grad_norm=max_grad_norm,
        reward_fn=reward_fn,
        repeated_prompts=prompts,
        rollout_responses=responses,
        repeated_ground_truths=ground_truths,
        group_size=2,
    )


def test_train_step_uneven_accumulation_matches_full_batch(tiny_train_model, tokenizer):
    full_model = copy.deepcopy(tiny_train_model)
    split_model = copy.deepcopy(tiny_train_model)
    full_loss, full_metadata = _update(
        full_model, tokenizer, torch.optim.SGD(full_model.parameters(), lr=0.01),
        1, None, _train_step_reward_fn,
    )
    # Four answers split into three microbatches of sizes 1, 1, and 2.
    split_loss, split_metadata = _update(
        split_model, tokenizer, torch.optim.SGD(split_model.parameters(), lr=0.01),
        3, None, _train_step_reward_fn,
    )
    torch.testing.assert_close(full_loss, split_loss, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(full_metadata["token_entropy"], split_metadata["token_entropy"])
    for full_param, split_param in zip(full_model.parameters(), split_model.parameters()):
        torch.testing.assert_close(full_param, split_param, atol=1e-7, rtol=1e-5)
        assert full_param.grad is None and split_param.grad is None


def test_train_step_clips_and_updates_once(tiny_train_model, tokenizer):
    model = tiny_train_model
    before = [p.detach().clone() for p in model.parameters()]
    lr, max_norm = 0.1, 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    with patch.object(optimizer, "step", wraps=optimizer.step) as optimizer_step:
        loss, metadata = _update(model, tokenizer, optimizer, 4, max_norm, _train_step_reward_fn)
    optimizer_step.assert_called_once()
    update_norm = torch.cat([
        (param.detach() - old).flatten() for param, old in zip(model.parameters(), before)
    ]).norm()
    assert 0 < update_norm.item() <= lr * max_norm * 1.001
    assert metadata["grad_norm"].item() > max_norm
    assert torch.isfinite(loss) and not loss.requires_grad
    assert all(p.grad is None for p in model.parameters())
    assert metadata["reward_mean"] == pytest.approx(0.375)
    assert metadata["format_reward_mean"] == pytest.approx(0.5)
    assert torch.isfinite(metadata["token_entropy"])


def test_train_step_equal_rewards_give_zero_update(tiny_train_model, tokenizer):
    model = tiny_train_model
    before = [p.detach().clone() for p in model.parameters()]

    def zero_reward(response, ground_truth):
        return {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}

    loss, metadata = _update(
        model, tokenizer, torch.optim.SGD(model.parameters(), lr=0.01),
        2, None, zero_reward,
    )
    assert loss.item() == 0.0 and metadata["grad_norm"].item() == 0.0
    for param, old in zip(model.parameters(), before):
        assert torch.equal(param.detach(), old) and param.grad is None
