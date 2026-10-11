"""Small integration checks using the assignment's real math reward functions."""

import pytest
import torch

from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.grpo import compute_rollout_rewards


def test_r1_rollout_rewards_real_grader():
    responses = [
        "3*4=12.</think> <answer>12</answer>",
        "3*4=7.</think> <answer>7</answer>",
        "<answer>12</answer>",
        "3*4=12.</think>\n<answer>12</answer>",
    ]
    rewards, metadata = compute_rollout_rewards(
        r1_zero_reward_fn, responses, ["12"] * len(responses)
    )
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert not rewards.requires_grad
    assert metadata == pytest.approx({
        "reward_mean": 0.25,
        "format_reward_mean": 0.5,
        "answer_reward_mean": 0.25,
    })
    print("R1 rewards:", rewards.tolist(), "metadata:", metadata)


def test_question_only_rollout_rewards_real_grader():
    responses = [r"The answer is \boxed{12}.", r"The answer is \boxed{7}.", "12"]
    rewards, metadata = compute_rollout_rewards(
        question_only_reward_fn, responses, ["12"] * len(responses)
    )
    torch.testing.assert_close(rewards, torch.tensor([1.0, 0.0, 0.0]))
    assert metadata == pytest.approx({
        "reward_mean": 1 / 3,
        "format_reward_mean": 2 / 3,
        "answer_reward_mean": 1 / 3,
    })
    print("Question-only rewards:", rewards.tolist(), "metadata:", metadata)
