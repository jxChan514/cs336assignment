"""Regression checks for sampled actions that do not survive text round-trips."""
from types import SimpleNamespace

import pytest
import torch

from cs336_alignment.grpo import grpo_train_step, pack_prompt_and_response_ids


class NoReencodingTokenizer:
    pad_token_id = 0

    def __call__(self, *args, **kwargs):
        raise AssertionError("Sampled trajectories must not be re-encoded from text")


class ConstantLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0., .2, -.3, .4, .1]))

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


def test_trajectory_labels_preserve_repeated_tokens_and_eos():
    packed = pack_prompt_and_response_ids([[4, 2], [4]], [[1, 1], [0]], 0)
    assert packed["input_ids"].tolist() == [[4, 2, 1], [4, 0, 0]]
    assert packed["labels"].tolist() == [[2, 1, 1], [0, 0, 0]]
    assert packed["response_mask"].tolist() == [[False, True, True], [True, False, False]]


@pytest.mark.parametrize("method", ["grpo", "rft"])
def test_sampled_update_matches_actual_action_objective(method):
    model = ConstantLogitModel()
    initial = model.logits.detach().clone().requires_grad_(True)
    log_probs = initial.log_softmax(0)
    if method == "grpo":
        advantage = .5 / ((.5 ** .5) + 1e-6)
        # Repeated newline ID 1 occurs twice, but is averaged as one sequence.
        expected_loss = (advantage * log_probs[1] - advantage * log_probs[2]) / 2
        kwargs = {}
    else:
        # Only the correct answer's actual ID contributes, divided by fixed Z.
        expected_loss = -log_probs[2] / 8
        kwargs = dict(baseline="none", advantage_normalizer="none",
                      loss_normalization="constant", normalization_constant=8)
    expected_loss.backward()
    expected_parameters = initial.detach() - .1 * initial.grad

    def reward(response, truth):
        value = float(response == "correct")
        return {"reward": value, "format_reward": 1., "answer_reward": value}

    loss, metadata = grpo_train_step(
        model, NoReencodingTokenizer(), torch.optim.SGD(model.parameters(), lr=.1),
        2, None, reward, ["prompt", "prompt"], ["\n\n", "correct"], ["1", "1"], 2,
        prompt_token_ids=[[4], [4]], response_token_ids=[[1, 1], [2]], **kwargs,
    )
    torch.testing.assert_close(loss, expected_loss.detach())
    torch.testing.assert_close(model.logits.detach(), expected_parameters)
    assert metadata["sampled_token_ids_used"]
    assert metadata["trajectory_response_token_count"] == 3


def test_partial_token_trajectory_is_rejected_before_update():
    model = ConstantLogitModel()
    original = model.logits.detach().clone()
    with pytest.raises(ValueError, match="Supply both"):
        grpo_train_step(
            model, NoReencodingTokenizer(), torch.optim.SGD(model.parameters(), lr=.1),
            1, None, lambda response, truth: {"reward": 1., "format_reward": 1., "answer_reward": 1.},
            ["prompt", "prompt"], ["a", "b"], ["1", "1"], 2,
            response_token_ids=[[1], [2]],
        )
    assert torch.equal(model.logits.detach(), original)
