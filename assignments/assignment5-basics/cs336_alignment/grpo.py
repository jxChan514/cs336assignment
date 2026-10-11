"""A5 GRPO implementation interfaces; fill in one component at a time."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Literal

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """将配对的提示和回答转为训练输入、下一词元标签及回答掩码。

    当前学习任务：在这个函数中完成自己的实现。
    prompt_strs 和 output_strs 等长，第 i 个提示与第 i 个回答配对；
    output_strs 是模型生成的回答，不是标准答案。
    分别分词时不添加特殊词元，拼接后用 tokenizer.pad_token_id 右侧补齐。
    input_ids 和 labels 使用 torch.long，response_mask 使用 torch.bool。
    掩码与 labels 对齐，仅回答词元对应的位置为 True。

    Args:
        prompt_strs: list[str]
            List of prompt strings.
        output_strs: list[str]
            List of output strings.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.

Returns:
        dict[str, torch.Tensor].
            Let prompt_and_output_lens be a list containing the lengths of the
            concatenated tokenized prompt and output strings. Then the returned
            dictionary should have the following keys:

            input_ids
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): the tokenized
                prompt and output strings, with the final token sliced off.
            labels
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): shifted input
                ids, i.e., the input ids without the first token.
            response_mask
                torch.Tensor of shape
                (batch_size, max(prompt_and_output_lens) - 1): a mask aligned
                with labels, with value 1 where the corresponding label token
                is part of the response and 0 otherwise.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs 和 output_strs 必须等长")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 必须设置 pad_token_id")

    batch_size = len(prompt_strs)
    all_concat_ids = []
    all_response_pos = []  # 回答在完整序列中的起始位置
    for p_str, o_str in zip(prompt_strs, output_strs):
        p_id = tokenizer(p_str, add_special_tokens=False)["input_ids"]
        o_id = tokenizer(o_str, add_special_tokens=False)["input_ids"]
        all_concat_ids.append(p_id + o_id)
        all_response_pos.append(len(p_id))

    max_length = max((len(ids) for ids in all_concat_ids), default=0)
    padded_ids = torch.full(
        (batch_size, max_length), tokenizer.pad_token_id, dtype=torch.long
    )
    full_response_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    for i, (ids, response_start) in enumerate(zip(all_concat_ids, all_response_pos)):
        sequence_length = len(ids)
        padded_ids[i, :sequence_length] = torch.tensor(ids, dtype=torch.long)
        full_response_mask[i, response_start:sequence_length] = True

    # 每列预测下一个词元，回答掩码采用与 labels 相同的切片，去掉第一列。
    return {
        "input_ids": padded_ids[:, :-1],
        "labels": padded_ids[:, 1:],
        "response_mask": full_response_mask[:, 1:],
    }


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token conditional log-probabilities (given the previous tokens)
    from a causal language model, and optionally the entropy of the model's
    next-token distribution.

    Args:
        model: PreTrainedModel
            HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor
            shape (batch_size, sequence_length), concatenated prompt + response
            tokens as produced by your tokenization method.
        labels: torch.Tensor
            shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool
            If True, also return per-token entropy.

    Returns:
        dict[str, torch.Tensor].
            "log_probs"
                shape (batch_size, sequence_length), conditional
                log-probabilities log p_(theta)(x_t | x_(<t)).
            "token_entropy"
                optional, shape (batch_size, sequence_length), per-token
                entropy for each position (present only if
                return_token_entropy=True).
    """
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input_ids 和 labels 必须具有相同的二维形状 (B, T)")

    logits = model(input_ids).logits  # (B, T, V)
    # 半精度模型的概率计算使用 float32，避免小概率下溢；保留梯度。
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    vocab_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs = vocab_log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    result = {"log_probs": log_probs}

    if return_token_entropy:
        probs = vocab_log_probs.exp()
        result["token_entropy"] = -(probs * vocab_log_probs).sum(dim=-1)

    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for a list of rollout responses, along with metadata for
    the reward components.

    Args:
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            raw_rewards
                shape (rollout_batch_size,). Unnormalized rewards for each
                rollout response.
            metadata
                Reward statistics to log. At minimum, include the mean total
                and format rewards over the rollout batch.
    """
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses 和 repeated_ground_truths 必须等长")

    scores = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(
        [float(score["reward"]) for score in scores], dtype=torch.float32
    )
    num_responses = len(scores)
    metadata = {
        f"{key}_mean": (
            sum(float(score[key]) for score in scores) / num_responses
            if num_responses else 0.0
        )
        for key in ("reward", "format_reward", "answer_reward")
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute advantages by applying the requested baseline and normalization
    within each group.

    Args:
        raw_rewards: torch.Tensor
            shape (rollout_batch_size,). Unnormalized rewards for each rollout
            response, where rollout_batch_size = n_prompts_per_rollout_batch *
            group_size.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            mean subtracts the per-group mean reward; none leaves rewards
            unchanged before normalization.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            std divides by the per-group sample standard deviation plus eps;
            mean divides by the per-group mean reward plus eps; none applies
            no normalization.

    Returns:
        tuple[torch.Tensor, dict[str, float]].
            advantages
                shape (rollout_batch_size,). Group-normalized rewards for each
                rollout response.
            metadata
                your choice of other statistics to log (e.g. mean, std, max/min
                of rewards).
    """
    if baseline not in ("mean", "none"):
        raise ValueError("baseline 必须为 'mean' 或 'none'")
    if advantage_normalizer not in ("std", "none", "mean"):
        raise ValueError("advantage_normalizer 必须为 'std'、'none' 或 'mean'")
    if raw_rewards.ndim != 1 or not raw_rewards.is_floating_point():
        raise ValueError("raw_rewards 必须是一维浮点张量")
    if not isinstance(group_size, int) or group_size < 1:
        raise ValueError("group_size 必须为正整数")
    if advantage_normalizer == "std" and group_size < 2:
        raise ValueError("标准 GRPO 使用样本标准差，group_size 必须至少为 2")
    if raw_rewards.numel() % group_size != 0:
        raise ValueError("奖励数量必须能被 group_size 整除")
    if advantage_eps <= 0:
        raise ValueError("advantage_eps 必须大于 0")

    # 奖励与 advantage 作为策略梯度的固定权重，不对评分过程求导。
    rewards = raw_rewards.detach()
    if rewards.numel() == 0:
        return rewards.clone(), {
            "reward_mean": 0.0,
            "reward_min": 0.0,
            "reward_max": 0.0,
            "group_std_mean": 0.0,
            "zero_std_group_fraction": 0.0,
        }

    grouped_rewards = rewards.reshape(-1, group_size)
    group_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    # std 归一化沿用作业的样本标准差；单样本组仅在非 std 模式下允许。
    group_std = grouped_rewards.std(
        dim=-1, keepdim=True, correction=1 if group_size > 1 else 0
    )
    advantages = grouped_rewards - group_mean if baseline == "mean" else grouped_rewards
    if advantage_normalizer == "std":
        advantages = advantages / (group_std + advantage_eps)
    elif advantage_normalizer == "mean":
        advantages = advantages / (group_mean + advantage_eps)
    metadata = {
        "reward_mean": rewards.mean().item(),
        "reward_min": rewards.min().item(),
        "reward_max": rewards.max().item(),
        "group_std_mean": group_std.mean().item(),
        "zero_std_group_fraction": (group_std == 0).float().mean().item(),
    }
    return advantages.reshape(-1), metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy-gradient loss at every token, where
    raw_rewards_or_advantages is either the raw reward or an
    already-normalized advantage.

    Args:
        raw_rewards_or_advantages: torch.Tensor
            Shape (batch_size,) or (batch_size, 1), scalar reward/advantage for
            each rollout response.
        policy_log_probs: torch.Tensor
            Shape (batch_size, sequence_length), logprobs for each token.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style
            token-level reweighting and clipping; "gspo": do GSPO-style
            sequence-level reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        response_mask: torch.Tensor | None = None
            Optional shape (batch_size, sequence_length) mask over response
            tokens. Required for GSPO implementations that average the
            sequence-level log-ratio over response tokens only.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            per_token_policy_gradient_loss
                Shape (batch_size, sequence_length), the per-token
                policy-gradient loss (to be aggregated across the batch and
                sequence dimensions in the training loop).
            metadata
                Statistics from the underlying loss call, such as
                clip-fraction components.
    """
    if importance_reweighting_method != "none":
        raise NotImplementedError("当前阶段仅支持 on-policy：importance_reweighting_method='none'")
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs 必须是二维张量 (B, T)")
    batch_size = policy_log_probs.shape[0]
    if raw_rewards_or_advantages.shape == (batch_size,):
        weights = raw_rewards_or_advantages.unsqueeze(-1)
    elif raw_rewards_or_advantages.shape == (batch_size, 1):
        weights = raw_rewards_or_advantages
    else:
        raise ValueError("奖励或 advantage 的形状必须为 (B,) 或 (B, 1)")

    # 固定奖励权重；同一回答的所有 token 共用一个权重。
    weights = weights.detach().to(device=policy_log_probs.device)
    per_token_loss = -weights * policy_log_probs
    # 回答掩码与序列长度归一化交给后续的聚合函数处理。
    return per_token_loss, {}


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate the per-token policy-gradient loss according to the response
    mask and loss-normalization strategy.

    Args:
        per_token_policy_gradient_loss: torch.Tensor
            Shape (batch_size, sequence_length), the per-token policy-gradient
            loss (to be aggregated across the batch and sequence dimensions in
            the training loop).
        mask
            torch.Tensor of shape (batch_size, sequence_length) denoting which
            positions should be included in the loss.
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant.
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        loss: torch.Tensor
            A scalar containing the average loss. Make sure you can later call
            backward on this loss.
    """
    if loss_normalization != "sequence":
        raise NotImplementedError("当前阶段仅支持 loss_normalization='sequence'")
    if per_token_policy_gradient_loss.ndim != 2 or mask.shape != per_token_policy_gradient_loss.shape:
        raise ValueError("token loss 和 mask 必须具有相同的二维形状 (B, T)")

    valid_mask = mask.to(device=per_token_policy_gradient_loss.device, dtype=torch.bool)
    # 使用 where 排除无效位置，避免 padding 位置的 NaN/Inf 污染总 loss。
    masked_loss = torch.where(valid_mask, per_token_policy_gradient_loss, 0.0)
    if masked_loss.shape[0] == 0:
        return masked_loss.sum()

    # 半精度时用 float32 累加；有效长度为 0 的回答贡献零 loss。
    sum_dtype = (
        torch.float32
        if masked_loss.dtype in (torch.float16, torch.bfloat16)
        else masked_loss.dtype
    )
    per_sequence_sum = masked_loss.sum(dim=-1, dtype=sum_dtype)
    response_lengths = valid_mask.sum(dim=-1).clamp_min(1)
    per_sequence_mean = per_sequence_sum / response_lengths
    return per_sequence_mean.mean()


def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Execute forward-and-backward passes, with gradient_accumulation_steps
    microbatches.

    Args:
        model: PreTrainedModel
            HuggingFace model to train.
        tokenizer: PreTrainedTokenizer
            Tokenizer to use for tokenization.
        optimizer: Optimizer
            Optimizer for the model.
        gradient_accumulation_steps: int
            Number of microbatches per optimizer step.
        max_grad_norm: float | None
            If not None, clip the gradient norm to this value before calling
            optimizer.step().
        reward_fn: Callable[[str, str], dict[str, float]]
            Scores the rollout responses against the ground truths, producing
            a dict with keys "reward", "format_reward", and "answer_reward".
        repeated_prompts: list[str]
            The prompts for the examples. The length of this list is
            rollout_batch_size, because the prompt for each example is repeated
            group_size times.
        rollout_responses: list[str]
            Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: list[str]
            The ground truths for the examples. The length of this list is
            rollout_batch_size, because the ground truth for each example is
            repeated group_size times.
        group_size: int
            Number of responses per question (group).
        baseline: Literal["mean", "none"]
            If mean, subtract the per-group mean reward; if none, do nothing.
        advantage_eps: float
            Small constant to avoid division by zero in normalization.
        advantage_normalizer: Literal["std", "none", "mean"]
            If std, divide by the per-group standard deviation; if none, do
            nothing; if mean, divide by the per-group mean reward.
        importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"]
            "none": no importance reweighting; "noclip": apply importance
            reweighting without clipping; "grpo": do PPO/GRPO-style token-level
            reweighting and clipping; "gspo": do GSPO-style sequence-level
            reweighting and clipping.
        old_log_probs: torch.Tensor | None
            Required unless importance_reweighting_method = "none"; shape
            (batch_size, sequence_length).
        cliprange: float | None = None
            Clip parameter epsilon, required when importance_reweighting_method
            is "grpo" or "gspo".
        loss_normalization: Literal["sequence", "constant"] = "sequence"
            "sequence": average loss over each sequence, then average over
            sequences; "constant": normalize total loss by a constant (fixed
            for all of training).
        normalization_constant: int | None = None
            The constant to divide total loss by; required if
            loss_normalization = "constant".

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss
                scalar tensor. The batch loss, adjusted for gradient
                accumulation. We return this so we can log it.
            metadata
                Dict with metadata from the underlying loss call, gradient norm
                before clipping, and any other statistics you might want to log.
    """
    if (
        baseline != "mean"
        or advantage_normalizer != "std"
        or importance_reweighting_method != "none"
        or loss_normalization != "sequence"
    ):
        raise NotImplementedError("当前阶段仅支持标准 on-policy GRPO（mean/std/none/sequence）")
    batch_size = len(rollout_responses)
    if not (len(repeated_prompts) == len(repeated_ground_truths) == batch_size):
        raise ValueError("提示、回答和标准答案的列表必须等长")
    if batch_size == 0:
        raise ValueError("训练 batch 不能为空")
    if (
        not isinstance(gradient_accumulation_steps, int)
        or not 1 <= gradient_accumulation_steps <= batch_size
    ):
        raise ValueError("梯度累积步数必须介于 1 和 batch_size 之间")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm 必须大于 0 或为 None")

    device = next(model.parameters()).device
    # 先在完整 rollout batch 上计算 advantage，再拆 microbatch。
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )
    advantages, group_metadata = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )
    train_batch = tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    advantages = advantages.to(device)

    model.train()
    # DDP 保留 bucket 的梯度视图，避免下一轮累积时额外分配一整份梯度。
    set_grad_to_none = not isinstance(model, torch.nn.parallel.DistributedDataParallel)
    optimizer.zero_grad(set_to_none=set_grad_to_none)
    batch_loss = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    response_token_count = torch.zeros((), device=device, dtype=torch.long)

    for microbatch_index in range(gradient_accumulation_steps):
        # 允许不能整除的 batch，按样本数量加权以保持完整 batch 的平均梯度。
        start = microbatch_index * batch_size // gradient_accumulation_steps
        end = (microbatch_index + 1) * batch_size // gradient_accumulation_steps
        input_ids = train_batch["input_ids"][start:end].to(device)
        labels = train_batch["labels"][start:end].to(device)
        response_mask = train_batch["response_mask"][start:end].to(device)
        # DDP 只在最后一个 microbatch 同步累积梯度；上下文必须包含前向和反向。
        sync_context = (
            model.no_sync()
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            and microbatch_index + 1 < gradient_accumulation_steps
            else nullcontext()
        )
        with sync_context:
            model_outputs = get_response_log_probs(
                model, input_ids, labels, return_token_entropy=True
            )
            per_token_loss, _ = compute_policy_gradient_loss(
                advantages[start:end], model_outputs["log_probs"],
                importance_reweighting_method="none",
            )
            microbatch_loss = aggregate_loss_across_microbatch(
                per_token_loss, response_mask, loss_normalization="sequence"
            )
            scaled_loss = microbatch_loss * ((end - start) / batch_size)
            scaled_loss.backward()
        batch_loss = batch_loss + scaled_loss.detach()
        entropy_sum = entropy_sum + torch.where(
            response_mask, model_outputs["token_entropy"].detach(), 0.0
        ).sum()
        response_token_count = response_token_count + response_mask.sum()
        # 释放这一 microbatch 的前向结果，下一轮不保留其计算图。
        del model_outputs, per_token_loss, microbatch_loss, scaled_loss

    # 返回值是裁剪前的范数；None 时仅计算范数，不限制梯度大小。
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_grad_norm if max_grad_norm is not None else float("inf"),
        error_if_nonfinite=True,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=set_grad_to_none)

    metadata = {
        **reward_metadata,
        **group_metadata,
        "loss": batch_loss.detach(),
        "grad_norm": grad_norm.detach(),
        "token_entropy": (entropy_sum / response_token_count.clamp_min(1)).detach(),
        "response_token_count": response_token_count.detach(),
    }
    return batch_loss.detach(), metadata
