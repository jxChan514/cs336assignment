"""Actual sampled-token batching and OLMo2 response-only scoring."""
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

def pack_prompt_and_response_ids(
    prompt_token_ids: list[list[int]],
    response_token_ids: list[list[int]],
    pad_token_id: int,
) -> dict[str, Tensor]:
    """Pad actual token trajectories and align next-token labels and masks.

    Unlike the assignment's text-tokenization interface, this path never
    decodes or re-encodes sampled actions. EOS IDs remain valid response labels,
    even when the same ID is used for padding.
    """
    if len(prompt_token_ids) != len(response_token_ids):
        raise ValueError("Prompt and response token batches must have equal lengths")
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    for sequence in prompt_token_ids + response_token_ids:
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in sequence):
            raise ValueError("Token IDs must be nonnegative integers")
    batch_size = len(prompt_token_ids)
    all_concat_ids = [prompt + response for prompt, response in zip(prompt_token_ids, response_token_ids)]
    all_response_pos = [len(prompt) for prompt in prompt_token_ids]

    max_length = max((len(ids) for ids in all_concat_ids), default=0)
    padded_ids = torch.full(
        (batch_size, max_length), pad_token_id, dtype=torch.long
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

def response_log_probs(model, input_ids, labels, response_mask,
                       return_token_entropy=False, head_chunk_size=512):
    if model.config.model_type != "olmo2":
        raise ValueError("This scorer is verified for OLMo2 only")
    if labels.shape != input_ids.shape or response_mask.shape != labels.shape:
        raise ValueError("Input, shifted labels, and response mask must align")
    if head_chunk_size < 1:
        raise ValueError("head_chunk_size must be positive")
    mask = response_mask.bool()
    # This is the same backbone and lm_head used by Olmo2ForCausalLM.forward.
    # Prompt/padding positions contribute no loss or reported entropy, so their
    # vocabulary projections can be omitted without changing the objective.
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    selected_hidden = hidden[mask]
    selected_labels = labels[mask]
    def project(features, targets):
        logits = model.lm_head(features).float()
        vocab_log_probs = logits.log_softmax(-1)
        log_probs = vocab_log_probs.gather(-1, targets[:, None]).squeeze(-1)
        if not return_token_entropy:
            return log_probs
        # Entropy is diagnostic, not a regularizer. Do not retain its graph.
        with torch.no_grad():
            entropy = -(vocab_log_probs.exp() * vocab_log_probs).sum(-1)
        return log_probs, entropy
    parts, entropies = [], []
    for start in range(0, len(selected_labels), head_chunk_size):
        features = selected_hidden[start:start + head_chunk_size]
        targets = selected_labels[start:start + head_chunk_size]
        out = checkpoint(project, features, targets, use_reentrant=False,
                         preserve_rng_state=False) if torch.is_grad_enabled() else project(features, targets)
        if return_token_entropy:
            lp, entropy = out
            entropies.append(entropy)
        else:
            lp = out
        parts.append(lp)
    if not parts:
        raise ValueError("The scoring batch contains no response tokens")
    # Invalid positions are explicitly zero; every consumer uses response_mask.
    result = {"log_probs": hidden.new_zeros(labels.shape, dtype=torch.float32).masked_scatter(mask, torch.cat(parts))}
    if return_token_entropy:
        result["token_entropy"] = torch.zeros_like(result["log_probs"]).masked_scatter(mask, torch.cat(entropies))
    return result
