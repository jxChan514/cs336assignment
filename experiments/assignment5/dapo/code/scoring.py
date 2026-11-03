"""OLMo2 response-only projection; the backbone still sees the whole prefix."""
import torch
from torch.utils.checkpoint import checkpoint


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
