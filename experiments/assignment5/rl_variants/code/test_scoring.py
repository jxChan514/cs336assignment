"""Compare response scores and parameter gradients to the original full head."""
import copy
import torch
from transformers import Olmo2Config, Olmo2ForCausalLM
from reference.grpo import get_response_log_probs
from scoring import response_log_probs


def main():
    torch.manual_seed(12)
    config = Olmo2Config(vocab_size=43, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                        max_position_embeddings=64, attention_dropout=0., tie_word_embeddings=False)
    reference = Olmo2ForCausalLM(config).float().train()
    reference.config.use_cache = False
    fast = copy.deepcopy(reference)
    ids = torch.randint(0, 43, (4, 17))
    labels = torch.randint(0, 43, ids.shape)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[0, 8:12] = True
    mask[1, 4:16] = True
    mask[2, 9:17] = True
    mask[3, 5:10] = True
    weights = torch.tensor([-.5, .5, 0., 1.])[:, None]
    expected = get_response_log_probs(reference, ids, labels, True)
    actual = response_log_probs(fast, ids, labels, mask, True, head_chunk_size=7)
    for name in ["log_probs", "token_entropy"]:
        torch.testing.assert_close(actual[name][mask], expected[name][mask], atol=2e-6, rtol=1e-5)
        assert (actual[name][~mask] == 0).all()
    assert not actual["token_entropy"].requires_grad
    (expected["log_probs"] * weights * mask).sum().backward()
    (actual["log_probs"] * weights * mask).sum().backward()
    max_difference = 0.
    for (name, p), (other, q) in zip(reference.named_parameters(), fast.named_parameters()):
        assert name == other
        torch.testing.assert_close(q.grad, p.grad, atol=4e-6, rtol=2e-4, msg=lambda msg: name + ": " + msg)
        max_difference = max(max_difference, float((p.grad - q.grad).abs().max()))
    with torch.no_grad():
        old = response_log_probs(fast, ids, labels, mask)
        assert not old["log_probs"].requires_grad
    print("A5_SCORING_EQUIVALENCE_PASS", {"max_parameter_gradient_difference": max_difference}, flush=True)


if __name__ == "__main__":
    main()
