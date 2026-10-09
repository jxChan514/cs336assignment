from pathlib import Path
import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.generation import generate

project_root = Path(__file__).resolve().parent.parent

tokenizer = Tokenizer.from_files(
    vocab_filepath=project_root / "data" / "vocab.json",
    merges_filepath=project_root / "data" / "merges.json",
    special_tokens=["<|endoftext|>"],
)
print(len(tokenizer.vocab))                     # 应为 765
print(tokenizer.encode("Once upon a time"))     # 应为 [419, 421, 258, 423]


model=TransformerLM(vocab_size=len(tokenizer.vocab),
                    context_length=32,
                    d_model=64,
                    d_ff=192,
                    num_layers=2,
                    num_heads=4,
                    rope_theta=10000,
                    device="cpu",
                    dtype=torch.float32)

checkpoint_path = project_root / "checkpoints" / "debug" / "final.pt"
checkpoint = torch.load(
    checkpoint_path, map_location="cpu", weights_only=True
)
model.load_state_dict(checkpoint["model"])

prompt="once upon a time."
token_id=tokenizer.encode(prompt)
input_id=torch.tensor([token_id],dtype=torch.long)
eos_token_id = tokenizer.token_to_id[b"<|endoftext|>"]
generated_ids=generate(model=model,
                      input_ids=input_id,
                      max_new_tokens=50,
                      context_length=32,
                      temperature=1.0,
                      top_p=0.9,
                      eos_token_id=eos_token_id)
output_ids = generated_ids[0].tolist()
output_text = tokenizer.decode(output_ids)
