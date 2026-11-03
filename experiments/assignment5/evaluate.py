"""Evaluate a pretrained or exported model with the fixed 3-shot prompt."""
import argparse
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=1024)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prompt", type=Path, default=root / "prompt.prompt")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from sampler import Sampler
    from utils import data_rows, stats, write_json
    cfg = json.loads((root / "configs/rl.json").read_text())
    rows = data_rows(args.val_file, args.examples, cfg["data_subset_seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampler = Sampler(args.model, args.output, cfg, args.gpu)
    try:
        sampler.start()
        records = sampler.generate(rows, tokenizer, args.prompt.read_text(), 1, cfg["eval_seed"])
        result = stats(records)
        write_json(args.output / "result.json", result)
        print(json.dumps(result))
    finally:
        sampler.stop()


if __name__ == "__main__":
    main()
