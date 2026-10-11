"""Run the A5 prompting baselines with the assignment's prompts and graders."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ASSIGNMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASSIGNMENT))
PROMPTS = ("question_only", "r1_zero", "r1_zero_three_shot_gsm8k")


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def category(rewards):
    pair = (rewards["format_reward"], rewards["answer_reward"])
    return {(1.0, 1.0): 1, (1.0, 0.0): 2, (0.0, 0.0): 3}[pair]


def summarize(records):
    counts = Counter(row["category"] for row in records)
    n = len(records)
    return {
        "n": n,
        "category_counts": {str(i): counts[i] for i in (1, 2, 3)},
        "accuracy": sum(row["rewards"]["answer_reward"] for row in records) / n,
        "format_rate": sum(row["rewards"]["format_reward"] for row in records) / n,
        "mean_output_tokens": sum(len(row["token_ids"]) for row in records) / n,
        "finish_reasons": dict(Counter(row["finish_reason"] for row in records)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("CS336_A5_MODEL_DIR", str(ASSIGNMENT / "models/OLMo-2-0425-1B"))))
    data = Path(os.environ.get("CS336_A5_DATA_DIR", str(ASSIGNMENT / "data/gsm8k")))
    parser.add_argument("--test-file", type=Path, default=data / "test.jsonl")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--port", type=int, default=18005)
    parser.add_argument("--memory-utilization", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Use a prefix of the test set for a smoke run")
    parser.add_argument("--prompts", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
    from cs336_alignment.vllm_utils import VLLMServer
    from vllm import SamplingParams

    # The provided HTTP helper omits top_p; verify that its server-side default
    # matches the handout's required value rather than altering provided code.
    assert SamplingParams().top_p == 1.0
    dataset_path = args.test_file
    dataset = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    assert len(dataset) == 1319
    if args.limit is not None:
        dataset = dataset[:args.limit]
    templates = {name: (ASSIGNMENT / "cs336_alignment" / "prompts" / f"{name}.prompt").read_text()
                 for name in args.prompts}
    model_dir = args.model
    manifest_path = model_dir / "resource-manifest.json"
    config = {
        "model_dir": str(model_dir), "model_manifest": json.loads(manifest_path.read_text()) if manifest_path.exists() else None,
        "dataset": "GSM8K test", "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "n_examples": len(dataset), "prompts": args.prompts,
        "prompt_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in templates.items()},
        "sampling": {"temperature": 1.0, "top_p": 1.0, "max_tokens": 512, "n": 1, "seed": args.seed},
        "r1_stop": ["</answer>"], "include_stop_str_in_output": True,
        "batch_size": args.batch_size, "gpu": args.gpu, "port": args.port,
        "memory_utilization": args.memory_utilization,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers")},
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        assert json.loads((output / "config.json").read_text()) == config, "Resume configuration differs"
    else:
        save_json(output / "config.json", config)
    server = VLLMServer(model_id=str(model_dir), gpu=args.gpu, port=args.port,
                        gpu_memory_utilization=args.memory_utilization, startup_timeout=600)
    metrics = {}
    audit_samples = {}
    started = time.monotonic()
    print(f"Evaluating {len(dataset)} GSM8K test examples per prompt on GPU {args.gpu}", flush=True)
    try:
        server.start()
        for prompt_number, name in enumerate(args.prompts):
            directory = output / name
            chunks = directory / "chunks"
            chunks.mkdir(parents=True, exist_ok=True)
            grader = question_only_reward_fn if name == "question_only" else r1_zero_reward_fn
            sampling = dict(config["sampling"])
            if name != "question_only":
                sampling.update(stop=["</answer>"], include_stop_str_in_output=True)
            records = []
            for start in range(0, len(dataset), args.batch_size):
                end = min(start + args.batch_size, len(dataset))
                chunk_path = chunks / f"{start:05d}.json"
                if chunk_path.exists():
                    batch_records = json.loads(chunk_path.read_text())
                    assert [row["index"] for row in batch_records] == list(range(start, end))
                else:
                    batch_prompts = [templates[name].format(question=row["question"]) for row in dataset[start:end]]
                    completions = server.generate_completions(batch_prompts, sampling)
                    assert len(completions) == end - start
                    batch_records = []
                    for index, prompt, completion in zip(range(start, end), batch_prompts, completions):
                        example = dataset[index]
                        assert "####" in example["answer"] and completion.token_ids
                        ground_truth = example["answer"].rsplit("####", 1)[1].strip()
                        rewards = grader(completion.text, ground_truth)
                        batch_records.append({
                            "index": index, "prompt_name": name, "question": example["question"],
                            "reference_solution": example["answer"], "ground_truth": ground_truth,
                            "prompt": prompt, "response": completion.text, "token_ids": completion.token_ids,
                            "finish_reason": completion.finish_reason, "rewards": rewards, "category": category(rewards),
                        })
                    save_json(chunk_path, batch_records)
                records.extend(batch_records)
                stats = summarize(records)
                save_json(directory / "progress.json", stats)
                print(f"{name}: {end}/{len(dataset)}, accuracy={stats['accuracy']:.4f}, "
                      f"categories={stats['category_counts']}, elapsed={time.monotonic()-started:.1f}s", flush=True)
            assert len(records) == len(dataset)
            (directory / "results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
            metrics[name] = summarize(records)
            save_json(output / "metrics.json", metrics)
            audit_samples[name] = {}
            for cat in (2, 3):
                candidates = [row for row in records if row["category"] == cat]
                chosen = random.Random(args.seed + prompt_number * 10 + cat).sample(candidates, min(10, len(candidates)))
                audit_samples[name][str(cat)] = [
                    {key: row[key] for key in ("index", "question", "reference_solution", "ground_truth", "response", "finish_reason")}
                    for row in chosen
                ]
            save_json(output / "audit_samples.json", audit_samples)
    finally:
        server.stop()
    print("Completed all requested prompting baselines:", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
