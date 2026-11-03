"""Dataset subsets, result metrics, and optional experiment tracking."""
import json
from pathlib import Path
import random


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def data_rows(path, count, subset_seed=42):
    rows = []
    for index, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if "####" not in row["answer"]:
            raise ValueError(f"Missing GSM8K answer delimiter on line {index + 1}")
        rows.append({"id": index, "question": row["question"],
                     "ground_truth": row["answer"].rsplit("####", 1)[1].strip()})
    random.Random(subset_seed).shuffle(rows)
    if not 0 < count <= len(rows):
        raise ValueError("Requested subset exceeds dataset size")
    return rows[:count]


def stats(records, group_size=1):
    import numpy as np
    rewards = np.asarray([r["rewards"]["reward"] for r in records])
    lengths = np.asarray([len(r["response_token_ids"]) for r in records])
    result = {"reward": float(rewards.mean()),
              "format_reward": sum(r["rewards"]["format_reward"] for r in records) / len(records),
              "response_length": float(lengths.mean()),
              "truncated_fraction": sum(r["finish_reason"] == "length" for r in records) / len(records)}
    if group_size > 1:
        result["zero_std_group_fraction"] = float((rewards.reshape(-1, group_size).std(-1) == 0).mean())
    return result


class Metrics:
    def __init__(self, output, config, project=None, entity=None):
        self.path = Path(output) / "metrics.jsonl"
        self.run = None
        if project:
            import wandb
            self.run = wandb.init(project=project, entity=entity, config=config,
                                  name=f"{config['method']}-seed{config['seed']}", dir=str(output))

    def log(self, row):
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if self.run is not None:
            self.run.log(row)
        print(json.dumps(row), flush=True)

    def close(self):
        if self.run is not None:
            self.run.finish()
