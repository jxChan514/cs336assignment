from pathlib import Path
import json
import hashlib
import random


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)

def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def data_rows(path, count):
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if "####" not in row["answer"]:
            raise ValueError(f"Missing GSM8K delimiter: {path}:{index}")
        rows.append({"id": index, "question": row["question"],
                     "ground_truth": row["answer"].rsplit("####", 1)[1].strip()})
    random.Random(42).shuffle(rows)
    if not 0 < count <= len(rows):
        raise ValueError((count, len(rows)))
    return rows[:count]

def stats_for_records(records, group_size=1):
    import numpy as np
    rewards = np.asarray([r["rewards"]["reward"] for r in records])
    lengths = np.asarray([len(r["response_token_ids"]) for r in records])
    result = {"reward": float(rewards.mean()),
              "format_reward": sum(r["rewards"]["format_reward"] for r in records) / len(records),
              "response_length": float(lengths.mean()), "response_length_p95": float(np.quantile(lengths, .95)),
              "response_tokens": int(lengths.sum()),
              "truncated_fraction": sum(r["finish_reason"] == "length" for r in records) / len(records)}
    if group_size > 1:
        groups = rewards.reshape(-1, group_size)
        result["zero_std_group_fraction"] = float((groups.std(axis=1) == 0).mean())
    return result
