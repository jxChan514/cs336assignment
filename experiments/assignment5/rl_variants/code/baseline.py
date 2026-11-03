"""Original OLMo model, fixed three-shot prompt, 1024-question GSM8K baseline."""
from pathlib import Path
import hashlib
import json
import random
import signal
import time


def main():
    import wandb
    from transformers import AutoTokenizer
    from reference.drgrpo_grader import r1_zero_reward_fn
    from reference.vllm_utils import _http_json
    from server import Sampler

    code = Path(__file__).resolve().parent
    root = code.parent
    output = root / "outputs/baseline-3shot"
    if (output / "config.json").exists():
        raise FileExistsError(f"Existing baseline is preserved: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((root / "protocol.json").read_text())
    spec = json.loads((code / "env-spec.json").read_text())
    model = spec["model"]
    template = (code / "prompt.prompt").read_text()
    assert template.count("User:") == 4 and template.count("{question}") == 1
    data = []
    source = root / "data/gsm8k/test.jsonl"
    for index, line in enumerate(source.read_text().splitlines()):
        row = json.loads(line)
        data.append({"id": index, "question": row["question"],
                     "ground_truth": row["answer"].rsplit("####", 1)[1].strip()})
    random.Random(42).shuffle(data)
    data = data[:1024]
    assert len(data) == 1024 and not any(row["question"] in template for row in data)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    prompts = [tokenizer.encode(template.format(question=row["question"]),
                                add_special_tokens=False) for row in data]
    assert max(map(len, prompts)) + 512 <= 2048
    config = {"kind": "original-model-three-shot-baseline", "model": model,
              "prompt_sha256": hashlib.sha256(template.encode()).hexdigest(),
              "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "question_ids": [row["id"] for row in data], "n": 1,
              "temperature": 1.0, "top_p": 1.0, "max_tokens": 512, "seed": 100000,
              "training": False, "protocol": protocol["common"]}
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    sampler = Sampler(model, output, seed=0)
    run = None
    completed = False
    started = time.monotonic()
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        run = wandb.init(entity="mrxian123456-ca-si-a", project="cs336-a5-grpo",
                         name="3shot-baseline-20261029", group="a5-3shot-20261029",
                         job_type="baseline", config=config, dir=str(output), mode="online")
        (output / "wandb-run.json").write_text(json.dumps({"id": run.id, "url": run.url}) + "\n")
        print("BASELINE_WANDB", run.url, flush=True)
        sampler.start()
        records = []
        for start in range(0, len(data), 128):
            response = _http_json("POST", sampler.base_url + "/v1/completions",
                                  {"model": model, "prompt": prompts[start:start + 128],
                                   "n": 1, "temperature": 1.0, "top_p": 1.0,
                                   "top_k": -1, "min_p": 0.0, "max_tokens": 512,
                                   "seed": 100000, "stop": ["</answer>"],
                                   "include_stop_str_in_output": True, "return_token_ids": True},
                                  timeout=900)
            choices = sorted(response["choices"], key=lambda row: row["index"])
            assert len(choices) == len(data[start:start + 128])
            for example, choice in zip(data[start:start + 128], choices, strict=True):
                ids = choice.get("token_ids")
                assert ids is not None and len(ids) <= 512
                row = {**example, "response": choice["text"], "response_token_ids": ids,
                       "response_tokens": len(ids), "finish_reason": choice.get("finish_reason"),
                       "rewards": r1_zero_reward_fn(choice["text"], example["ground_truth"])}
                records.append(row)
                with (output / "responses.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("BASELINE_PROGRESS", len(records), "/", len(data), flush=True)
        metrics = {"val/reward": sum(r["rewards"]["reward"] for r in records) / len(records),
                   "val/format_reward": sum(r["rewards"]["format_reward"] for r in records) / len(records),
                   "val/response_length": sum(r["response_tokens"] for r in records) / len(records),
                   "val/truncated_fraction": sum(r["finish_reason"] == "length" for r in records) / len(records),
                   "val/examples": len(records), "system/baseline_seconds": time.monotonic() - started}
        run.log(metrics, step=0)
        run.log({"samples/baseline": wandb.Table(
            columns=["id", "question", "ground_truth", "response", "reward", "tokens"],
            data=[[r["id"], r["question"], r["ground_truth"], r["response"],
                   r["rewards"]["reward"], r["response_tokens"]] for r in records[:20]])})
        run.summary.update(metrics)
        result = {"status": "PASS", "metrics": metrics, "wandb_url": run.url, "training": False}
        (output / "completed.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        completed = True
        print("A5_3SHOT_BASELINE_COMPLETE", json.dumps(result), flush=True)
    except BaseException as error:
        (output / "failure.json").write_text(json.dumps(
            {"type": type(error).__name__, "message": str(error)}, ensure_ascii=False) + "\n")
        raise
    finally:
        sampler.stop()
        if run is not None:
            run.finish(exit_code=0 if completed else 1)


if __name__ == "__main__":
    main()
