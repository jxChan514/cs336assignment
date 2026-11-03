"""Preserve rolled-back evidence while selecting one checkpoint-consistent history."""
import json
from pathlib import Path


def rows(path):
    path = Path(path)
    if not path.exists():return []
    lines = path.read_text().splitlines(); result = []
    for i, line in enumerate(lines):
        if not line.strip():continue
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1:raise
            # A power loss may leave one incomplete final append, not a completed event.
    return result


def attempt_paths(root):
    root = Path(root)
    return [root]+sorted(p for p in (root/'attempts').glob('attempt-*') if (p/'resume.json').exists())


def canonical_rows(root, attempt, filename):
    root = Path(root).resolve(); directory = (root/attempt).resolve()
    if not directory.is_relative_to(root):raise ValueError('Attempt outside run root')
    result = []
    if (directory/'resume.json').exists():
        resume = json.loads((directory/'resume.json').read_text())
        parent = resume['parent_attempt']
        if parent == attempt:raise ValueError('Self-referencing resume')
        result = [r for r in canonical_rows(root, parent, filename)
                  if r.get('rollout_step', 0) <= resume['checkpoint_rollout_step']
                  and r.get('optimizer_update', 0) <= resume['checkpoint_optimizer_updates']]
    result += [{**r, '_attempt': str(directory.relative_to(root))} for r in rows(directory/filename)]
    return result


def lifetime_work(root):
    updates = []; events = []
    for directory in attempt_paths(root):
        updates.extend(rows(directory/'updates.jsonl'))
        events.extend(rows(directory/'sampling-events.jsonl'))
    return {'lifetime_optimizer_updates': len(updates),
            'lifetime_generated_responses': sum(r['generated_responses'] for r in events),
            'lifetime_generated_tokens': sum(r['generated_tokens'] for r in events)}
