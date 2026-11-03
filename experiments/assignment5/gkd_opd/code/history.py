"""Read effective update lineage separately from all preserved lifetime work."""
from pathlib import Path
import json


def active_attempt(root):
    root=Path(root);p=root/'active-attempt.json'
    return json.loads(p.read_text())['attempt_relative'] if p.exists() else '.'


def canonical_rows(root,filename):
    root=Path(root)
    def collect(relative,limit=None):
        attempt=root/relative;p=attempt/'resume.json';rows=[]
        if p.exists():
            resume=json.loads(p.read_text());parent_limit=resume['checkpoint_update']
            rows.extend(collect(resume['parent_attempt'],min(limit,parent_limit) if limit is not None else parent_limit))
        f=attempt/filename
        if f.exists():
            for line in f.read_text().splitlines():
                if not line.strip():continue
                row=json.loads(line)
                if limit is None or row['optimizer_update']<=limit:rows.append(row)
        return rows
    return collect(active_attempt(root))


def lifetime_updates(root):
    root=Path(root);files=[root/'updates.jsonl']+list((root/'attempts').glob('*/updates.jsonl'))
    return sum(sum(bool(x.strip()) for x in p.read_text().splitlines()) for p in files if p.exists())
