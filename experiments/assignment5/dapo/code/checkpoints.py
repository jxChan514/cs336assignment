"""Recoverable trainer snapshots with an explicit operational retention policy."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from common import write_json, sha, append_json


def capture_state(model, optimizer, data_rng, progress, config, cuda_device=None):
    return {'format_version': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'progress': progress, 'config': config,
            'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch_cpu': torch.get_rng_state(), 'data': data_rng.getstate(),
                    'torch_cuda': torch.cuda.get_rng_state(cuda_device) if cuda_device is not None else None}}


def restore_state(state, model, optimizer, data_rng, cuda_device=None):
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    rng = state['rng']
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu']); data_rng.setstate(rng['data'])
    if cuda_device is not None and rng['torch_cuda'] is not None:
        torch.cuda.set_rng_state(rng['torch_cuda'], cuda_device)
    return state['progress']


def checkpoint_policy():
    path=Path(__file__).resolve().parent.parent/'checkpoint-policy.json'
    return json.loads(path.read_text()) if path.exists() else {'remote_backup_required':True,'source_keep':1}


def save_recoverable(output, model, optimizer, data_rng, progress, config, cuda_device):
    directory = Path(output)/'checkpoints'
    directory.mkdir(exist_ok=True)
    old_files=sorted(directory.glob('step-*.pt'))
    needed=sum(p.numel()*p.element_size() for p in model.parameters())*3+1024**3
    if shutil.disk_usage(directory).free+sum(p.stat().st_size for p in old_files)<needed:
        raise OSError('Insufficient space even after authorized local checkpoint rotation; previous checkpoint preserved')
    policy=checkpoint_policy()
    deadline=time.monotonic()+1800
    for old in old_files:
        receipt=Path(output)/'checkpoint-backups'/(old.stem+'.json')
        while policy.get('remote_backup_required',True) and not receipt.exists():
            if time.monotonic()>deadline:
                raise TimeoutError('Previous checkpoint backup not verified; preserved source '+str(old))
            print('WAIT_CHECKPOINT_BACKUP',old.name,flush=True)
            time.sleep(15)
            policy=checkpoint_policy()
        if policy.get('remote_backup_required',True):
            backed=json.loads(receipt.read_text())
            manifest=json.loads(old.with_suffix('.json').read_text())
            if (backed.get('status')!='VERIFIED' or backed.get('remote_backup_performed') is False
                or backed.get('sha256')!=manifest['sha256'] or backed.get('bytes')!=old.stat().st_size):
                raise RuntimeError('Remote backup receipt mismatch; source preserved')
        append_json(Path(output)/'checkpoint-rotation-events.jsonl',{'old_path':str(old),
            'next_rollout':progress['rollout_step'],'remote_backup_required':policy.get('remote_backup_required',True),
            'reason':'Keep latest one checkpoint on training server under explicit user policy',
            'timestamp':time.time()})
        old.unlink()
    if shutil.disk_usage(directory).free<needed:
        raise OSError('Insufficient disk for full optimizer checkpoint; no incomplete checkpoint published')
    step = progress['rollout_step']
    path = directory/f'step-{step:04d}.pt'
    temporary = path.with_suffix('.pt.tmp')
    torch.save(capture_state(model, optimizer, data_rng, progress, config, cuda_device), temporary)
    digest = sha(temporary)
    temporary.replace(path)
    manifest = {'status': 'READY', 'path': str(path), 'rollout_step': step,
                'optimizer_updates': progress['optimizer_updates'], 'bytes': path.stat().st_size,
                'sha256': digest, 'format_version': 1, 'wandb_run_id': progress['wandb_run_id'],
                'effective_retention_policy': policy,
                'contents': ['FP32 model', 'Adam states', 'RNG states', 'counters', 'configuration', 'validation history']}
    write_json(path.with_suffix('.json'), manifest)
    write_json(Path(output)/'checkpoint-status.json', manifest)
    print('CHECKPOINT_READY', json.dumps(manifest), flush=True)
    return manifest
