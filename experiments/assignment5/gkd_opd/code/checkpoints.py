"""Split recoverable snapshots; publish new state before deleting previous files."""
from pathlib import Path
import json
import random
import shutil
import time
import numpy as np
import torch
from common import write_json,sha,append_json


def save_checkpoint(root,output,model_root,profile,model,optimizer,data_rng,progress,config,device=None):
    root=Path(root);output=Path(output);model_root=Path(model_root)
    model_root.mkdir(parents=True,exist_ok=True)
    directory=output/'checkpoints'/f'step-{progress["optimizer_updates"]:04d}'
    directory.mkdir(parents=True,exist_ok=False)
    model_path=model_root/f'{profile}-step-{progress["optimizer_updates"]:04d}-model.pt'
    optimizer_path=directory/'optimizer.pt';state_path=directory/'state.pt'
    model_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values())
    optimizer_bytes=sum(t.numel()*t.element_size() for s in optimizer.state.values() for t in s.values() if isinstance(t,torch.Tensor))
    reserve=512*1024**2
    if shutil.disk_usage(model_root).free<model_bytes+reserve or shutil.disk_usage(directory).free<optimizer_bytes+reserve:
        raise OSError('Insufficient space for atomic checkpoint overlap; old checkpoint preserved')
    old_catalog=root/'monitoring/latest-recoverable-checkpoint.json'
    previous=json.loads(old_catalog.read_text()) if old_catalog.exists() else None
    started=time.monotonic()
    payload={n:t.detach().cpu() for n,t in model.state_dict().items()}
    temporary=model_path.with_suffix('.pt.writing');torch.save(payload,temporary);del payload
    temporary.replace(model_path)
    temporary=optimizer_path.with_suffix('.pt.writing');torch.save(optimizer.state_dict(),temporary);temporary.replace(optimizer_path)
    state={'format_version':2,'progress':dict(progress),'config':config,
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'data':data_rng.getstate(),
            'torch_cuda':torch.cuda.get_rng_state(device) if device is not None else None}}
    torch.save(state,state_path)
    files={key:{'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)} for key,path in [('model',model_path),('optimizer',optimizer_path),('state',state_path)]}
    manifest={'status':'READY','format_version':2,'optimizer_updates':progress['optimizer_updates'],'profile':profile,
        'attempt_relative':progress.get('attempt_relative','.'),'files':files,'manifest_path':str(directory/'manifest.json'),
        'protocol_sha256':config['protocol_sha256'],'saved_seconds':time.monotonic()-started,'remote_backup_performed':False}
    write_json(directory/'manifest.json',manifest)
    write_json(output/'latest-checkpoint.json',manifest)
    write_json(old_catalog,manifest)
    if previous and previous['manifest_path']!=manifest['manifest_path']:
        removed=[]
        for key,item in previous['files'].items():
            path=Path(item['path'])
            allowed=path.parent==model_root if key=='model' else path.is_relative_to(root/'outputs')
            if not allowed:raise RuntimeError('Previous artifact outside authorized experiment directories')
            if path.exists():path.unlink();removed.append(str(path))
        append_json(root/'monitoring/checkpoint-rotation.jsonl',{'old_manifest':previous['manifest_path'],'new_manifest':manifest['manifest_path'],
            'removed':removed,'reason':'Verified new local snapshot; retain latest one under source-only policy','time':time.time()})
        old_manifest=Path(previous['manifest_path'])
        if old_manifest.exists():write_json(old_manifest,{**previous,'status':'ROTATED','superseded_by':manifest['manifest_path']})
    print('CHECKPOINT_READY',json.dumps(manifest),flush=True)
    return manifest


def restore_checkpoint(manifest_path,model,optimizer,data_rng,device=None):
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest['status']!='READY':raise ValueError('Checkpoint is not the current retained snapshot')
    for item in manifest['files'].values():
        path=Path(item['path'])
        if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Checkpoint size/SHA256 verification failed')
    model.load_state_dict(torch.load(manifest['files']['model']['path'],map_location='cpu',weights_only=True))
    optimizer.load_state_dict(torch.load(manifest['files']['optimizer']['path'],map_location='cpu',weights_only=False))
    state=torch.load(manifest['files']['state']['path'],map_location='cpu',weights_only=False)
    rng=state['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch_cpu']);data_rng.setstate(rng['data'])
    if device is not None and rng['torch_cuda'] is not None:torch.cuda.set_rng_state(rng['torch_cuda'],device)
    return state
