"""Honor logs-only user policy in the live trainer's already-imported save function.

The legacy filename is consumed by the existing running code. These records
verify LOCAL file integrity and authorize local retention; they explicitly do
not claim a remote copy or remote backup verification.
"""
from pathlib import Path
import json,time,datetime
from common import sha,write_json

root=Path(__file__).resolve().parent.parent
run=root/'outputs/formal/dapo/seed0'
while True:
    policy=json.loads((root/'checkpoint-policy.json').read_text())
    if policy.get('remote_backup_required',True):raise RuntimeError('Local rotation is not authorized')
    f=run/'checkpoint-status.json'
    if f.exists():
        manifest=json.loads(f.read_text());path=Path(manifest['path'])
        target=run/'local-checkpoint-verifications'/(path.stem+'.json')
        if path.exists() and not target.exists():
            digest=sha(path)
            if path.stat().st_size!=manifest['bytes'] or digest!=manifest['sha256']:
                write_json(root/'monitoring/local-checkpoint-policy-failure.json',
                    {'status':'FAILED','path':str(path),'reason':'Local checkpoint size/SHA256 mismatch'})
                raise RuntimeError('Local checkpoint integrity failed; rotation not authorized')
            record={'status':'VERIFIED','verification_scope':'local_checkpoint_integrity_only',
                'sha256':digest,'bytes':path.stat().st_size,'rollout_step':manifest['rollout_step'],
                'remote_backup_performed':False,'destination':None,
                'allows_single_local_checkpoint_rotation':True,
                'authorization':'User disabled automatic weight synchronization; retain latest training-server checkpoint',
                'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write_json(target,record)
            # Compatibility with the live function, not a remote backup receipt.
            write_json(run/'checkpoint-backups'/(path.stem+'.json'),
                {**record,'legacy_consumer_bridge':True})
            print('LOCAL_ONLY_ROTATION_AUTHORIZED',manifest['rollout_step'],flush=True)
    queue=root/'outputs/queue-status.json'
    if queue.exists() and json.loads(queue.read_text()).get('status') in ['complete','interrupted','stopped']:
        break
    time.sleep(5)
