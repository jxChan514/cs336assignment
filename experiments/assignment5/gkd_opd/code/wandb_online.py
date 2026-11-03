"""Publish only real append-only server metrics to the same W&B run, continuously."""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time
import wandb
from common import write_json

ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'outputs/formal/gkd_opd/seed0'
PATH='mrxian123456-ca-si-a/cs336-a5-grpo/'


def source_rows():
    files=[RUN/'metrics.jsonl']+sorted((RUN/'attempts').glob('*/metrics.jsonl'))
    rows=[]
    for path in files:
        if not path.exists():continue
        for line in path.read_text().splitlines(keepends=True):
            if line.endswith('\n') and line.strip():rows.append(json.loads(line))
    return rows


def verify_cloud(run_id,rows):
    cloud=wandb.Api(timeout=20).run(PATH+run_id)
    history=list(cloud.scan_history(keys=['_step','optimizer_update'],min_step=max(0,len(rows)-5),max_step=len(rows),page_size=100))
    last=max((r['_step'] for r in history),default=-1)
    for item in history:
        index=int(item['_step']);assert rows[index]['optimizer_update']==item['optimizer_update']
    return last,cloud.url


def main():
    (ROOT/'monitoring').mkdir(exist_ok=True)
    with (ROOT/'monitoring/wandb-online.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=json.loads((RUN/'config.json').read_text())
        run_id=json.loads((RUN/'wandb-run.json').read_text())['id']
        target=RUN/'online-logging';target.mkdir(exist_ok=True)
        run=wandb.init(entity='mrxian123456-ca-si-a',project='cs336-a5-grpo',id=run_id,resume='allow',
            name='gkd-opd-formal-seed0',group=config['experiment'],job_type='formal',
            config={**config,'logging_transport':'online bridge from real server metrics; offline trainer events retained as safety copy'},
            dir=str(target),mode='online',settings=wandb.Settings(init_timeout=60))
        run.define_metric('lifetime_optimizer_updates')
        for namespace in ['train','val','system','debug']:run.define_metric(namespace+'/*',step_metric='lifetime_optimizer_updates')
        submitted=int(run.step);last_check=0.;stopping=False;terminal=False
        def stop(number,frame):
            nonlocal stopping
            stopping=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        publication={'status':'ONLINE','id':run_id,'url':run.url,'pid':os.getpid(),'transport':'source JSONL -> W&B online SDK',
            'trainer_sdk_mode':'offline','training_restarted':False,'weights_uploaded':False}
        write_json(ROOT/'monitoring/wandb-publication.json',publication)
        write_json(RUN/'wandb-online.json',publication)
        old=json.loads((RUN/'wandb-run.json').read_text())
        write_json(RUN/'wandb-run.json',{**old,'url':run.url,'cloud_publication':'online_bridge','cloud_url':run.url})
        print('WANDB_ONLINE_BRIDGE_READY',json.dumps(publication),flush=True)
        try:
            while not stopping:
                rows=source_rows()
                for index in range(submitted,len(rows)):
                    run.log(rows[index],step=index);submitted=index+1
                if time.monotonic()-last_check>=20:
                    last_check=time.monotonic()
                    try:
                        last,url=verify_cloud(run_id,rows)
                        status={'status':'VERIFIED' if last>=submitted-1 else 'UPLOADING','id':run_id,'url':url,
                            'submitted_rows':submitted,'confirmed_cloud_step':last,
                            'confirmed_optimizer_update':rows[last]['optimizer_update'] if last>=0 else None,
                            'source_last_optimizer_update':rows[-1]['optimizer_update'] if rows else None,
                            'timestamp':time.time(),'pid':os.getpid(),'training_restarted':False}
                        write_json(ROOT/'monitoring/wandb-sync-status.json',status)
                        print('WANDB_CLOUD_PROGRESS',json.dumps(status),flush=True)
                    except Exception as error:
                        write_json(ROOT/'monitoring/wandb-sync-status.json',{'status':'RETRYING','submitted_rows':submitted,'timestamp':time.time(),'pid':os.getpid(),'error':str(error)[:300]})
                queue_path=ROOT/'outputs/queue-status.json'
                queue=json.loads(queue_path.read_text()) if queue_path.exists() else {}
                if queue.get('status') in ['COMPLETE','INTERRUPTED','FAILED','VERIFICATION_FAILED','STOPPED'] and submitted==len(rows):
                    report=ROOT/'outputs/report/summary.json'
                    if report.exists():run.summary.update(json.loads(report.read_text()))
                    run.summary['logging_transport']='online_bridge';run.summary['source_trainer_sdk_mode']='offline'
                    terminal=True;run.finish(exit_code=0 if queue['status']=='COMPLETE' else 1)
                    last,url=verify_cloud(run_id,rows)
                    if last!=len(rows)-1:raise RuntimeError('Final cloud history is incomplete')
                    receipt={'status':'VERIFIED','id':run_id,'url':url,'rows':len(rows),'confirmed_cloud_step':last,
                        'final_optimizer_update':rows[last]['optimizer_update'],'training_status':queue['status'],'timestamp':time.time(),'weights_uploaded':False}
                    write_json(ROOT/'monitoring/wandb-final-receipt.json',receipt)
                    print('WANDB_FINAL_UPLOAD_VERIFIED',json.dumps(receipt),flush=True);return
                time.sleep(3)
        finally:
            if not terminal:run.finish(exit_code=1)


if __name__=='__main__':main()
