"""Single locked formal coordinator, gated on the real one-update smoke."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from common import write_json

ROOT=Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path);args=parser.parse_args()
    with (ROOT/'preflight/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in ['math-cpu-tests.json','smoke-validation.json','doc-following-validation.json']:
            assert json.loads((ROOT/'preflight'/name).read_text())['status']=='PASS',name
        heartbeat=json.loads((ROOT/'monitoring/backup-heartbeat.json').read_text())
        assert heartbeat['status']=='SYNCED' and time.time()-heartbeat['timestamp']<900,'Logs worker is not ready'
        formal=ROOT/'outputs/formal/gkd_opd/seed0'
        if (formal/'completed.json').exists():raise FileExistsError('Formal run already completed; refusing duplicate')
        if args.resume:assert json.loads(args.resume.read_text())['profile']=='formal','Smoke checkpoint cannot initialize formal training'
        command=[sys.executable,str(ROOT/'code/train.py'),'--profile','formal']
        if args.resume:command+=['--resume',str(args.resume.resolve())]
        child=None
        def terminate(number,frame):
            if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
        signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
        queue={'status':'RUNNING','method':'gkd_reverse_kl_full_vocab','seed':0,'target_optimizer_updates':800,'coordinator_pid':os.getpid(),
            'started_at':time.time(),'resume_manifest':str(args.resume) if args.resume else None}
        write_json(ROOT/'outputs/queue-status.json',queue)
        with (ROOT/'outputs/formal-driver.log').open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            queue['trainer_pid']=child.pid;write_json(ROOT/'outputs/queue-status.json',queue)
            code=child.wait()
        if code==0:
            verification=subprocess.run([sys.executable,str(ROOT/'code/verify_run.py'),'--profile','formal'],check=False)
            queue['status']='COMPLETE' if verification.returncode==0 else 'VERIFICATION_FAILED'
        elif (formal/'interrupted.json').exists():queue['status']='INTERRUPTED'
        else:queue['status']='STOPPED' if code in [-15,143,130] else 'FAILED'
        queue.update(exit_code=code,finished_at=time.time())
        if queue['status']!='COMPLETE':
            progress=json.loads((formal/'progress.json').read_text()) if (formal/'progress.json').exists() else {}
            write_json(ROOT/'outputs/report/summary.json',{'status':queue['status'],'optimizer_updates':progress.get('optimizer_updates',0),
                'target_updates':800,'last_observed_validation':progress.get('last_validation'),'scope':'partial run; no completed800-update final result'})
        write_json(ROOT/'outputs/queue-status.json',queue)
        print('OPD_COORDINATOR_EXIT',json.dumps(queue),flush=True)


if __name__=='__main__':main()
