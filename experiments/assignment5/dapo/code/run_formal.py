"""One formal run, including checkpoint resume and its complete archive lifecycle."""
import argparse
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from common import write_json, sha


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    root.joinpath('preflight').mkdir(exist_ok=True)
    with (root/'preflight/formal.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ['unit-tests-pass.json', 'code-review-pass.json', 'smoke-pass.json']:
            assert json.loads((root/'preflight'/name).read_text())['status'] == 'PASS'
        heartbeat = json.loads((root/'monitoring/backup-heartbeat.json').read_text())
        if heartbeat['status'] != 'PASS' or time.time()-heartbeat['timestamp'] > 180:
            raise RuntimeError('Checkpoint backup worker is not currently ready')
        out = root/'outputs/formal/dapo/seed0'; out.mkdir(parents=True, exist_ok=True)
        if (out/'completed.json').exists():raise RuntimeError('Formal run already complete; no duplicate launch')
        if args.resume:
            manifest = json.loads((out/'checkpoint-status.json').read_text())
            if args.resume.resolve() != Path(manifest['path']).resolve():
                raise ValueError('Resume must explicitly select the latest published checkpoint')
            if args.resume.stat().st_size != manifest['bytes'] or sha(args.resume) != manifest['sha256']:
                raise ValueError('Resume checkpoint size/hash mismatch')
        elif (out/'config.json').exists():
            raise RuntimeError('Existing formal run requires --resume after inspection')
        base = {'method': 'dapo', 'seed': 0, 'coordinator_pid': os.getpid(), 'output': str(out)}
        write_json(root/'outputs/queue-status.json', {**base, 'status': 'running', 'phase': 'training'})
        phase = 'training'; child = None
        def terminate(number, frame):
            raise SystemExit(128+number)
        signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
        try:
            command = [sys.executable, str(root/'code/train_dapo.py'), '--profile', 'formal', '--output', str(out)]
            if args.resume:command += ['--resume', str(args.resume)]
            with (out/'driver.log').open('a') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                returncode = child.wait()
            if returncode:raise RuntimeError('Trainer exit code '+str(returncode))
            phase = 'verification'
            subprocess.run([sys.executable, str(root/'code/verify_run.py'), '--run', str(out)], check=True)
            phase = 'report'
            subprocess.run([sys.executable, str(root/'code/report.py')], check=True)
            write_json(root/'outputs/queue-status.json', {**base, 'status': 'complete',
                'completed_formal_runs': 1, 'completed_rollouts': 200, 'optimizer_updates': 800, 'archive': 'pending'})
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    write_json(root/'outputs/queue-status.json', {**base, 'status': 'stopping',
                        'error': 'Trainer cleanup still pending; do not archive a live run', 'trainer_pid': child.pid})
                    raise
            write_json(root/'outputs/queue-status.json', {**base, 'status': 'stopped', 'phase': phase,
                'error': str(error), 'completed_formal_runs': 0, 'archive': 'pending'})
            raise


if __name__ == '__main__':main()
