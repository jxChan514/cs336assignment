"""Verify already-written GKD evidence. Never generate, train, or load GPU models."""
import argparse
import gzip
import json
import math
from pathlib import Path
from common import write_json,sha,stats_for_records
from history import canonical_rows
from reference.drgrpo_grader import r1_zero_reward_fn

ROOT=Path(__file__).resolve().parent.parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['smoke','formal'],required=True);args=parser.parse_args()
    formal=args.profile=='formal';run=ROOT/'outputs'/args.profile/'gkd_opd/seed0'
    summary=json.loads((run/'completed.json').read_text());target=800 if formal else 1
    assert summary['status']=='COMPLETE' and summary['optimizer_updates']==target
    updates=canonical_rows(run,'updates.jsonl')
    assert [r['optimizer_update'] for r in updates]==list(range(1,target+1))
    assert summary['generated_responses']==target*64
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['grad_norm']>0 and r['parameter_probe_changed'] and r['teacher_probe_unchanged'] and r['teacher_gradients_absent'] for r in updates)
    assert all(r['sampler_audit']['mean_abs']<=.1 and r['sampler_audit']['p99_abs']<=.5 for r in updates)
    if not formal:
        assert json.loads((run/'checkpoint-restore-check.json').read_text())['status']=='PASS'
        assert json.loads((run/'post-sync-check.json').read_text())['examples']==2
        with gzip.open(run/'training-details/step-0001.json.gz','rt') as f:rows=json.load(f)
        assert len(rows)==64
        for row in rows:
            assert len(row['response_token_ids'])==len(row['token_reverse_kl'])==len(row['student_sampled_log_probs'])==len(row['teacher_sampled_log_probs'])
        tokens=sum(len(r['response_token_ids']) for r in rows)
        kl=sum(sum(r['token_reverse_kl']) for r in rows)/tokens
        assert math.isclose(kl,updates[0]['loss'],rel_tol=2e-5,abs_tol=2e-6)
    else:
        validation_logs=[r for r in canonical_rows(run,'metrics.jsonl') if 'val/reward' in r and r['optimizer_update']==800]
        assert validation_logs
        path=run/validation_logs[-1].get('attempt','.')/'validation/step-0800.json.gz'
        with gzip.open(path,'rt') as f:rows=json.load(f)
        assert len(rows)==1024
        for row in rows:assert row['rewards']==r1_zero_reward_fn(row['response'],row['ground_truth'])
        recomputed=stats_for_records(rows)
        for name,value in recomputed.items():assert math.isclose(value,summary['last_validation'][name],abs_tol=1e-12)
    checkpoint_path=ROOT/'monitoring/latest-recoverable-checkpoint.json' if formal else run/'checkpoints/step-0001/manifest.json'
    checkpoint=json.loads(checkpoint_path.read_text())
    assert checkpoint['profile']==args.profile and checkpoint['optimizer_updates']==target
    assert checkpoint['status']=='READY' or (not formal and checkpoint['status']=='ROTATED' and checkpoint.get('superseded_by'))
    if checkpoint['status']=='READY':
        for item in checkpoint['files'].values():assert Path(item['path']).is_file() and Path(item['path']).stat().st_size==item['bytes']
    result={'status':'PASS','profile':args.profile,'optimizer_updates':target,'responses':64*target,'full_vocab_reverse_kl':True,
        'checkpoint_restore_tested':not formal,'teacher_frozen':True,'sampler_audits_passed':True,'verification_scope':'saved evidence; no new GPU work'}
    write_json(run/'verification.json',result)
    if not formal:write_json(ROOT/'preflight/smoke-validation.json',result)
    else:
        final_info=json.loads((run/'final-model.json').read_text())
        best_path=run/'best-model.json'
        best_info=json.loads(best_path.read_text()) if best_path.exists() else {'optimizer_update':0,'path':json.loads((run/'config.json').read_text())['student'],'origin':'original baseline'}
        for info in [final_info,best_info]:
            assert list(Path(info['path']).glob('*.safetensors'))
        curve=[r['val/reward'] for r in canonical_rows(run,'metrics.jsonl') if 'val/reward' in r]
        write_json(ROOT/'outputs/report/summary.json',{**summary,'verification':result,'final_model':final_info,'saved_best_model':best_info,'canonical_peak_val_reward':max(curve),
            'comparison_scope':'OPD uses additional7B teacher supervision; not a matched algorithm-only ablation'})
    print('GKD_OPD_SAVED_RUN_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
