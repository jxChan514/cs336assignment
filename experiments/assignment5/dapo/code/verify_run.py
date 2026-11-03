"""Read the canonical checkpoint lineage; preserve and count rolled-back work."""
import argparse
import json
import math
from pathlib import Path
from history import canonical_rows, lifetime_work
from common import write_json

parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
p=args.run;summary=json.loads((p/'completed.json').read_text());config=json.loads((p/'config.json').read_text())
attempt=summary.get('active_attempt','.')
updates=canonical_rows(p,attempt,'updates.jsonl');metrics=canonical_rows(p,attempt,'metrics.jsonl')
train=[x for x in metrics if 'train/reward' in x];val=[x for x in metrics if 'val/reward' in x]
steps=config['effective_rollout_steps']
assert summary['status']=='PASS' and summary['completed_rollout_steps']==steps
assert len(updates)==steps*4==summary['optimizer_updates']
assert [r['optimizer_update'] for r in updates]==list(range(1,steps*4+1))
assert len(train)==steps and all(x['sampling/accepted_groups']==32 for x in train)
assert len(val)>=2 and val[-1]['rollout_step']==steps
assert summary['nonzero_gradient_updates']>0 and summary['changed_parameter_updates']>0
first_update_errors=[]
for step in range(1,steps+1):
 rows=[x for x in updates if x['rollout_step']==step]
 assert len(rows)==4 and len({x['old_log_probs_sha256'] for x in rows})==1
 assert sorted(i for x in rows for i in x['sample_indices'])==list(range(256))
 assert all(len(x['sample_indices'])==64 for x in rows)
 # BF16 no-grad vs gradient-enabled kernels need numerical, not bitwise equality.
 first=rows[0]
 first_error=max(abs(math.log(first['ratio_min'])),abs(math.log(first['ratio_max'])))
 first_update_errors.append(first_error)
 assert first_error <= .02
 assert abs(first['ratio_mean']-1.) <= 1e-4
 assert abs(math.log(first['ratio_p99'])) <= .005
 assert first['approx_kl_k3'] <= 1e-6 and first['clip_fraction'] == 0.
 audit=json.loads((p/rows[0]['_attempt']/'sync-audits'/f'step-{step:04d}.json').read_text())
 assert audit['mean_abs']<=config['sampler_score_audit_mean_abs_limit']
 assert audit['p99_abs']<=config['sampler_score_audit_p99_abs_limit']
work=lifetime_work(p)
assert all(summary[k]==v for k,v in work.items())
peak=max(val,key=lambda x:x['val/reward'])
receipt={'status':'PASS','steps':steps,'updates':len(updates),**work,
 'canonical_peak_validation':peak['val/reward'],'canonical_peak_rollout':peak['rollout_step'],
 'final_reward':summary['final_validation']['reward'],'checkpoint_enabled':config['checkpoint_enabled'],
 'first_update_abs_log_ratio_limit':.02,'first_update_approx_kl_limit':1e-6,
 'first_update_max_abs_log_ratio_observed':max(first_update_errors)}
write_json(p/'verification.json',receipt)
print('DAPO_SAVED_RUN_PASS',json.dumps(receipt))
