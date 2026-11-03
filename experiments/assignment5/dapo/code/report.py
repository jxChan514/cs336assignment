import json,csv
from pathlib import Path
root=Path(__file__).resolve().parent.parent
run=root/'outputs/formal/dapo/seed0';out=root/'outputs/report';out.mkdir(exist_ok=True)
summary=json.loads((run/'completed.json').read_text())
assert summary['status']=='PASS' and summary['completed_rollout_steps']==200 and summary['optimizer_updates']==800
verification=json.loads((run/'verification.json').read_text())
report={**summary,**verification,'status':'COMPLETE','baseline_accuracy':.205078125,
 'effective_storage_policy':json.loads((root/'checkpoint-policy.json').read_text()),
 'model_storage':'training_server_only','minimind_archive_scope':'logs_only',
 'final_gain_percentage_points':100*(summary['final_validation']['reward']-.205078125),
 'comparison_scope':'Adapted DAPO recipe, seed0. Different learning rate, update schedule and generated sample count from historical methods; not a single-component ablation.',
 'checkpoint_status':json.loads((run/'checkpoint-status.json').read_text())}
(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
with (out/'summary.csv').open('w') as f:
 columns=['method','seed','completed_rollout_steps','optimizer_updates','best_val_reward','best_validation_step','generated_responses','generated_tokens','elapsed_seconds','final_gain_percentage_points']
 w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerow(report)
print('DAPO_REPORT_COMPLETE',json.dumps(report))
