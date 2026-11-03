"""Fresh student rollouts, one direct full-vocabulary reverse-KL update per batch."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
import torch
import wandb
from transformers import AutoModelForCausalLM,AutoTokenizer
from common import write_json,append_json,sha,data_rows,stats_for_records
from gkd import response_objective,warmup_lr
from checkpoints import save_checkpoint,restore_checkpoint
from history import lifetime_updates,canonical_rows
from reference.grpo import pack_prompt_and_response_ids
from reference.drgrpo_grader import r1_zero_reward_fn
from reference.vllm_utils import _http_json
from server import Sampler

ROOT=Path(__file__).resolve().parent.parent
OLD=Path('/root/autodl-tmp/cs336-a5-20261029')
QUALIFICATION=Path('/root/autodl-tmp/cs336-a5-opd-teachercheck-20261102')


class QualityStop(RuntimeError):pass


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['smoke','formal'],required=True);parser.add_argument('--resume',type=Path)
    args=parser.parse_args();formal=args.profile=='formal'
    cfg=json.loads((ROOT/'protocol.json').read_text());target=cfg['optimizer_updates'] if formal else 1
    run_root=ROOT/'outputs'/args.profile/'gkd_opd/seed0';output=run_root;attempt='.'
    if (run_root/'config.json').exists() and args.resume is None:raise FileExistsError('Existing run preserved; use an explicit checkpoint resume')
    run_root.mkdir(parents=True,exist_ok=True)
    if args.resume:
        index=1
        while (run_root/'attempts'/f'attempt-{index:04d}').exists():index+=1
        output=run_root/'attempts'/f'attempt-{index:04d}';output.mkdir(parents=True)
        attempt=str(output.relative_to(run_root))
    config={**cfg,'profile':args.profile,'target_updates':target,'protocol_sha256':sha(ROOT/'protocol.json'),
        'prompt_sha256':sha(ROOT/'code/prompt.prompt'),'train_sha256':sha(OLD/'data/gsm8k/train.jsonl'),
        'val_sha256':sha(OLD/'data/gsm8k/test.jsonl'),'initial_baseline_origin':'reused exact matched original1B baseline, not newly generated',
        'critical_code_sha256':{n:sha(ROOT/'code'/n) for n in ['train.py','gkd.py','checkpoints.py','server.py']}}
    assert json.loads((QUALIFICATION/'preflight/tokenizer-compatibility.json').read_text())['status']=='PASS'
    assert json.loads((QUALIFICATION/'preflight/teacher-weights-verified.json').read_text())['revision']==cfg['teacher_revision']
    write_json(output/'config.json',config)
    write_json(run_root/'active-attempt.json',{'attempt_relative':attempt})
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);rng=random.Random(cfg['seed'])
    train=data_rows(OLD/'data/gsm8k/train.jsonl',6400);validation=data_rows(OLD/'data/gsm8k/test.jsonl',1024)
    template=(ROOT/'code/prompt.prompt').read_text();assert template.count('User:')==4
    tokenizer=AutoTokenizer.from_pretrained(cfg['student'],local_files_only=True)
    baseline_config=json.loads((OLD/'outputs/baseline-3shot/config.json').read_text())
    assert baseline_config['question_ids']==[r['id'] for r in validation] and baseline_config['prompt_sha256']==config['prompt_sha256']
    with (OLD/'outputs/baseline-3shot/responses.jsonl').open() as f:base_rows=[json.loads(x) for x in f if x.strip()]
    initial_val=stats_for_records(base_rows);assert initial_val['reward']==.205078125 and len(base_rows)==1024
    progress={'optimizer_updates':0,'generated_responses':0,'generated_tokens':0,'initial_validation':initial_val,'last_validation':initial_val,
        'best_val_reward':initial_val['reward'],'best_validation_update':0,'nonzero_gradient_updates':0,'changed_parameter_updates':0,
        'cumulative_seconds':0.,'attempt_relative':attempt,'lifetime_optimizer_updates':lifetime_updates(run_root)}
    policy=None;teacher=None;sampler=None;run=None;complete=False;quality_stopped=False;started=time.monotonic();previous_seconds=0.
    def terminate(number,frame):raise SystemExit(128+number)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    def status(phase,**extra):
        write_json(run_root/'progress.json',{'status':phase,'attempt_relative':attempt,'optimizer_updates':progress['optimizer_updates'],
            'target_updates':target,'lifetime_optimizer_updates':progress['lifetime_optimizer_updates'],
            'last_validation':progress.get('last_validation'),'best_val_reward':progress.get('best_val_reward'),**extra})
    def save_records(folder,name,rows):
        path=output/folder/name;path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.writing')
        with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=5) as f:json.dump(rows,f,ensure_ascii=False)
        temporary.replace(path)
    def probe(model):
        params=[p for p in model.parameters() if p.numel()>=4096]
        values=[p.detach().flatten()[::max(p.numel()//1024,1)][:1024].float().cpu() for p in [params[0],params[len(params)//2],params[-1]]]
        return hashlib.sha256(torch.cat(values).numpy().tobytes()).hexdigest()
    def log(payload):
        payload={**payload,'attempt':attempt,'lifetime_optimizer_updates':progress['lifetime_optimizer_updates']}
        append_json(output/'metrics.jsonl',payload);run.log(payload)
    try:
        status('LOADING_MODELS')
        policy=AutoModelForCausalLM.from_pretrained(cfg['student'],dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to(device)
        policy.config.use_cache=False;policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        teacher=AutoModelForCausalLM.from_pretrained(cfg['teacher'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to(device)
        teacher.eval().requires_grad_(False);teacher.config.use_cache=False
        assert policy.config.model_type==teacher.config.model_type=='olmo2'
        assert policy.config.vocab_size==teacher.config.vocab_size
        optimizer=torch.optim.AdamW(policy.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']),weight_decay=0,foreach=False)
        if args.resume:
            restored=restore_checkpoint(args.resume,policy,optimizer,rng,device)
            if restored['config']['profile']!=args.profile or restored['config']['target_updates']!=target or restored['config']['protocol_sha256']!=config['protocol_sha256'] or restored['config']['critical_code_sha256']!=config['critical_code_sha256']:
                raise ValueError('Resume protocol or critical implementation differs')
            progress=restored['progress'];parent=progress.get('attempt_relative','.')
            write_json(output/'resume.json',{'parent_attempt':parent,'checkpoint_update':progress['optimizer_updates'],'manifest':str(args.resume)})
            previous_seconds=progress['cumulative_seconds'];progress['attempt_relative']=attempt;progress['lifetime_optimizer_updates']=lifetime_updates(run_root)
        teacher_probe=probe(teacher)
        access=ROOT/'preflight/wandb-access.json'
        online=access.exists() and json.loads(access.read_text()).get('status')=='PASS'
        run=wandb.init(entity=cfg['wandb_entity'],project=cfg['wandb_project'],group=cfg['experiment'],job_type=args.profile,
            name=f'gkd-opd-{args.profile}-seed0'+('' if attempt=='.' else '-'+Path(attempt).name),config={**config,'attempt':attempt},dir=str(output),mode='online' if online else 'offline')
        progress['wandb_run_id']=run.id
        write_json(output/'wandb-run.json',{'id':run.id,'url':run.url if online else None,'mode':'online' if online else 'offline'})
        run.define_metric('lifetime_optimizer_updates')
        for namespace in ['train','val','system','debug']:run.define_metric(namespace+'/*',step_metric='lifetime_optimizer_updates')
        write_json(run_root/'active-process.json',{'pid':os.getpid(),'profile':args.profile,'attempt_relative':attempt})
        sampler=Sampler(cfg['student'],output,seed=cfg['seed'],gpu=cfg['sampler_gpu'],port=cfg['sampler_port'])
        sampler.start();sampler.init_weight_sync('cuda:0');sampler.sync_policy_weights(policy)
        def generate(examples,seed,scoring=False):
            prompts=[tokenizer.encode(template.format(question=r['question']),add_special_tokens=False) for r in examples]
            assert max(map(len,prompts))+cfg['max_new_tokens']<=cfg['context_length']
            records=[]
            for begin in range(0,len(examples),128):
                payload={'model':cfg['student'],'prompt':prompts[begin:begin+128],'n':1,'temperature':1.,'top_p':1.,'top_k':-1,'min_p':0.,
                    'max_tokens':512,'seed':seed,'stop':['</answer>'],'include_stop_str_in_output':True,'return_token_ids':True}
                if scoring:payload['logprobs']=1
                response=_http_json('POST',sampler.base_url+'/v1/completions',payload,timeout=900)
                choices=sorted(response['choices'],key=lambda c:c['index']);assert len(choices)==len(examples[begin:begin+128])
                for item,prompt,choice in zip(examples[begin:begin+128],prompts[begin:begin+128],choices,strict=True):
                    ids=choice['token_ids'];assert ids and len(ids)<=512
                    row={**item,'prompt_token_ids':prompt,'response_token_ids':ids,'response':choice['text'],'finish_reason':choice.get('finish_reason'),
                        'rewards':r1_zero_reward_fn(choice['text'],item['ground_truth'])}
                    if scoring:
                        row['sampler_log_probs']=choice['logprobs']['token_logprobs'];assert len(row['sampler_log_probs'])==len(ids) and None not in row['sampler_log_probs']
                    records.append(row)
            return records
        def export_model(name,update,metric):
            storage=Path(cfg['checkpoint_model_root']);target_path=storage/name;temporary=storage/('.'+name+'-writing');backup=storage/('.'+name+'-previous')
            if temporary.exists():shutil.rmtree(temporary)
            weights={n:t.detach().cpu().to(torch.bfloat16) for n,t in policy.state_dict().items()}
            policy.save_pretrained(temporary,state_dict=weights,safe_serialization=True);tokenizer.save_pretrained(temporary);del weights
            info={'optimizer_update':update,'validation':metric,'attempt_relative':attempt,'path':str(target_path)}
            write_json(temporary/'checkpoint-info.json',info)
            if backup.exists():shutil.rmtree(backup)
            if target_path.exists():target_path.rename(backup)
            temporary.rename(target_path)
            if backup.exists():shutil.rmtree(backup)
            write_json(run_root/(name+'.json'),info)
        def evaluate(update):
            begun=time.monotonic();rows=generate(validation,cfg['eval_seed']);save_records('validation',f'step-{update:04d}.json.gz',rows)
            metric=stats_for_records(rows);progress['last_validation']=metric
            if metric['reward']>progress['best_val_reward']:
                progress['best_val_reward']=metric['reward'];progress['best_validation_update']=update
                export_model('best-model',update,metric)
            log({'optimizer_update':update,**{'val/'+k:v for k,v in metric.items()},'system/eval_seconds':time.monotonic()-begun})
            print('OPD_EVAL',update,json.dumps(metric),flush=True)
            return metric
        if not args.resume:
            write_json(output/'validation/step-0000-reference.json',{'origin':'reused exact matched original1B baseline','source':str(OLD/'outputs/baseline-3shot/responses.jsonl'),
                'sha256':sha(OLD/'outputs/baseline-3shot/responses.jsonl'),'metrics':initial_val})
            log({'optimizer_update':0,'val/reward':initial_val['reward'],'val/format_reward':initial_val['format_reward'],'val/response_length':initial_val['response_length'],
                'val/truncated_fraction':initial_val['truncated_fraction'],'debug/initial_validation_reused':True})
        recent=[]
        for update in range(progress['optimizer_updates']+1,target+1):
            iteration=time.monotonic();status('SAMPLING',pending_update=update)
            begun=time.monotonic()
            if update>progress['optimizer_updates']+1 or update>1:sampler.sync_policy_weights(policy)
            sync_seconds=time.monotonic()-begun
            begun=time.monotonic();examples=rng.sample(train,cfg['responses_per_update']);rows=generate(examples,cfg['seed']+update*1000,True)
            generation_seconds=time.monotonic()-begun
            save_records('rollouts',f'step-{update:04d}.json.gz',rows)
            append_json(output/'sampling-events.jsonl',{'optimizer_update':update,'generated_responses':len(rows),'generated_tokens':sum(len(r['response_token_ids']) for r in rows)})
            pack=pack_prompt_and_response_ids([r['prompt_token_ids'] for r in rows],[r['response_token_ids'] for r in rows],tokenizer.pad_token_id)
            assert all(pack['labels'][i][pack['response_mask'][i]].tolist()==r['response_token_ids'] for i,r in enumerate(rows))
            denominator=int(pack['response_mask'].sum());lr=warmup_lr(update,cfg['learning_rate'],cfg['warmup_updates'])
            for group in optimizer.param_groups:group['lr']=lr
            policy.train();optimizer.zero_grad(set_to_none=True);before=probe(policy);loss_sum=0.;details=[]
            torch.cuda.reset_peak_memory_stats();begun=time.monotonic();status('TRAINING',pending_update=update)
            for begin in range(0,len(rows),cfg['microbatch_size']):
                end=begin+cfg['microbatch_size'];mask=pack['response_mask'][begin:end].to(device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,info=response_objective(policy,teacher,pack['input_ids'][begin:end].to(device),pack['labels'][begin:end].to(device),mask,denominator,cfg['head_chunk_tokens'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-distribution KL loss')
                loss.backward();loss_sum+=float(loss.detach());details.append(info);del loss,info
            details=torch.cat(details);sampler_lp=torch.tensor([x for r in rows for x in r['sampler_log_probs']],dtype=torch.float32)
            difference=(details[:,3]-sampler_lp).abs();audit={'mean_abs':float(difference.mean()),'p99_abs':float(difference.quantile(.99)),'max_abs':float(difference.max()),'tokens':denominator}
            write_json(output/'sync-audits'/f'step-{update:04d}.json',audit)
            if audit['mean_abs']>cfg['sampler_audit_mean_abs_limit'] or audit['p99_abs']>cfg['sampler_audit_p99_abs_limit']:
                raise RuntimeError('HF/vLLM student scoring audit exceeded declared tolerances')
            norm=float(torch.nn.utils.clip_grad_norm_(policy.parameters(),cfg['max_grad_norm'],error_if_nonfinite=True))
            assert all(p.grad is None for p in teacher.parameters())
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            changed=probe(policy)!=before;teacher_unchanged=probe(teacher)==teacher_probe
            if not teacher_unchanged:raise RuntimeError('Frozen teacher probe changed')
            progress['optimizer_updates']=update;progress['lifetime_optimizer_updates']+=1
            progress['nonzero_gradient_updates']+=int(norm>0);progress['changed_parameter_updates']+=int(changed)
            progress['generated_responses']+=len(rows);progress['generated_tokens']+=denominator
            train_seconds=time.monotonic()-begun;offset=0
            for row in rows:
                n=len(row['response_token_ids']);info=details[offset:offset+n];offset+=n
                row['token_reverse_kl']=info[:,0].tolist();row['student_sampled_log_probs']=info[:,3].tolist();row['teacher_sampled_log_probs']=info[:,4].tolist()
            save_records('training-details',f'step-{update:04d}.json.gz',rows)
            metric={'optimizer_update':update,'train/loss':loss_sum,'train/reverse_kl':float(details[:,0].mean()),
                'train/reverse_kl_p95':float(details[:,0].quantile(.95)),'train/student_entropy':float(details[:,1].mean()),'train/teacher_entropy':float(details[:,2].mean()),
                'train/grad_norm':norm,'train/lr':lr,**{'train/'+k:v for k,v in stats_for_records(rows).items()},
                'debug/sampler_hf_mean_abs':audit['mean_abs'],'debug/sampler_hf_p99_abs':audit['p99_abs'],
                'debug/parameter_probe_changed':changed,'debug/teacher_probe_unchanged':teacher_unchanged,
                'system/generation_seconds':generation_seconds,'system/train_seconds':train_seconds,'system/weight_sync_seconds':sync_seconds,
                'system/step_seconds':time.monotonic()-iteration,'system/trainer_peak_gib':torch.cuda.max_memory_allocated()/2**30}
            append_json(output/'updates.jsonl',{'optimizer_update':update,'lifetime_optimizer_updates':progress['lifetime_optimizer_updates'],
                'loss':loss_sum,'grad_norm':norm,'lr':lr,'parameter_probe_changed':changed,'teacher_probe_unchanged':teacher_unchanged,
                'response_tokens':denominator,'teacher_gradients_absent':True,'sampler_audit':audit})
            log(metric);recent.append(metric);recent=recent[-5:]
            progress['cumulative_seconds']=previous_seconds+time.monotonic()-started
            status('TRAINING',metrics=metric);print('OPD_UPDATE',update,'/',target,json.dumps(metric),flush=True)
            del pack,details,rows,sampler_lp
            if formal and (update%cfg['eval_every_updates']==0 or update==target):
                sampler.sync_policy_weights(policy);evaluate(update)
            if update==1 or update%cfg['checkpoint_every_updates']==0 or update==target:
                status('SAVING_CHECKPOINT');progress['cumulative_seconds']=previous_seconds+time.monotonic()-started
                checkpoint=save_checkpoint(ROOT,output,cfg['checkpoint_model_root'],args.profile,policy,optimizer,rng,progress,config,device)
                if not formal:
                    expected_probe=probe(policy);expected_steps=[float(s['step']) for s in optimizer.state.values() if 'step' in s]
                    with torch.no_grad():next(policy.parameters()).flatten()[0].add_(.01)
                    restore_checkpoint(checkpoint['manifest_path'],policy,optimizer,rng,device)
                    assert probe(policy)==expected_probe and all(float(s['step'])==expected_steps[i] for i,s in enumerate(optimizer.state.values()) if 'step' in s)
                    write_json(output/'checkpoint-restore-check.json',{'status':'PASS','optimizer_updates':1,'full_model_and_adam_loaded':True,'model_probe_restored':True})
                    sampler.sync_policy_weights(policy)
                    post=generate(train[:2],91111);save_records('post-sync','two-examples.json.gz',post)
                    write_json(output/'post-sync-check.json',{'status':'PASS','examples':len(post),'optimizer_updates':1})
            if formal and update%cfg['eval_every_updates']==0:
                vals=[r['val/reward'] for r in canonical_rows(run_root,'metrics.jsonl') if 'val/reward' in r and r['optimizer_update']>0]
                severe=len(vals)>=2 and max(vals[-2:])<initial_val['reward']/4
                broken=len(recent)>=5 and all(r['train/truncated_fraction']>=.95 or (r['train/reward']==0 and r['train/grad_norm']==0) for r in recent)
                if severe and broken:raise QualityStop('Two validations below baseline/4 and five updates with lost signal or >=95% truncation')
        if formal:export_model('final-model',target,progress['last_validation'])
        summary={'status':'COMPLETE','method':cfg['method'],'profile':args.profile,'seed':0,'optimizer_updates':progress['optimizer_updates'],
            'lifetime_optimizer_updates':progress['lifetime_optimizer_updates'],'generated_responses':progress['generated_responses'],'generated_tokens':progress['generated_tokens'],
            'initial_validation':initial_val,'initial_validation_reused':True,'last_validation':progress['last_validation'],
            'best_val_reward':progress['best_val_reward'],'best_validation_update':progress['best_validation_update'],
            'nonzero_gradient_updates':progress['nonzero_gradient_updates'],'changed_parameter_updates':progress['changed_parameter_updates'],
            'elapsed_seconds':previous_seconds+time.monotonic()-started,'wandb_mode':'online' if online else 'offline','wandb_run_id':run.id,
            'active_attempt':attempt,'checkpoint_manifest':str(ROOT/'monitoring/latest-recoverable-checkpoint.json')}
        write_json(run_root/'completed.json',summary);run.summary.update(summary);complete=True;status('COMPLETE',summary=summary)
        print('OPD_COMPLETE',json.dumps(summary),flush=True)
    except BaseException as error:
        quality_stopped=isinstance(error,QualityStop)
        failure={'status':'INTERRUPTED' if quality_stopped else 'FAILED','type':type(error).__name__,'message':str(error),
            'optimizer_updates':progress['optimizer_updates'],'lifetime_optimizer_updates':progress['lifetime_optimizer_updates'],
            'monitoring_policy_stop':quality_stopped,'attempt_relative':attempt}
        write_json(output/'failure.json',failure)
        if quality_stopped:write_json(run_root/'interrupted.json',failure)
        status(failure['status'],error=str(error));raise
    finally:
        if sampler is not None:sampler.stop()
        if run is not None:run.finish(exit_code=0 if complete else 1)


if __name__=='__main__':main()
