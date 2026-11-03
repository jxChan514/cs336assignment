"""Single-seed DAPO supplement, using frozen A5 data/prompt/scorer resources."""
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import write_json, append_json, sha, data_rows, stats_for_records
from dapo import shaped_rewards, normalized_advantages, select_mixed_groups, token_loss, warmup_lr
from checkpoints import save_recoverable, restore_state
from history import lifetime_work
from reference.grpo import pack_prompt_and_response_ids
from reference.drgrpo_grader import r1_zero_reward_fn
from reference.vllm_utils import _http_json
from scoring import response_log_probs
from server import Sampler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['smoke', 'formal'], default='formal')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    root = code.parent
    cfg = json.loads((root/'protocol.json').read_text())
    formal = args.profile == 'formal'
    steps = cfg['rollout_steps'] if formal else 2
    output = args.output.resolve()
    if (output/'config.json').exists() and args.resume is None:
        raise FileExistsError('Existing run preserved; explicit --resume checkpoint is required')
    output.mkdir(parents=True, exist_ok=True)
    run_output = output
    attempt_relative = '.'
    data_root = Path('/root/autodl-tmp/cs336-a5-20261029/data/gsm8k')
    model_path = json.loads((code/'env-spec.json').read_text())['model']
    train = data_rows(data_root/'train.jsonl', cfg['train_examples'])
    validation = data_rows(data_root/'test.jsonl', cfg['val_examples'] if formal else 32)
    template = (code/'prompt.prompt').read_text()
    assert template.count('User:') == 4 and not any(x['question'] in template for x in train+validation)
    config = {**cfg, 'profile': args.profile, 'effective_rollout_steps': steps,
              'model': model_path, 'actual_validation_examples': len(validation),
              'precision': 'FP32 parameters/Adam; BF16 autocast/inference',
              'entropy_population': 'all accepted response tokens; candidate entropy not measured',
              'zero_advantage_pruning': False, 'checkpoint_enabled': formal,
              'train_ids': [x['id'] for x in train], 'val_ids': [x['id'] for x in validation],
              'prompt_sha256': sha(code/'prompt.prompt'), 'train_sha256': sha(data_root/'train.jsonl'),
              'val_sha256': sha(data_root/'test.jsonl'), 'protocol_sha256': sha(root/'protocol.json'),
              'code_sha256': {x.name: sha(x) for x in code.glob('*.py')}}
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    rng = random.Random(cfg['seed'])
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    state = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if state:
        if state['config']['protocol_sha256'] != config['protocol_sha256']:
            raise ValueError('Resume protocol differs from saved checkpoint')
        if state['config']['prompt_sha256'] != config['prompt_sha256']:
            raise ValueError('Resume prompt differs from saved checkpoint')
        index = 1
        while (run_output/'attempts'/f'attempt-{index:04d}').exists():index += 1
        output = run_output/'attempts'/f'attempt-{index:04d}'
        output.mkdir(parents=True)
        attempt_relative = str(output.relative_to(run_output))
        write_json(output/'resume.json', {'parent_attempt': state['progress'].get('attempt_relative', '.'),
            'checkpoint_rollout_step': state['progress']['rollout_step'],
            'checkpoint_optimizer_updates': state['progress']['optimizer_updates'],
            'checkpoint_path': str(args.resume.resolve()), 'checkpoint_sha256': sha(args.resume)})
        write_json(output/'config.json', config)
    else:
        write_json(output/'config.json', config)
    run = None; sampler = None; completed = False
    start_time = time.monotonic()
    progress = state['progress'] if state else {'rollout_step': 0, 'optimizer_updates': 0,
        'total_generated_responses': 0, 'total_generated_tokens': 0, 'total_candidate_groups': 0,
        'best_val_reward': -1., 'best_validation_step': None, 'initial_validation': None,
        'last_validation': None, 'nonzero_gradient_updates': 0, 'changed_parameter_updates': 0,
        'cumulative_wall_seconds': 0.}
    progress['attempt_relative'] = attempt_relative
    progress.update(lifetime_work(run_output))
    saved_best = run_output/'best-model/checkpoint-info.json'
    if state and saved_best.exists():
        best = json.loads(saved_best.read_text())
        if best['validation']['reward'] > progress['best_val_reward']:
            progress['best_val_reward'] = best['validation']['reward']
            progress['best_validation_step'] = best['rollout_step']
    previous_seconds = progress['cumulative_wall_seconds']
    current_step = progress['rollout_step']
    def terminate(number, frame):
        raise SystemExit(128+number)
    signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
    def records_file(folder, name, records):
        path = output/folder/name; path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'wt', encoding='utf-8', compresslevel=5) as f:
            json.dump(records, f, ensure_ascii=False)
    def log(payload):
        payload = {**payload, 'attempt': attempt_relative, 'lifetime_optimizer_updates': progress['lifetime_optimizer_updates']}
        append_json(output/'metrics.jsonl', payload); run.log(payload)
    def live_status(status, **values):
        write_json(run_output/'progress.json', {'status': status, 'attempt_relative': attempt_relative, 'rollout_step': current_step,
            'last_completed_rollout': progress['rollout_step'], 'target_steps': steps,
            'optimizer_updates': progress['optimizer_updates'],
            'lifetime_optimizer_updates': progress['lifetime_optimizer_updates'], **values})
    try:
        init = {'id': progress['wandb_run_id'], 'resume': 'must'} if state else {}
        run = wandb.init(entity=cfg['wandb_entity'], project=cfg['wandb_project'],
            group=cfg['experiment'], job_type=args.profile, name=f'dapo-{args.profile}-seed0',
            config=config, dir=str(output), mode='online', **init)
        progress['wandb_run_id'] = run.id
        run.define_metric('rollout_step')
        run.define_metric('lifetime_optimizer_updates')
        for namespace in ['train', 'val', 'sampling', 'system', 'debug']:
            run.define_metric(namespace+'/*', step_metric='lifetime_optimizer_updates')
        write_json(run_output/'wandb-run.json', {'id': run.id, 'url': run.url})
        write_json(run_output/'active-attempt.json', {'attempt_relative': attempt_relative})
        print('DAPO_RUN', run.url, flush=True)
        policy = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32,
            attn_implementation='sdpa', local_files_only=True).to(device)
        policy.config.use_cache = False
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg['learning_rate'],
            betas=tuple(cfg['adam_betas']), weight_decay=cfg['weight_decay'], foreach=False)
        if state:
            progress = restore_state(state, policy, optimizer, rng, device)
            del state
        probe_tensors = [p for p in policy.parameters() if p.numel() >= 4096]
        probes = [probe_tensors[0], probe_tensors[len(probe_tensors)//2], probe_tensors[-1]]
        def probe():
            values = [p.detach().flatten()[::max(p.numel()//2048, 1)][:2048].float().cpu() for p in probes]
            return hashlib.sha256(torch.cat(values).numpy().tobytes()).hexdigest()
        def score(ids, labels, mask, entropy=False):
            return response_log_probs(policy, ids, labels, mask, entropy, head_chunk_size=512)
        def export_model(name, step, metric):
            temporary = run_output/('.'+name+'-writing'); target = run_output/name; backup = run_output/('.'+name+'-previous')
            if temporary.exists():shutil.rmtree(temporary)
            weights = {n: t.detach().cpu().to(torch.bfloat16) for n, t in policy.state_dict().items()}
            policy.save_pretrained(temporary, state_dict=weights, safe_serialization=True)
            tokenizer.save_pretrained(temporary)
            write_json(temporary/'checkpoint-info.json', {'rollout_step': step, 'attempt_relative': attempt_relative,
                'optimizer_updates': progress['optimizer_updates'], 'validation': metric})
            del weights
            if backup.exists():shutil.rmtree(backup)
            if target.exists():target.rename(backup)
            temporary.rename(target)
            if backup.exists():shutil.rmtree(backup)
        sampler = Sampler(model_path, output, seed=cfg['seed'], port=cfg['sampler_port'])
        sampler.start(); sampler.init_weight_sync('cuda:0')
        def generate(examples, n, seed, record_log_probs=False):
            prompts = [tokenizer.encode(template.format(question=x['question']), add_special_tokens=False) for x in examples]
            assert max(map(len, prompts))+cfg['max_new_tokens'] <= 2048
            records = []
            for start in range(0, len(examples), 128):
                payload = {'model': model_path, 'prompt': prompts[start:start+128], 'n': n,
                    'temperature': cfg['temperature'], 'top_p': cfg['top_p'], 'top_k': -1, 'min_p': 0.,
                    'max_tokens': cfg['max_new_tokens'], 'seed': seed, 'stop': ['</answer>'],
                    'include_stop_str_in_output': True, 'return_token_ids': True}
                if record_log_probs:payload['logprobs'] = 1
                response = _http_json('POST', sampler.base_url+'/v1/completions', payload, timeout=900)
                choices = sorted(response['choices'], key=lambda x: x['index'])
                assert len(choices) == len(examples[start:start+128])*n
                for j, example in enumerate(examples[start:start+128]):
                    for k in range(n):
                        choice = choices[j*n+k]; tokens = choice.get('token_ids')
                        assert tokens and len(tokens) <= cfg['max_new_tokens']
                        row = {**example, 'sample_index': k, 'prompt_token_ids': prompts[start+j],
                            'response': choice['text'], 'response_token_ids': tokens,
                            'finish_reason': choice.get('finish_reason'),
                            'rewards': r1_zero_reward_fn(choice['text'], example['ground_truth'])}
                        if record_log_probs:
                            lp = choice['logprobs']['token_logprobs']
                            if len(lp) != len(tokens) or any(x is None for x in lp):
                                raise RuntimeError('Sampler token IDs/log-probabilities do not align')
                            row['sampler_log_probs'] = lp
                        records.append(row)
            return records
        def evaluate(step):
            started = time.monotonic(); rows = generate(validation, 1, cfg['eval_seed'])
            records_file('validation', f'step-{step:04d}.json.gz', rows)
            metric = stats_for_records(rows)
            progress['last_validation'] = metric
            if progress['initial_validation'] is None:progress['initial_validation'] = metric
            if metric['reward'] > progress['best_val_reward']:
                progress['best_val_reward'] = metric['reward']; progress['best_validation_step'] = step
                if formal:export_model('best-model', step, metric)
            log({'rollout_step': step, 'optimizer_updates': progress['optimizer_updates'],
                 **{'val/'+k: v for k, v in metric.items()}, 'system/eval_seconds': time.monotonic()-started})
            run.log({'rollout_step': step, 'samples/validation': wandb.Table(
                columns=['id', 'response', 'reward', 'tokens'], data=[[r['id'], r['response'],
                r['rewards']['reward'], len(r['response_token_ids'])] for r in rows[:12]])})
            print('EVAL', step, metric, flush=True)
        sampler.sync_policy_weights(policy)
        if not args.resume:evaluate(0)
        for step in range(progress['rollout_step']+1, steps+1):
            current_step = step; iteration_started = time.monotonic()
            live_status('SAMPLING')
            phase = time.monotonic()
            if step > 1:sampler.sync_policy_weights(policy)
            sync_seconds = time.monotonic()-phase
            phase = time.monotonic(); kept = []; candidates = []
            candidate_prompts = rng.sample(train, cfg['max_candidate_groups'])
            group_counts = {'all_wrong': 0, 'all_correct': 0, 'mixed': 0}
            waves = 0
            for start in range(0, len(candidate_prompts), cfg['generation_prompt_batch']):
                waves += 1
                rows = generate(candidate_prompts[start:start+cfg['generation_prompt_batch']],
                    cfg['group_size'], cfg['seed']+step*1000+waves, True)
                for j, row in enumerate(rows):row['candidate_group_id'] = f'{step}:{waves}:{j//8}'
                accepted, counts = select_mixed_groups(rows, cfg['group_size'])
                for k, v in counts.items():group_counts[k] += v
                selected = accepted[:cfg['accepted_prompt_groups']*cfg['group_size']-len(kept)]
                selected_groups = {r['candidate_group_id'] for r in selected}
                for row in rows:row['selected_for_training'] = row['candidate_group_id'] in selected_groups
                kept.extend(selected); candidates.extend(rows)
                records_file('candidates', f'step-{step:04d}-batch-{waves:02d}.json.gz', rows)
                event = {'rollout_step': step, 'wave': waves, 'generated_responses': len(rows),
                         'generated_tokens': sum(len(r['response_token_ids']) for r in rows)}
                append_json(output/'sampling-events.jsonl', event)
                progress['lifetime_generated_responses'] += event['generated_responses']
                progress['lifetime_generated_tokens'] += event['generated_tokens']
                live_status('SAMPLING', generation_batches=waves, accepted_groups=len(kept)//8,
                            candidate_groups=len(candidates)//8)
                if len(kept) == 256:break
            generated_tokens = sum(len(r['response_token_ids']) for r in candidates)
            progress['total_generated_responses'] += len(candidates)
            progress['total_generated_tokens'] += generated_tokens
            progress['total_candidate_groups'] += len(candidates)//8
            sample_metrics = {'sampling/task_reward': sum(r['rewards']['reward'] for r in candidates)/len(candidates),
                'sampling/candidate_groups': len(candidates)//8, 'sampling/accepted_groups': len(kept)//8,
                'sampling/mixed_group_fraction': group_counts['mixed']/(len(candidates)//8),
                'sampling/all_wrong_groups': group_counts['all_wrong'], 'sampling/all_correct_groups': group_counts['all_correct'],
                'sampling/generation_batches': waves, 'sampling/generated_responses': len(candidates),
                'sampling/generated_tokens': generated_tokens,
                'sampling/candidate_truncated_fraction': sum(r['finish_reason']=='length' for r in candidates)/len(candidates),
                'sampling/total_generated_responses': progress['total_generated_responses'],
                'sampling/total_generated_tokens': progress['total_generated_tokens']}
            if len(kept) != 256:
                log({'rollout_step': step, **sample_metrics})
                raise RuntimeError('Dynamic sampling exhausted256 candidate groups without32 mixed groups')
            records_file('rollouts', f'step-{step:04d}.json.gz', kept)
            generation_seconds = time.monotonic()-phase
            task = torch.tensor([r['rewards']['reward'] for r in kept])
            shaped, penalties = shaped_rewards(task, [len(r['response_token_ids']) for r in kept],
                cfg['max_new_tokens'], cfg['overlong_cache'])
            advantages = normalized_advantages(shaped, cfg['group_size'])
            pack = pack_prompt_and_response_ids([r['prompt_token_ids'] for r in kept],
                [r['response_token_ids'] for r in kept], tokenizer.pad_token_id)
            for i, r in enumerate(kept):assert pack['labels'][i][pack['response_mask'][i]].tolist() == r['response_token_ids']
            phase = time.monotonic(); policy.eval(); old = torch.zeros(pack['labels'].shape)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for begin in range(0, 256, 8):
                    result = score(pack['input_ids'][begin:begin+8].to(device),
                        pack['labels'][begin:begin+8].to(device), pack['response_mask'][begin:begin+8].to(device))
                    old[begin:begin+8] = result['log_probs'].cpu()
            sampler_scores = torch.zeros_like(old)
            for i, r in enumerate(kept):sampler_scores[i][pack['response_mask'][i]] = torch.tensor(r['sampler_log_probs'])
            difference = (old-sampler_scores)[pack['response_mask']].abs()
            audit = {'mean_abs': float(difference.mean()), 'p99_abs': float(difference.quantile(.99)),
                     'max_abs': float(difference.max()), 'tokens': len(difference)}
            write_json(output/'sync-audits'/f'step-{step:04d}.json', audit)
            if audit['mean_abs'] > cfg['sampler_score_audit_mean_abs_limit'] or audit['p99_abs'] > cfg['sampler_score_audit_p99_abs_limit']:
                raise RuntimeError('HF/vLLM scoring differs beyond declared sync-audit tolerances: '+str(audit))
            old_hash = hashlib.sha256(old.numpy().tobytes()).hexdigest()
            (output/'old_log_probs').mkdir(exist_ok=True)
            torch.save({'log_probs': old, 'advantages': advantages, 'task_rewards': task,
                'shaped_rewards': shaped, 'response_mask': pack['response_mask'], 'sha256': old_hash},
                output/'old_log_probs'/f'step-{step:04d}.pt')
            old_seconds = time.monotonic()-phase
            order = list(range(256)); random.Random(cfg['seed']+1000000+step).shuffle(order)
            policy.train(); updates_before = progress['optimizer_updates']; all_meta = []; losses = []; norms = []
            entropy_sum = 0.; entropy_count = 0; train_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            for begin in range(0, 256, cfg['train_batch_size']):
                selected = order[begin:begin+cfg['train_batch_size']]
                denominator = int(pack['response_mask'][selected].sum())
                lr = warmup_lr(progress['optimizer_updates']+1, cfg['learning_rate'],
                               cfg['warmup_rollouts']*cfg['optimizer_updates_per_rollout'])
                for group in optimizer.param_groups:group['lr'] = lr
                before = probe(); optimizer.zero_grad(set_to_none=True); batch_loss = 0.; batch_meta = []
                for mb in range(0, len(selected), cfg['microbatch_size']):
                    ids = selected[mb:mb+cfg['microbatch_size']]; mask = pack['response_mask'][ids].to(device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = score(pack['input_ids'][ids].to(device), pack['labels'][ids].to(device), mask, True)
                        loss, meta = token_loss(result['log_probs'], old[ids].to(device),
                            advantages[ids].to(device), mask, denominator, cfg['clip_low'], cfg['clip_high'])
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    loss.backward(); batch_loss += float(loss.detach()); batch_meta.append(meta)
                    entropy_sum += float(result['token_entropy'].masked_fill(~mask, 0.).sum())
                    entropy_count += int(mask.sum()); del result, loss
                norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True))
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                progress['optimizer_updates'] += 1
                progress['lifetime_optimizer_updates'] += 1
                changed = before != probe()
                progress['nonzero_gradient_updates'] += int(norm > 0)
                progress['changed_parameter_updates'] += int(changed)
                ratios = torch.cat([x['ratio_values'] for x in batch_meta]); tokens = sum(x['tokens'] for x in batch_meta)
                meta = {'ratio_min': float(ratios.min()), 'ratio_max': float(ratios.max()),
                    'ratio_p99': float(ratios.quantile(.99)), 'ratio_mean': float(ratios.mean()),
                    'clip_fraction': sum(x['clipped_tokens'] for x in batch_meta)/tokens,
                    'ratio_outside_fraction': sum(x['outside_tokens'] for x in batch_meta)/tokens,
                    'approx_kl_k3': sum(x['kl_k3_sum'] for x in batch_meta)/tokens, 'response_tokens': tokens}
                append_json(output/'updates.jsonl', {'rollout_step': step,
                    'optimizer_update': progress['optimizer_updates'],
                    'lifetime_optimizer_updates': progress['lifetime_optimizer_updates'], 'sample_indices': selected,
                    'loss': batch_loss, 'grad_norm': norm, 'lr': lr, 'parameter_probe_changed': changed,
                    'old_log_probs_sha256': old_hash, **meta})
                live_status('TRAINING', update_in_rollout=progress['optimizer_updates']-updates_before)
                all_meta.append(meta); losses.append(batch_loss); norms.append(norm)
            assert progress['optimizer_updates']-updates_before == 4
            assert hashlib.sha256(old.numpy().tobytes()).hexdigest() == old_hash
            progress['rollout_step'] = step
            progress['cumulative_wall_seconds'] = previous_seconds+time.monotonic()-start_time
            token_count = sum(x['response_tokens'] for x in all_meta)
            metrics = {'rollout_step': step, 'optimizer_updates': progress['optimizer_updates'],
                **{'train/'+k: v for k, v in stats_for_records(kept, 8).items()}, **sample_metrics,
                'train/shaped_reward': float(shaped.mean()), 'train/length_penalty': float(penalties.mean()),
                'train/loss': float(np.mean(losses)), 'train/grad_norm': float(np.mean(norms)),
                'train/max_grad_norm': max(norms), 'train/lr': lr,
                'train/token_entropy': entropy_sum/entropy_count,
                'train/clip_fraction': sum(x['clip_fraction']*x['response_tokens'] for x in all_meta)/token_count,
                'train/approx_kl_k3': sum(x['approx_kl_k3']*x['response_tokens'] for x in all_meta)/token_count,
                'train/max_ratio': max(x['ratio_max'] for x in all_meta),
                'debug/sampler_hf_mean_abs': audit['mean_abs'], 'debug/sampler_hf_p99_abs': audit['p99_abs'],
                'system/step_seconds': time.monotonic()-iteration_started,
                'system/generation_seconds': generation_seconds, 'system/train_seconds': time.monotonic()-train_started,
                'system/old_logprob_seconds': old_seconds, 'system/weight_sync_seconds': sync_seconds,
                'system/actor_peak_gib': torch.cuda.max_memory_allocated()/2**30,
                'system/cumulative_wall_seconds': progress['cumulative_wall_seconds']}
            log(metrics); live_status('TRAINING', metrics=metrics)
            print('TRAIN_DAPO', step, '/', steps, 'val_previous', progress['last_validation']['reward'],
                  'accepted_reward', metrics['train/reward'], 'candidate_reward', metrics['sampling/task_reward'],
                  'seconds', metrics['system/step_seconds'], flush=True)
            del pack, old, advantages, sampler_scores, kept, candidates
            if step % cfg['eval_every_rollouts'] == 0 or step == steps:
                sampler.sync_policy_weights(policy); evaluate(step)
            if formal and (step % cfg['checkpoint_every_rollouts'] == 0 or step == steps):
                live_status('SAVING_CHECKPOINT')
                progress['cumulative_wall_seconds'] = previous_seconds+time.monotonic()-start_time
                save_recoverable(run_output, policy, optimizer, rng, progress, config, device)
        assert progress['optimizer_updates'] == steps*4
        assert progress['nonzero_gradient_updates'] > 0 and progress['changed_parameter_updates'] > 0
        if formal:export_model('final-model', steps, progress['last_validation'])
        summary = {'status': 'PASS', 'method': 'dapo', 'profile': args.profile, 'seed': 0,
            'completed_rollout_steps': progress['rollout_step'], 'optimizer_updates': progress['optimizer_updates'],
            'initial_validation': progress['initial_validation'], 'final_validation': progress['last_validation'],
            'best_val_reward': progress['best_val_reward'], 'best_validation_step': progress['best_validation_step'],
            'nonzero_gradient_updates': progress['nonzero_gradient_updates'],
            'changed_parameter_updates': progress['changed_parameter_updates'],
            'generated_responses': progress['total_generated_responses'], 'generated_tokens': progress['total_generated_tokens'],
            'elapsed_seconds': previous_seconds+time.monotonic()-start_time, 'wandb_url': run.url,
            'recoverable_checkpoint_enabled': formal, 'inference_models_saved': formal,
            'active_attempt': attempt_relative, **lifetime_work(run_output),
            'best_metric_scope': 'best observed across all preserved attempts; canonical peak computed separately'}
        write_json(run_output/'completed.json', summary); run.summary.update(summary)
        completed = True; live_status('COMPLETE', summary=summary)
        print('DAPO_COMPLETE', json.dumps(summary), flush=True)
    except BaseException as error:
        write_json(output/'failure.json', {'type': type(error).__name__, 'message': str(error),
            'rollout_step': current_step, 'last_completed_rollout': progress['rollout_step'],
            'optimizer_updates': progress['optimizer_updates']})
        write_json(run_output/'active-failure.json', json.loads((output/'failure.json').read_text()))
        live_status('FAILED', error=str(error))
        raise
    finally:
        if sampler is not None:sampler.stop()
        if run is not None:run.finish(exit_code=0 if completed else 1)


if __name__ == '__main__':
    main()
