"""Exact full-vocabulary conditional reverse KL on student-induced prefixes."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def categorical_reverse_kl(student_logits, teacher_logits):
    dtype=torch.float64 if student_logits.dtype==torch.float64 else torch.float32
    log_p=F.log_softmax(student_logits.to(dtype),dim=-1)
    with torch.no_grad():
        log_q=F.log_softmax(teacher_logits.detach().to(dtype),dim=-1)
    p=log_p.exp()
    kl=(p*(log_p-log_q)).sum(-1)
    with torch.no_grad():
        entropy_s=-(p*log_p).sum(-1)
        entropy_t=-(log_q.exp()*log_q).sum(-1)
    return kl,entropy_s,entropy_t,log_p,log_q


def response_objective(student,teacher,input_ids,labels,mask,denominator,chunk_size=128):
    if input_ids.shape!=labels.shape or labels.shape!=mask.shape:
        raise ValueError('Shifted inputs, labels and response mask must align')
    if denominator<=0 or chunk_size<=0:raise ValueError('Invalid token denominator/chunk size')
    mask=mask.bool()
    with torch.no_grad():
        teacher_hidden=teacher.model(input_ids=input_ids,use_cache=False).last_hidden_state[mask]
    student_hidden=student.model(input_ids=input_ids,use_cache=False).last_hidden_state[mask]
    targets=labels[mask]
    if not len(targets):raise ValueError('Microbatch has no response tokens')
    def project(s,t,y):
        s_logits=student.lm_head(s)
        with torch.no_grad():t_logits=teacher.lm_head(t)
        kl,h_s,h_t,log_p,log_q=categorical_reverse_kl(s_logits,t_logits)
        with torch.no_grad():
            info=torch.stack([kl.detach(),h_s,h_t,log_p.gather(-1,y[:,None]).squeeze(-1),log_q.gather(-1,y[:,None]).squeeze(-1)],dim=-1)
        return kl.sum(),info
    sums=[];statistics=[]
    for begin in range(0,len(targets),chunk_size):
        args=(student_hidden[begin:begin+chunk_size],teacher_hidden[begin:begin+chunk_size],targets[begin:begin+chunk_size])
        total,info=checkpoint(project,*args,use_reentrant=False,preserve_rng_state=False) if torch.is_grad_enabled() else project(*args)
        sums.append(total);statistics.append(info)
    # The caller uses the same complete optimizer-batch denominator in every microbatch.
    return torch.stack(sums).sum()/denominator,torch.cat(statistics).detach().cpu()


def warmup_lr(update,base=1e-6,warmup=80):
    return base*min(update/max(warmup,1),1.)
