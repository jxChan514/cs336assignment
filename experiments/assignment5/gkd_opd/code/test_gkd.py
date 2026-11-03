"""Small independent categorical-KL oracle, accumulation, and resume checks."""
from pathlib import Path
from types import SimpleNamespace
import random
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical,kl_divergence
from gkd import categorical_reverse_kl,response_objective
from checkpoints import save_checkpoint,restore_checkpoint
from reference.grpo import pack_prompt_and_response_ids


class Backbone(nn.Module):
    def __init__(self):super().__init__();self.embedding=nn.Embedding(17,7).double()
    def forward(self,input_ids,use_cache=False):return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class Toy(nn.Module):
    def __init__(self):super().__init__();self.model=Backbone();self.lm_head=nn.Linear(7,17).double()


class Checks(unittest.TestCase):
    def test_categorical_oracle_and_teacher_frozen(self):
        torch.manual_seed(4)
        student=torch.randn(2,3,5,dtype=torch.float64,requires_grad=True)
        teacher=torch.randn_like(student,requires_grad=True)
        actual=categorical_reverse_kl(student,teacher)[0].sum()
        expected=kl_divergence(Categorical(logits=student),Categorical(logits=teacher.detach())).sum()
        a=torch.autograd.grad(actual,(student,teacher),allow_unused=True,retain_graph=True)
        e=torch.autograd.grad(expected,student)[0]
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(a[0],e,rtol=1e-11,atol=1e-12)
        self.assertIsNone(a[1])
        self.assertTrue(torch.autograd.gradcheck(lambda x:categorical_reverse_kl(x,teacher)[0].sum(),(student,)))

    def test_chunked_response_loss_and_uneven_accumulation(self):
        torch.manual_seed(6);student=Toy();teacher=Toy().requires_grad_(False)
        ids=torch.randint(0,17,(3,5));labels=torch.randint(0,17,(3,5))
        mask=torch.tensor([[0,1,1,0,0],[0,1,1,1,1],[0,0,1,0,0]],dtype=torch.bool);denominator=int(mask.sum())
        s=student.lm_head(student.model(ids).last_hidden_state)
        with torch.no_grad():t=teacher.lm_head(teacher.model(ids).last_hidden_state)
        oracle=kl_divergence(Categorical(logits=s),Categorical(logits=t)).masked_fill(~mask,0).sum()/denominator
        expected=torch.autograd.grad(oracle,tuple(student.parameters()))
        losses=[]
        for begin,end in [(0,1),(1,3)]:
            loss,meta=response_objective(student,teacher,ids[begin:end],labels[begin:end],mask[begin:end],denominator,1)
            losses.append(float(loss.detach()));loss.backward()
            self.assertEqual(len(meta),int(mask[begin:end].sum()))
        self.assertAlmostEqual(sum(losses),float(oracle.detach()),places=11)
        for p,e in zip(student.parameters(),expected,strict=True):torch.testing.assert_close(p.grad,e,rtol=1e-10,atol=1e-11)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_shift_and_true_eos_pad_alias(self):
        batch=pack_prompt_and_response_ids([[1,2],[3]],[[4,0],[5]],0)
        self.assertEqual(batch['input_ids'].tolist(),[[1,2,4],[3,5,0]])
        self.assertEqual(batch['labels'].tolist(),[[2,4,0],[5,0,0]])
        self.assertEqual(batch['response_mask'].tolist(),[[False,True,True],[True,False,False]])

    def test_split_checkpoint_restores_adam_rng_and_next_update(self):
        torch.manual_seed(7);random.seed(7);np.random.seed(7);rng=random.Random(19)
        model=nn.Linear(3,2).double();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.randn(4,3,dtype=torch.float64)
        model(x).square().mean().backward();optimizer.step();optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'outputs/run';output.mkdir(parents=True);(root/'monitoring').mkdir()
            progress={'optimizer_updates':1};config={'protocol_sha256':'toy'}
            manifest=save_checkpoint(root,output,root/'model-storage','smoke',model,optimizer,rng,progress,config)
            expected_random=(random.random(),np.random.random(),torch.rand(1),rng.random())
            model(x).square().mean().backward();optimizer.step();optimizer.zero_grad()
            expected={n:p.detach().clone() for n,p in model.named_parameters()}
            restore_checkpoint(manifest['manifest_path'],model,optimizer,rng)
            actual_random=(random.random(),np.random.random(),torch.rand(1),rng.random())
            self.assertEqual(expected_random[:2],actual_random[:2]);torch.testing.assert_close(expected_random[2],actual_random[2]);self.assertEqual(expected_random[3],actual_random[3])
            model(x).square().mean().backward();optimizer.step();optimizer.zero_grad()
            for n,p in model.named_parameters():torch.testing.assert_close(p,expected[n],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
