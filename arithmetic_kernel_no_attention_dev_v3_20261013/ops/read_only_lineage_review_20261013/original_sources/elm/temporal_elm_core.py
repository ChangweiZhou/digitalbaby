#!/usr/bin/env python3
"""Small scalar-output ELM population for genuine temporal event streams."""
from __future__ import annotations
import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_bounded(value:torch.Tensor,lo:float,hi:float)->torch.Tensor:
    u=((value-lo)/(hi-lo)).clamp(1e-5,1-1e-5);return torch.logit(u)

class TemporalELM(nn.Module):
    def __init__(self,input_dim:int,num_classes:int,n:int=64,memory_dim:int=6,branches:int=8,synapses_per_branch:int=4,input_synapses_per_branch:int=3,hidden:int=12,seed:int=7):
        super().__init__();self.input_dim=input_dim;self.num_classes=num_classes;self.n=n;self.m_dim=memory_dim;self.r=branches;self.s=synapses_per_branch;self.s_in=input_synapses_per_branch;self.s_rec=self.s-self.s_in;self.hidden=hidden;self.k_lo=.08;self.k_hi=.995;self.firing_gain=4.
        idx=self._initial_topology(seed);self.register_buffer("source_idx",idx)
        self.syn_weight=nn.Parameter(torch.empty(n,branches,synapses_per_branch));self.syn_weight.data[:,:,:self.s_in].normal_(0,.20);self.syn_weight.data[:,:,self.s_in:].normal_(0,.14)
        self.branch_bias=nn.Parameter(torch.full((n,branches),-.05));self.branch_gain=nn.Parameter(torch.ones(n,branches))
        q=branches+memory_dim;self.w1=nn.Parameter(torch.empty(n,q,hidden));self.b1=nn.Parameter(torch.zeros(n,hidden));self.w2=nn.Parameter(torch.empty(n,hidden,memory_dim));self.b2=nn.Parameter(torch.zeros(n,memory_dim))
        with torch.no_grad():
            for i in range(n): nn.init.xavier_uniform_(self.w1[i]);nn.init.xavier_uniform_(self.w2[i])
        # Correct log-tau initialization.
        logtau=torch.linspace(math.log(1.5),math.log(40.),memory_dim).unsqueeze(0).repeat(n,1);g=torch.Generator().manual_seed(seed+123);logtau+=torch.randn(logtau.shape,generator=g)*.10;tau=logtau.exp().clamp(1.5,40.);k=torch.exp(-1/tau);self.kappa_raw=nn.Parameter(inv_bounded(k,self.k_lo,self.k_hi))
        self.mem_to_act=nn.Parameter(torch.randn(n,memory_dim)*.25);self.branch_to_act=nn.Parameter(torch.randn(n,branches)*.08);self.activity_bias=nn.Parameter(torch.zeros(n));self.threshold=nn.Parameter(torch.full((n,),.50));self.readout=nn.Linear(n,num_classes)

    def _initial_topology(self,seed:int)->torch.Tensor:
        rng=np.random.default_rng(seed);idx=torch.empty(self.n,self.r,self.s,dtype=torch.long)
        # Place cells and sensory channels on a 1D sheet. For images the caller may
        # reorder channels spatially before input; for SHD this is cochlear frequency.
        cellpos=np.linspace(0,1,self.n);srcpos=np.linspace(0,1,self.input_dim)
        for i in range(self.n):
            nearest=np.argsort(np.abs(srcpos-cellpos[i]))[:min(128,self.input_dim)].tolist();used=set()
            rec_order=[int(j) for j in np.argsort(np.abs(cellpos-cellpos[i])) if int(j)!=i];rused=set()
            for b in range(self.r):
                for s in range(self.s_in):
                    pool=list(range(self.input_dim)) if rng.random()<.15 else nearest;avail=[x for x in pool if x not in used] or [x for x in range(self.input_dim) if x not in used];x=int(rng.choice(avail));used.add(x);idx[i,b,s]=x
                for s in range(self.s_in,self.s):
                    pool=rec_order if rng.random()<.20 else rec_order[:min(24,len(rec_order))];avail=[x for x in pool if x not in rused and x!=i] or [x for x in rec_order if x not in rused];x=int(rng.choice(avail));rused.add(x);idx[i,b,s]=self.input_dim+x
        return idx

    def kappa(self): return self.k_lo+(self.k_hi-self.k_lo)*torch.sigmoid(self.kappa_raw)

    def forward(self,x:torch.Tensor)->Dict[str,torch.Tensor]:
        # x: [B,T,input_dim]
        B,T,_=x.shape;mem=torch.zeros(B,self.n,self.m_dim,device=x.device,dtype=x.dtype);prev=torch.zeros(B,self.n,device=x.device,dtype=x.dtype);hist=[];srcidx=self.source_idx.reshape(-1);k=self.kappa().unsqueeze(0);gain=F.softplus(self.branch_gain).unsqueeze(0)
        for t in range(T):
            src=torch.cat([x[:,t],prev],1);g=src.index_select(1,srcidx).reshape(B,self.n,self.r,self.s);bs=(g*self.syn_weight.unsqueeze(0)).sum(-1)+self.branch_bias.unsqueeze(0);ba=torch.tanh(gain*bs);internal=torch.cat([ba,mem],-1);h=torch.tanh(torch.einsum('bnq,nqh->bnh',internal,self.w1)+self.b1.unsqueeze(0));cand=torch.tanh(torch.einsum('bnh,nhm->bnm',h,self.w2)+self.b2.unsqueeze(0));mem=k*mem+(1-k)*cand;v=(mem*self.mem_to_act.unsqueeze(0)).sum(-1)+(ba*self.branch_to_act.unsqueeze(0)).sum(-1)+self.activity_bias.unsqueeze(0);prev=torch.sigmoid(self.firing_gain*(v-self.threshold.unsqueeze(0)));hist.append(prev)
        ah=torch.stack(hist,1);return {"logits":self.readout(prev),"activity":ah,"final_activity":prev,"memory":mem}

class TemporalLeaky(nn.Module):
    def __init__(self,reference:TemporalELM,num_classes:int):
        super().__init__();self.input_dim=reference.input_dim;self.n=reference.n;self.r=reference.r;self.s=reference.s;self.register_buffer('source_idx',reference.source_idx.detach().clone());self.w=nn.Parameter(torch.randn(self.n,self.r,self.s)*.16);self.leak_raw=nn.Parameter(torch.zeros(self.n));self.bias=nn.Parameter(torch.zeros(self.n));self.readout=nn.Linear(self.n,num_classes)
    def forward(self,x):
        B,T,_=x.shape;h=torch.zeros(B,self.n,device=x.device,dtype=x.dtype);prev=torch.zeros_like(h);hist=[];leak=.5+.49*torch.sigmoid(self.leak_raw);idx=self.source_idx.reshape(-1)
        for t in range(T):
            src=torch.cat([x[:,t],prev],1);g=src.index_select(1,idx).reshape(B,self.n,self.r,self.s);d=(g*self.w.unsqueeze(0)).sum((2,3))+self.bias;h=leak.unsqueeze(0)*h+(1-leak).unsqueeze(0)*torch.tanh(d);prev=torch.sigmoid(4*h);hist.append(prev)
        ah=torch.stack(hist,1);return {'logits':self.readout(prev),'activity':ah,'final_activity':prev,'memory':h}
