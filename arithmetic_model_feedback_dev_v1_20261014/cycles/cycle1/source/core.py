"""Raw-byte recurrent policies. No evaluator, arithmetic or attention input."""
import copy
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parent
path=ROOT.parent/'arithmetic_source_derived_kernel_dev_v1_20261013/core.py'
spec=importlib.util.spec_from_file_location('source_gru',path)
S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
EOS,BEGIN,REPLY=256,257,258
MODELS=('R0','KEEP')

class Learner:
    def __init__(self,model,seed):
        if model not in MODELS:raise ValueError('fixed model')
        self.model=model;self.net=S.Kernel('B24',seed)
        assert self.net.elm is False
        if model=='KEEP':self.net.p['C']=torch.nn.Parameter(torch.zeros(64,64,dtype=torch.float64))
        self.net.p['critic']=torch.nn.Parameter(torch.zeros(65,dtype=torch.float64))
        self.mom={k:torch.zeros_like(p) for k,p in self.net.p.items()}
        self.actor_steps=self.critic_steps=0;self.last={}
        self.resource=dict(records=0,observed_answer_bytes=0,scalar_feedback=0,actor_steps=0,critic_steps=0,
            transition_rows=0,vocabulary_rows=0,generated_payload_bytes=0,queries=0,production_cpu=0.)
    def clone(self):return copy.deepcopy(self)
    def digest(self):
        h=hashlib.sha256()
        for name,items in [('parameter',self.net.p),('moment',self.mom)]:
            for k,v in sorted(items.items()):h.update((name+k).encode());h.update(v.detach().numpy().tobytes())
        h.update(json.dumps([self.actor_steps,self.critic_steps,self.last],sort_keys=True,allow_nan=False).encode())
        return h.hexdigest()
    def step(self,h,b):
        self.resource['transition_rows']+=h.shape[0]
        return self.net.step(h,b)
    def decode_step(self,h,b,c):
        new=self.step(h,b)
        return new+c@self.net.p['C'].T if self.model=='KEEP' else new
    def logits(self,h):
        self.resource['vocabulary_rows']+=h.shape[0];return self.net.logits(h)
    def encode(self,prompt):
        if type(prompt) is not bytes or len(prompt)>256:raise ValueError('bounded raw bytes')
        h=self.net.initial()
        for b in (BEGIN,*prompt):h=self.step(h,torch.tensor([b]))
        c=h;h=self.decode_step(h,torch.tensor([REPLY]),c)
        return c,h
    def critic(self,c):
        u=torch.cat((c.detach(),torch.ones(c.shape[0],1,dtype=torch.float64)),1)
        u=u/torch.linalg.vector_norm(u,dim=1,keepdim=True)
        return torch.sigmoid(u@self.net.p['critic'])
    def act(self,prompt,seed,credit,cap=16):
        start=time.process_time();g=torch.Generator().manual_seed(int(seed))
        with torch.set_grad_enabled(credit):
            c,h=self.encode(prompt);baseline=self.critic(c).detach()[0]
            logps=[];out=bytearray();status=None;actions=[]
            for _ in range(cap+1):
                logp=F.log_softmax(self.logits(h)[0],dim=0)
                token=int(torch.multinomial(logp.detach().exp(),1,generator=g))
                actions.append(token)
                if credit:logps.append(logp[token])
                if token==EOS:status='EOS';break
                if len(out)==cap:status='TRUNCATED';break
                out.append(token);h=self.decode_step(h,torch.tensor([token]),c)
        self.resource['production_cpu']+=time.process_time()-start
        self.resource['generated_payload_bytes']+=len(out)
        return bytes(out),status,dict(c=c.detach(),baseline=baseline,
            logp=torch.stack(logps).sum() if logps else None,actions=actions,
            version=self.actor_steps)
    def ce_loss(self,prompt,answer):
        if type(answer) is not bytes:raise TypeError('only observed answer bytes')
        c,h=self.encode(prompt);losses=[]
        for b in (*answer,EOS):
            losses.append(F.cross_entropy(self.logits(h),torch.tensor([b])))
            if b!=EOS:h=self.decode_step(h,torch.tensor([b]),c)
        return torch.stack(losses).mean()
    def sequence_logp(self,prompt,actions):
        c,h=self.encode(prompt);logs=[]
        for b in actions:
            logs.append(F.log_softmax(self.logits(h)[0],dim=0)[b])
            if b!=EOS:h=self.decode_step(h,torch.tensor([b]),c)
        return torch.stack(logs).sum()
    def update(self,prompt,observed_answer,reward,cache,mode,allowed=True):
        if type(reward) is not int or reward not in (0,1):raise ValueError('actual scalar correctness only')
        if mode not in ('WARM','HALF','SPARSE','SCALAR','DENSE'):raise ValueError('public feedback phase')
        if mode=='SCALAR' and observed_answer is not None:raise ValueError('hidden answer denied')
        self.resource['records']+=1;self.resource['scalar_feedback']+=1
        if observed_answer is not None:self.resource['observed_answer_bytes']+=len(observed_answer)
        if not allowed:return dict(actor_steps=0,critic_steps=0,ce=None,pg=None,critic=None)
        if cache['version']!=self.actor_steps:raise ValueError('stale action credit')
        start=time.process_time();ce=self.ce_loss(prompt,observed_answer) if observed_answer is not None else None
        if mode=='WARM' and ce is None:raise ValueError('warm demonstration required')
        pg=None
        if mode!='WARM':
            if cache['logp'] is None:raise ValueError('missing actual-action credit')
            pg=-(reward-cache['baseline'])*cache['logp']
        loss=ce if pg is None else pg if ce is None else ce+pg
        actor={k:p for k,p in self.net.p.items() if k!='critic'}
        grads=torch.autograd.grad(loss,list(actor.values()))
        norm=math.sqrt(sum(float((g*g).sum()) for g in grads));clip=min(1.,1/max(norm,1e-30))
        self.actor_steps+=1
        with torch.no_grad():
            for (k,p),g in zip(actor.items(),grads):
                if not torch.isfinite(g).all():raise FloatingPointError('actor gradient')
                g=g*clip;self.mom[k].mul_(.99).add_(g*g,alpha=.01)
                p.sub_(.001*g/(torch.sqrt(self.mom[k]/(1-.99**self.actor_steps))+1e-8))
        # c belongs to the actual pre-action model and is detached: critic cannot
        # backpropagate a hidden scalar-as-answer representation objective.
        vl=(self.critic(cache['c'])[0]-reward).square()
        vg=torch.autograd.grad(vl,self.net.p['critic'])[0]
        vn=float(vg.norm());vg=vg*min(1.,1/max(vn,1e-30));self.critic_steps+=1
        with torch.no_grad():
            self.mom['critic'].mul_(.99).add_(vg*vg,alpha=.01)
            self.net.p['critic'].sub_(.001*vg/(torch.sqrt(self.mom['critic']/(1-.99**self.critic_steps))+1e-8))
        self.resource['actor_steps']+=1;self.resource['critic_steps']+=1
        self.resource['production_cpu']+=time.process_time()-start
        if any(not torch.isfinite(p).all() for p in self.net.p.values()):raise FloatingPointError('parameter')
        self.last=dict(actor_steps=1,critic_steps=1,ce=None if ce is None else float(ce.detach()),
            pg=None if pg is None else float(pg.detach()),critic=float(vl.detach()),
            actor_gradient_norm=norm,critic_gradient_norm=vn,clip=clip,
            component_gradient_norms={k:float(g.norm()) for k,g in zip(actor,grads)})
        return dict(self.last)
    @torch.no_grad()
    def generate_batch(self,prompts,cap=16):
        if not prompts:return []
        if any(type(p) is not bytes or len(p)>256 for p in prompts):raise ValueError('raw prompt')
        start=time.process_time();n=len(prompts);seq=[[BEGIN,*p] for p in prompts];h=self.net.initial(n)
        for t in range(max(map(len,seq))):
            active=torch.tensor([t<len(s) for s in seq]);b=torch.tensor([s[t] if t<len(s) else 0 for s in seq])
            h=torch.where(active[:,None],self.step(h,b),h)
        c=h;h=self.decode_step(h,torch.full((n,),REPLY),c)
        out=[bytearray() for _ in prompts];status=[None]*n;active=torch.ones(n,dtype=torch.bool)
        for _ in range(cap+1):
            b=self.logits(h).argmax(1)
            for i in range(n):
                if not active[i]:continue
                token=int(b[i])
                if token==EOS:status[i]='EOS';active[i]=False
                elif len(out[i])==cap:status[i]='TRUNCATED';active[i]=False
                else:out[i].append(token)
            if not bool(active.any()):break
            h=torch.where(active[:,None],self.decode_step(h,b,c),h)
        self.resource['queries']+=n;self.resource['generated_payload_bytes']+=sum(map(len,out))
        self.resource['production_cpu']+=time.process_time()-start
        return [(bytes(a),s) for a,s in zip(out,status)]
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        meta=dict(model=self.model,actor_steps=self.actor_steps,critic_steps=self.critic_steps,last=self.last,resource=self.resource,digest=self.digest())
        data={'metadata':np.array(json.dumps(meta,allow_nan=False))}
        data.update({'p_'+k:p.detach().numpy() for k,p in self.net.p.items()});data.update({'m_'+k:p.numpy() for k,p in self.mom.items()})
        temp=path.with_name(path.name+'.pending')
        with temp.open('wb') as f:np.savez_compressed(f,**data)
        temp.replace(path)
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:
            m=json.loads(str(data['metadata']));c=cls(m['model'],0)
            with torch.no_grad():
                for k,p in c.net.p.items():p.copy_(torch.from_numpy(data['p_'+k]))
                for k,p in c.mom.items():p.copy_(torch.from_numpy(data['m_'+k]))
            for k in ('actor_steps','critic_steps','last','resource'):setattr(c,k,m[k])
        if c.digest()!=m['digest']:raise ValueError('archive mismatch')
        return c
