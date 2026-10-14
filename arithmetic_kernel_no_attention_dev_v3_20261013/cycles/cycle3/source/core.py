"""Generic learned byte kernels. No evaluator imports or semantic token classes."""
import copy, hashlib, math, pickle, random, time
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

EOS, BEGIN, REPLY = 256, 257, 258
torch.set_num_threads(1)

@dataclass(frozen=True)
class Config:
    kind: str = 'G'
    width: int = 64
    embedding: int = 16
    passes: int = 1
    untied: bool = False
    replay: bool = False
    cadence: int = 1
    stop_policy: bool = False
    reset_modulation: bool = False
    context: int = 128
    rate: float = .001

def configuration(arm):
    configs = {
      'G0':Config(), 'G1':Config(replay=True), 'G4':Config(cadence=4),
      'R1':Config(kind='R',width=32),
      'R8':Config(kind='R',width=32,passes=8),
      'RU8':Config(kind='R',width=32,passes=8,untied=True),
      'C':Config(kind='C',width=32),
      'P':Config(kind='P',width=32,cadence=4),
      'P_STOP':Config(kind='P',width=32,cadence=4,stop_policy=True),
      'P_RESET':Config(kind='P',width=32,cadence=4,reset_modulation=True)}
    return configs[arm]

class Kernel(nn.Module):
    def __init__(self, config, seed):
        super().__init__(); self.config=config; d=config.width; e=config.embedding
        self.E=nn.Embedding(259,e)
        if config.kind in ('G','P'):
            self.scan=nn.GRU(e,d,batch_first=True)
            self.head=nn.Linear(d,257)
            if config.kind=='P':self.policy=nn.Sequential(nn.Linear(2*d+257,d),nn.Tanh(),nn.Linear(d,3*d))
        elif config.kind=='R':
            n=config.passes if config.untied else 1
            self.initial=nn.ModuleList([nn.Linear(d,d) for _ in range(n)])
            self.scan=nn.ModuleList([nn.GRU(e+d,d,batch_first=True) for _ in range(n)])
            self.work=nn.ModuleList([nn.GRUCell(d,d) for _ in range(n)])
            self.head=nn.Linear(2*d,257)
        elif config.kind=='C':
            self.project=nn.Linear(e,d)
            self.cell=nn.Conv1d(d,2*d,3,padding=1)
            self.head=nn.Linear(d,257)
        else:raise ValueError('kernel kind')
        # Explicit, task-independent initialization, no reliance on default constructors.
        with torch.no_grad():
            for name,p in sorted(self.named_parameters()):
                # Untied controls start as exact copies of the shared transition.
                normalized=name
                if config.kind=='R' and name.split('.')[0] in ('initial','scan','work'):
                    parts=name.split('.');parts[1]='0';normalized='.'.join(parts)
                draw_seed=int.from_bytes(hashlib.sha256(f'{seed}|{normalized}'.encode()).digest()[:8],'big')%(2**63)
                g=torch.Generator().manual_seed(draw_seed)
                if name=='E.weight':p.copy_(torch.randn(p.shape,generator=g)/math.sqrt(e))
                elif ('scan' in name or 'work' in name):
                    p.copy_(torch.rand(p.shape,generator=g)*(2/math.sqrt(d))-1/math.sqrt(d))
                elif p.ndim>=2:
                    fan_in,fan_out=nn.init._calculate_fan_in_and_fan_out(p)
                    bound=math.sqrt(6/(fan_in+fan_out));p.copy_(torch.rand(p.shape,generator=g)*2*bound-bound)
                else:p.zero_()
            if config.untied:
                for i in range(1,config.passes):
                    self.initial[i].load_state_dict(self.initial[0].state_dict())
                    self.scan[i].load_state_dict(self.scan[0].state_dict())
                    self.work[i].load_state_dict(self.work[0].state_dict())

    def decision(self, ids, working=None, modulation=None):
        ids=ids[-self.config.context:]; d=self.config.width
        x=self.E(torch.tensor(ids,dtype=torch.long)).unsqueeze(0)
        if self.config.kind in ('G','P'):
            h=torch.zeros(1,1,d)
            if self.config.kind=='P':
                # Fixed recurrent path: neuronwise bias, not memory-address selection.
                for i in range(x.shape[1]):
                    _,h=self.scan(x[:,i:i+1],h)
                    h=torch.tanh(h+0.1*torch.tanh(modulation).view(1,1,d))
            else:_,h=self.scan(x,h)
            state=h.reshape(d);return self.head(state),None,state
        if self.config.kind=='R':
            s=torch.zeros(d) if working is None else working
            for k in range(self.config.passes):
                j=k if self.config.untied else 0
                h=torch.tanh(self.initial[j](s)).view(1,1,d)
                xx=torch.cat((x,s.view(1,1,d).expand(1,x.shape[1],d)),dim=-1)
                _,h=self.scan[j](xx,h); f=h.reshape(d)
                s=self.work[j](f.view(1,d),s.view(1,d)).reshape(d)
            return self.head(torch.cat((f,s))),s,torch.cat((f,s))
        z=torch.tanh(self.project(x[0])).T.unsqueeze(0)
        for _ in range(max(1,len(ids)-1)):
            a,b=self.cell(z).chunk(2,dim=1);gate=torch.sigmoid(a)
            z=(1-gate)*z+gate*torch.tanh(b)
        state=z[0,:,-1];return self.head(state),None,state

    def feedback_modulation(self,m,state,logits,target):
        err=logits.softmax(-1)-F.one_hot(torch.tensor(target),257).to(logits.dtype)
        # Activities/errors are observations; policy learns from later losses through m.
        v=torch.cat((m,state.detach(),err.detach()))
        a,b,c=self.policy(v).chunk(3)
        new=torch.tanh(torch.sigmoid(a)*m+0.1*torch.sigmoid(b)*torch.tanh(c))
        return new.detach() if self.config.stop_policy else new

class Learner:
    def __init__(self, arm, seed):
        self.arm=arm; self.config=configuration(arm); self.net=Kernel(self.config,seed)
        self.opt=torch.optim.AdamW(self.net.parameters(),lr=self.config.rate,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
        self.m=torch.zeros(self.config.width);self.pending=[];self.buffer=[]
        self.rng=random.Random(seed+711);self.eligible=0;self.steps=0
        self.resource={'external_records':0,'external_payload_bytes':0,'actual_target_applications':0,'replay_records':0,'replay_target_applications':0,'optimizer_steps':0,'decision_calls':0,'recurrent_positions':0,'local_site_steps':0,'generation_calls':0,'generated_decisions':0,'context_evictions':0,'cpu_training':0.,'cpu_generation':0.}
        self.last_gradient={};self.last_update={}

    def _decision(self,ids,work,m):
        L=min(len(ids),self.config.context)
        if len(ids)>self.config.context:self.resource['context_evictions']+=1
        self.resource['decision_calls']+=1
        if self.config.kind=='C':self.resource['local_site_steps']+=L*max(1,L-1)
        else:self.resource['recurrent_positions']+=L*(self.config.passes if self.config.kind=='R' else 1)
        return self.net.decision(ids,work,m)

    @staticmethod
    def _validate_bytes(value):
        if not isinstance(value,bytes):raise TypeError('payload must be bytes')

    def loss_record(self,prompt,reply,learn_modulation):
        self._validate_bytes(prompt);self._validate_bytes(reply)
        if learn_modulation and len(reply)>64:raise ValueError('training reply exceeds declared 64-byte credit cap')
        ids=[BEGIN,*prompt,REPLY]; work=None;losses=[];rows=[]
        m=torch.zeros_like(self.m) if self.config.reset_modulation else self.m
        for target in [*reply,EOS]:
            logits,work,state=self._decision(ids,work,m)
            loss=F.cross_entropy(logits.view(1,257),torch.tensor([target]));losses.append(loss)
            rows.append({'target':target,'choice':int(logits.detach().argmax()),'loss':float(loss.detach())})
            if learn_modulation and self.config.kind=='P':m=self.net.feedback_modulation(m,state,logits,target)
            if target!=EOS:ids.append(target)
        if learn_modulation and self.config.kind=='P':self.m=m
        return torch.stack(losses).mean(),rows

    def observe(self,prompt,reply,allowed):
        if type(allowed) is not bool:raise TypeError('whole-record permission must be bool')
        self._validate_bytes(prompt);self._validate_bytes(reply)
        if allowed and len(reply)>64:raise ValueError('training reply exceeds declared 64-byte credit cap')
        start=time.process_time(); self.resource['external_records']+=1
        self.resource['external_payload_bytes']+=len(prompt)+len(reply)
        if not allowed:
            with torch.no_grad():loss,rows=self.loss_record(prompt,reply,False)
        else:
            loss,rows=self.loss_record(prompt,reply,True)
            self.resource['actual_target_applications']+=len(reply)+1
            if self.config.replay and self.buffer:
                pp,rr=self.buffer[self.rng.randrange(len(self.buffer))]
                old,_=self.loss_record(pp,rr,False);loss=(loss+old)/2
                self.resource['replay_records']+=1;self.resource['replay_target_applications']+=len(rr)+1
            self.pending.append(loss)
            if len(self.pending)==self.config.cadence:self.flush()
            if self.config.replay and len(prompt)+len(reply)<=64:
                self.eligible+=1;item=(bytes(prompt),bytes(reply))
                if len(self.buffer)<64:self.buffer.append(item)
                else:
                    j=self.rng.randrange(self.eligible)
                    if j<64:self.buffer[j]=item
        self.resource['cpu_training']+=time.process_time()-start
        return rows

    def flush(self):
        if not self.pending:return
        self.opt.zero_grad(set_to_none=True)
        loss=torch.stack(self.pending).mean();loss.backward()
        self.last_gradient={name:float(p.grad.norm()) if p.grad is not None else 0. for name,p in self.net.named_parameters()}
        norm=float(nn.utils.clip_grad_norm_(self.net.parameters(),1.))
        before=[p.detach().clone() for p in self.net.parameters()]
        self.opt.step(); count=sum(p.numel() for p in self.net.parameters())
        square=sum(float(((p.detach()-old).double()**2).sum()) for p,old in zip(self.net.parameters(),before))
        proposed=math.sqrt(square/count);scale=min(1.,self.config.rate/max(proposed,1e-30))
        with torch.no_grad():
            for p,old in zip(self.net.parameters(),before):p.copy_(old+(p-old)*scale)
        applied=math.sqrt(sum(float(((p.detach()-old).double()**2).sum()) for p,old in zip(self.net.parameters(),before))/count)
        self.last_update={'gradient_norm':norm,'proposed_RMS':proposed,'applied_RMS':applied,'loss':float(loss.detach())}
        self.pending=[];self.m=self.m.detach();self.steps+=1;self.resource['optimizer_steps']+=1
        self.opt.zero_grad(set_to_none=True)
        if not all(torch.isfinite(p).all() for p in self.net.parameters()) or not torch.isfinite(self.m).all():raise FloatingPointError('nonfinite state')

    def generate(self,prompt,cap=16):
        self._validate_bytes(prompt)
        if type(cap) is not int or cap<0:raise ValueError('nonnegative output cap')
        start=time.process_time();ids=[BEGIN,*prompt,REPLY];work=None;out=[]
        self.resource['generation_calls']+=1
        modulation=torch.zeros_like(self.m) if self.config.reset_modulation else self.m
        with torch.no_grad():
            for _ in range(cap+1):
                logits,work,_=self._decision(ids,work,modulation)
                token=int(logits.argmax());self.resource['generated_decisions']+=1
                if token==EOS:status='EOS';break
                if len(out)==cap:status='TRUNCATED';break
                out.append(token);ids.append(token)
        self.resource['cpu_generation']+=time.process_time()-start
        return bytes(out),status

    def digest(self):
        h=hashlib.sha256()
        for name,p in sorted(self.net.state_dict().items()):h.update(name.encode());h.update(p.detach().numpy().tobytes())
        def canonical(value):
            if torch.is_tensor(value):return ('tensor',str(value.dtype),tuple(value.shape),value.detach().numpy().tobytes())
            if isinstance(value,dict):return tuple((k,canonical(v)) for k,v in sorted(value.items(),key=lambda x:str(x[0])))
            if isinstance(value,(list,tuple)):return tuple(canonical(v) for v in value)
            return value
        h.update(pickle.dumps(canonical(self.opt.state_dict()),protocol=5))
        h.update(self.m.detach().numpy().tobytes());h.update(pickle.dumps((self.steps,self.eligible,self.buffer,self.rng.getstate()),protocol=5))
        # Pending graphs represented by loss value; separate gradient ancestry tested explicitly.
        h.update(pickle.dumps([float(x.detach()) for x in self.pending],protocol=5))
        return h.hexdigest()

    def clone(self):
        if self.pending:raise RuntimeError('flush allowed window before branch fork')
        return copy.deepcopy(self)

    def state_bytes(self):
        n=sum(p.numel()*p.element_size() for p in self.net.parameters())
        moments=sum(x.numel()*x.element_size() for s in self.opt.state.values() for x in s.values() if torch.is_tensor(x))
        return {'parameters':n,'optimizer_tensors':moments,'fast_modulation':self.m.numel()*self.m.element_size(),'raw_replay_payload':sum(len(p)+len(r) for p,r in self.buffer),'pending_records':len(self.pending),'transient_backward_peak':'not measured; pending graph is bounded by cadence and context'}
