"""Four distinct engineered operations on permitted actual observations only."""
import copy, importlib.util, json, math, random
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import config
spec=importlib.util.spec_from_file_location('unchanged_support_core',config.PRIOR/'cores.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
features=parent.features;emitted=parent.emitted;state_hash=parent.state_hash

def graph():
    rng=random.Random(20261011055);a=np.zeros((31,31))
    for j in range(30):
        bank=range(0,15) if j<15 else range(15,30)
        for i in rng.sample([i for i in bank if i!=j],2):a[i,j]=1.
    a.flags.writeable=False
    return a
A=graph()
role_perm=list(range(15));random.Random(20261011077).shuffle(role_perm)

class Core(parent.Core):
    def __init__(self,arm,world_id=0):
        super().__init__(arm)
        if arm not in config.BASELINES:
            self.mechanism_rng=random.Random(20261011034 if arm.startswith('RETRO') else 20261011047)
            self.extra={'events':0,'scans':0,'solves':0,'edges':0,'changed':0,'previous_selected':None,'repeat_selections':0,
                        'proposed_norm_sum':0.,'realized_norm_sum':0.}
            if arm.startswith('COVER'):self.buffer=[]
            elif arm.startswith('RETRO'):
                self.ledger=[];self.h=np.eye(31)*.1;self.b=np.zeros(31)
            elif arm.startswith('ROLE'):
                self.queue=[];self.seen_left=[False]*15;self.seen_right=[False]*15
                self.agree=self.disagree=0
        self.birth=copy.deepcopy(self.state())
    def state(self):
        out=super().state()
        if hasattr(self,'extra'):
            out.update(extra=self.extra,mechanism_rng=self.mechanism_rng.getstate())
            if self.arm.startswith('COVER'):out['buffer']=self.buffer
            elif self.arm.startswith('RETRO'):out.update(ledger=self.ledger,h=self.h,b=self.b)
            elif self.arm.startswith('ROLE'):
                out.update(queue=self.queue,seen_left=self.seen_left,seen_right=self.seen_right,
                           agree=self.agree,disagree=self.disagree)
        return out
    def restore(self,state):
        if set(state)!=set(self.state()):raise ValueError('complete learned state required')
        def tuples(v):return tuple(tuples(x) for x in v) if isinstance(v,(list,tuple)) else v
        for k,v in state.items():
            if k in ('rng','mechanism_rng'):getattr(self,k).setstate(tuples(v))
            else:setattr(self,k,copy.deepcopy(v))
    def support(self,ids,target):
        # Every candidate retains actual SUPPORT, not its random rate control.
        key=[int(config.OPERANDS[ids[0]]),int(config.OPERANDS[ids[1]-15])]
        index=next((i for i,row in enumerate(self.cache) if row['key']==key),None)
        past=self.cache[index]['answers'] if index is not None else []
        votes=[past.count(y) for y in range(9)];n=len(past);count=max(votes)
        mode=votes.index(count) if count>=2 and 2*count>n else None
        eligible=mode is not None;conflict=eligible and target!=mode
        draw=self.rng.random();prob=sum(self.history)/len(self.history) if self.history else 0.
        return key,index,dict(eligible=eligible,prior_count=n,mode=mode,mode_count=count,
                conflict=bool(conflict),probability=prob,random=draw,factor=.25 if conflict else 1.)
    def remember_support(self,key,index,target,gate):
        if gate['eligible']:self.history=(self.history+[int(gate['conflict'])])[-32:]
        if index is None:self.cache=(self.cache+[dict(key=key,answers=[target])])[-32:]
        else:self.cache[index]['answers']=(self.cache[index]['answers']+[target])[-8:]
    def update(self,ids,target,allowed,cached_value):
        if self.arm in config.BASELINES:
            out=super().update(ids,target,allowed,cached_value);out['mechanism']=None;return out
        # Same strict public argument validation as the inherited writer.
        if type(allowed) is not bool or type(target) is not int or not 0<=target<=8:raise ValueError('observed target/permission')
        if len(ids)!=3 or any(type(i) is not int for i in ids) or not(0<=ids[0]<15 and 15<=ids[1]<30 and ids[2]==30):raise ValueError('ordered features')
        out=dict(performed=False,replay_index=None,extra_performed=False,extra_ids=None,extra_target=None,gate=None,mechanism=None)
        if not allowed:return out
        key,index,gate=self.support(ids,target)
        r=target-cached_value;q=self.p[:,ids[0]]+self.p[:,ids[1]]+self.p[:,ids[2]]
        u=math.fsum(float(q[i]) for i in ids)
        if u<0 or not math.isfinite(u):raise ValueError('invalid confidence')
        robust=1. if abs(r)<=1 else 1./abs(r);gamma=gate['factor']*robust;d=1.+gamma*u
        if not math.isfinite(d) or d<=0:raise ValueError('invalid denominator')
        gate.update(gamma=gamma,gamma_r=robust)
        event=None
        if self.arm.startswith('RETRO'):
            self.ledger.append(dict(ids=list(ids),target=target,arrival=gamma,effective=gamma,arrival_index=self.updates))
            self.ledger=self.ledger[-32:]
            self.h=self.h+gamma*self.outer_x(ids);self.b=self.b+gamma*target*self.vector(ids)
            rows=[j for j,z in enumerate(self.ledger) if z['ids']==list(ids)][-8:]
            votes=[sum(self.ledger[j]['target']==y for j in rows) for y in range(9)]
            mode=votes.index(max(votes)) if len(rows)>=3 and 4*max(votes)>=3*len(rows) else None
            eligible_rows=rows[:-1]
            factors=[.25 if mode is not None and self.ledger[j]['target']!=mode else 1. for j in eligible_rows]
            if self.arm=='RETRO-RANDOM':self.mechanism_rng.shuffle(factors)
            changed=[]
            if self.arm!='RETRO-FIXED':
                for j,factor in zip(eligible_rows,factors):
                    row=self.ledger[j];new=row['arrival']*factor;delta=new-row['effective']
                    if delta!=0:
                        self.h=self.h+delta*self.outer_x(row['ids']);self.b=self.b+delta*row['target']*self.vector(row['ids'])
                        changed.append(dict(index=j,delta=delta,factor=factor));row['effective']=new
            self.h=(self.h+self.h.T)/2
            self.p=cho_solve(cho_factor(self.h,lower=True),np.eye(31));self.p=(self.p+self.p.T)/2
            self.w=np.clip(self.p@self.b,-4*config.KAPPA,4*config.KAPPA)
            self.extra['scans']+=len(self.ledger);self.extra['solves']+=1;self.extra['changed']+=len(changed)
            self.scalar_ops+=11000+2*31**3+len(changed)*3000;self.traffic_bytes+=192000+len(changed)*24000
            event=dict(window=rows,eligible_rows=eligible_rows,mode=mode,changes=changed)
        else:
            delta=(gamma*r/d)*q
            if self.arm.startswith('KCI'):
                x=self.vector(ids);g=1./(1.+config.KCI_LAMBDA*(1.-x)*(A[:,ids[0]]+A[:,ids[1]]+A[:,30]));g[30]=1.
                proposed=g*delta
                ratio=float(np.linalg.norm(g*q)/np.linalg.norm(q)) if np.linalg.norm(q)>0 else 1.
                if self.arm=='KCI-GLOBAL-NORM':proposed=ratio*delta
                before=self.w.copy();self.w=np.clip(self.w+proposed,-4*config.KAPPA,4*config.KAPPA)
                event=dict(g=g.tolist(),ratio=ratio,proposed_norm=float(np.linalg.norm(proposed)),
                           realized_norm=float(np.linalg.norm(self.w-before)))
                self.extra['proposed_norm_sum']+=event['proposed_norm'];self.extra['realized_norm_sum']+=event['realized_norm'];self.extra['edges']+=4
                self.scalar_ops+=500;self.traffic_bytes+=2048
            else:self.w=np.clip(self.w+delta,-4*config.KAPPA,4*config.KAPPA)
            self.p=self.p-(gamma/d)*np.outer(q,q);self.p=(self.p+self.p.T)/2
            if self.arm.startswith('COVER'):
                chosen=None
                if self.buffer:
                    if self.arm=='COVER-UNIFORM':chosen=self.mechanism_rng.randrange(len(self.buffer))
                    else:
                        score=[]
                        for row in self.buffer:
                            if self.arm=='COVER':
                                qr=self.p[:,row['ids'][0]]+self.p[:,row['ids'][1]]+self.p[:,30]
                                score.append(math.fsum(float(qr[i]) for i in row['ids']))
                            else:score.append(abs(row['target']-self.value(row['ids'])))
                        chosen=max(range(len(score)),key=lambda i:score[i])
                    row=self.buffer[chosen];qi=self.p[:,row['ids'][0]]+self.p[:,row['ids'][1]]+self.p[:,30]
                    ri=row['target']-self.value(row['ids']);gi=row['factor']*(1. if abs(ri)<=1 else 1./abs(ri))
                    ui=math.fsum(float(qi[i]) for i in row['ids']);di=1.+gi*ui
                    self.w=np.clip(self.w+(gi*ri/di)*qi,-4*config.KAPPA,4*config.KAPPA)
                    self.extra_updates+=1
                    self.extra['repeat_selections']+=int(self.extra['previous_selected']==row['arrival_index'])
                    self.extra['previous_selected']=row['arrival_index']
                    out.update(replay_index=chosen,extra_performed=True,extra_ids=row['ids'],extra_target=row['target'])
                self.extra['scans']+=len(self.buffer)
                event=dict(chosen=chosen,buffer_size=len(self.buffer),selected_arrival=None if chosen is None else row['arrival_index'])
                self.buffer=(self.buffer+[dict(ids=list(ids),target=target,factor=gate['factor'],arrival_index=self.updates)])[-32:]
                self.scalar_ops+=3000+len(self.buffer)*200;self.traffic_bytes+=12000+len(self.buffer)*1200
            elif self.arm.startswith('ROLE'):
                left,right=ids[0],ids[1]-15;self.seen_left[left]=True;self.seen_right[right]=True
                found=next((j for j,z in enumerate(self.queue) if z['left']==right and z['right']==left),None) if left!=right else None
                if found is not None:
                    old=self.queue.pop(found);good=abs(target-old['target'])<=1
                    self.agree+=int(good);self.disagree+=int(not good)
                elif left!=right:self.queue=(self.queue+[dict(left=left,right=right,target=target)])[-32:]
                eta=.05*max(0.,(self.agree-self.disagree)/(self.agree+self.disagree+4))
                if self.arm=='ROLE-FIXED':eta=.05
                coupled=[]
                for i in sorted(set((left,right))):
                    j=role_perm[i] if self.arm=='ROLE-PERMUTED' else i
                    if not (self.seen_left[i] and self.seen_right[i]):continue
                    a,b=float(self.w[i]),float(self.w[15+j]);m=(a+b)/2;v=(a-b)/2
                    if eta>0:self.w[i]=m+(1.-eta)*v;self.w[15+j]=m-(1.-eta)*v
                    coupled.append([i,j])
                self.extra['changed']+=len(coupled) if eta>0 else 0
                event=dict(paired=found is not None,eta=eta,coupled=coupled,agree=self.agree,disagree=self.disagree)
                self.scalar_ops+=1500;self.traffic_bytes+=4096
        self.remember_support(key,index,target,gate)
        self.updates+=1;self.extra['events']+=1
        self.scalar_ops+=11256;self.traffic_bytes+=97024
        out.update(performed=True,gate=gate,mechanism=event);return out
    @staticmethod
    def vector(ids):
        x=np.zeros(31);x[list(ids)]=1.;return x
    @staticmethod
    def outer_x(ids):
        x=Core.vector(ids);return np.outer(x,x)
    def fixed_bytes(self):return A.nbytes if self.arm.startswith('KCI') else (120 if self.arm=='ROLE-PERMUTED' else 0)

class ByteLearner(parent.ByteLearner):
    def __init__(self,arm,world_id=0):
        self.core=Core(arm,world_id);self.line=b'';self.pending=None;self.ticks=0;self.bytes_seen=0
