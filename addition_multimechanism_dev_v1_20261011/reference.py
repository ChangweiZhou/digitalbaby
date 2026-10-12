"""Separate transition reconstruction. Does not import production cores."""
import copy, hashlib, json, math, random
from collections import Counter
import numpy as np
from scipy.linalg import cho_factor, cho_solve
import config

def checksum(s):
    h=hashlib.sha256()
    for k,v in sorted(s.items()):
        h.update(k.encode())
        if isinstance(v,np.ndarray):
            h.update(str((v.shape,v.dtype.str)).encode());a=np.array(v,copy=True);a[a==0.]=0.;h.update(a.tobytes())
        else:h.update(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    return h.hexdigest()
def answer(v):
    values=[(k-v)*(k-v) for k in range(9)];return values.index(min(values))
def input_vector(ids):
    v=np.zeros(31)
    for i in ids:v[i]=1.
    return v
def adjacency():
    a=np.zeros((31,31));r=random.Random(20261011055)
    for src in range(30):
        lo=0 if src<15 else 15
        choices=r.sample([i for i in range(lo,lo+15) if i!=src],2)
        for dst in choices:a[dst,src]=1.
    return a

class Reference:
    def __init__(self,arm,world_id=0):
        if arm not in config.ARMS:raise ValueError('arm')
        self.arm=arm
        self.scalar_ops=0;self.traffic_bytes=0
        self.s=dict(w=np.zeros(31),p=np.diag([10.]*31),cache=[],history=[],updates=0,
                    extra_updates=0,rng=random.Random(20261011047).getstate())
        if arm not in config.BASELINES:
            self.s.update(extra=dict(events=0,scans=0,solves=0,edges=0,changed=0,previous_selected=None,
                         repeat_selections=0,proposed_norm_sum=0.,realized_norm_sum=0.))
            if arm in ('RETRO-RANDOM','COVER-UNIFORM'):
                self.s['mechanism_rng']=random.Random(20261011034 if arm=='RETRO-RANDOM' else 20261011047).getstate()
            if arm.startswith('COVER'):self.s['buffer']=[]
            elif arm.startswith('RETRO'):self.s.update(ledger=[],h=np.diag([.1]*31),b=np.zeros(31))
            elif arm.startswith('ROLE'):self.s.update(queue=[],seen_left=[False]*15,seen_right=[False]*15,agree=0,disagree=0)
    def state(self):return self.s
    def clone(self):return copy.deepcopy(self)
    def digest(self):return checksum(self.s)
    def predict(self,ids):return math.fsum(float(self.s['w'][i]) for i in ids)
    def step(self,ids,target,allowed,predicted):
        result=dict(performed=False,replay_index=None,extra_performed=False,extra_ids=None,extra_target=None,gate=None,mechanism=None)
        if not allowed:return result
        s=self.s
        gate=dict(eligible=False,prior_count=0,mode=None,mode_count=0,conflict=False,probability=0.,random=None,factor=1.)
        if self.arm!='COV-R':
            key=[config.OPERANDS[ids[0]],config.OPERANDS[ids[1]-15]]
            found=next((i for i,row in enumerate(s['cache']) if row['key']==key),None)
            past=s['cache'][found]['answers'] if found is not None else []
            tally=Counter(past);n=len(past);high=max(tally.values(),default=0)
            modes=[y for y in range(9) if tally[y]>=2 and 2*tally[y]>n]
            mode=modes[0] if modes else None
            rgen=random.Random();rgen.setstate(s['rng']);draw=rgen.random();s['rng']=rgen.getstate()
            eligible=mode is not None;conflict=bool(eligible and target!=mode)
            probability=sum(s['history'])/len(s['history']) if s['history'] else 0.
            gate.update(eligible=eligible,prior_count=n,mode=mode,mode_count=high,conflict=conflict,
                        probability=probability,random=draw,factor=.25 if conflict else 1.)
        residual=target-predicted;q=s['p'][:,ids[0]]+s['p'][:,ids[1]]+s['p'][:,30]
        u=math.fsum(float(q[i]) for i in ids)
        robust=1. if abs(residual)<=1 else 1./abs(residual);gamma=gate['factor']*robust;denominator=1.+gamma*u
        if u<0 or not math.isfinite(denominator) or denominator<=0:raise AssertionError('positive finite denominator')
        gate.update(gamma=gamma,gamma_r=robust)
        event=None
        if self.arm.startswith('RETRO'):
            s['ledger'].append(dict(ids=list(ids),target=target,arrival=gamma,effective=gamma,arrival_index=s['updates']))
            if len(s['ledger'])>32:s['ledger'].pop(0)
            v=input_vector(ids)
            s['h']=s['h']+gamma*np.outer(v,v);s['b']=s['b']+gamma*target*v
            window=[i for i,row in enumerate(s['ledger']) if row['ids']==list(ids)][-8:]
            tally=Counter(s['ledger'][i]['target'] for i in window)
            mode=next((y for y in range(9) if len(window)>=3 and 4*tally[y]>=3*len(window)),None)
            eligible_window=window[:-1]
            factors=[.25 if mode is not None and s['ledger'][i]['target']!=mode else 1. for i in eligible_window]
            if self.arm=='RETRO-RANDOM':
                rng=random.Random();rng.setstate(s['mechanism_rng']);rng.shuffle(factors);s['mechanism_rng']=rng.getstate()
            changed=[]
            if self.arm!='RETRO-FIXED':
                for i,factor in zip(eligible_window,factors):
                    row=s['ledger'][i];desired=row['arrival']*factor;difference=desired-row['effective']
                    if difference!=0.:
                        v=input_vector(row['ids']);s['h']=s['h']+difference*np.outer(v,v)
                        s['b']=s['b']+difference*row['target']*v;row['effective']=desired
                        changed.append(dict(index=i,delta=difference,factor=factor))
            s['h']=(s['h']+s['h'].T)/2
            s['p']=cho_solve(cho_factor(s['h'],lower=True),np.eye(31));s['p']=(s['p']+s['p'].T)/2
            s['w']=np.minimum(4*config.KAPPA,np.maximum(-4*config.KAPPA,s['p']@s['b']))
            s['extra']['scans']+=len(s['ledger']);s['extra']['solves']+=1;s['extra']['changed']+=len(changed)
            event=dict(window=window,eligible_rows=eligible_window,mode=mode,changes=changed)
        else:
            delta=(gamma*residual/denominator)*q
            if self.arm.startswith('KCI'):
                v=input_vector(ids);a=adjacency();g=1./(1.+config.KCI_LAMBDA*(1.-v)*(a[:,ids[0]]+a[:,ids[1]]+a[:,30]));g[30]=1.
                ratio=float(np.linalg.norm(g*q)/np.linalg.norm(q)) if np.linalg.norm(q)>0 else 1.
                proposal=(ratio*delta if self.arm=='KCI-GLOBAL-NORM' else g*delta)
                before=s['w'].copy();s['w']=np.minimum(4*config.KAPPA,np.maximum(-4*config.KAPPA,s['w']+proposal))
                event=dict(g=g.tolist(),ratio=ratio,proposed_norm=float(np.linalg.norm(proposal)),realized_norm=float(np.linalg.norm(s['w']-before)))
                s['extra']['proposed_norm_sum']+=event['proposed_norm'];s['extra']['realized_norm_sum']+=event['realized_norm'];s['extra']['edges']+=4
            else:s['w']=np.minimum(4*config.KAPPA,np.maximum(-4*config.KAPPA,s['w']+delta))
            lowered=s['p']-(gamma/denominator)*(q[:,None]*q[None,:]);s['p']=(lowered+lowered.T)/2.
            if self.arm.startswith('COVER'):
                chosen=None
                if s['buffer']:
                    if self.arm=='COVER-UNIFORM':
                        rng=random.Random();rng.setstate(s['mechanism_rng']);chosen=rng.randrange(len(s['buffer']));s['mechanism_rng']=rng.getstate()
                    else:
                        scores=[]
                        for row in s['buffer']:
                            if self.arm=='COVER':
                                z=s['p'][:,row['ids'][0]]+s['p'][:,row['ids'][1]]+s['p'][:,30]
                                scores.append(math.fsum(float(z[i]) for i in row['ids']))
                            else:scores.append(abs(row['target']-self.predict(row['ids'])))
                        chosen=scores.index(max(scores))
                    row=s['buffer'][chosen];z=s['p'][:,row['ids'][0]]+s['p'][:,row['ids'][1]]+s['p'][:,30]
                    residual2=row['target']-self.predict(row['ids']);u2=math.fsum(float(z[i]) for i in row['ids'])
                    gamma2=row['factor']*(1. if abs(residual2)<=1 else 1./abs(residual2));d2=1.+gamma2*u2
                    s['w']=np.minimum(4*config.KAPPA,np.maximum(-4*config.KAPPA,s['w']+(gamma2*residual2/d2)*z))
                    s['extra_updates']+=1
                    s['extra']['repeat_selections']+=int(s['extra']['previous_selected']==row['arrival_index']);s['extra']['previous_selected']=row['arrival_index']
                    result.update(replay_index=chosen,extra_performed=True,extra_ids=row['ids'],extra_target=row['target'])
                s['extra']['scans']+=len(s['buffer'])
                event=dict(chosen=chosen,buffer_size=len(s['buffer']),selected_arrival=None if chosen is None else row['arrival_index'])
                s['buffer'].append(dict(ids=list(ids),target=target,factor=gate['factor'],arrival_index=s['updates']))
                if len(s['buffer'])>32:s['buffer'].pop(0)
            elif self.arm.startswith('ROLE'):
                a,b=ids[0],ids[1]-15;s['seen_left'][a]=True;s['seen_right'][b]=True
                found=next((i for i,z in enumerate(s['queue']) if z['left']==b and z['right']==a),None) if a!=b else None
                if found is not None:
                    prev=s['queue'].pop(found);agree=abs(target-prev['target'])<=1
                    s['agree']+=int(agree);s['disagree']+=int(not agree)
                elif a!=b:
                    s['queue'].append(dict(left=a,right=b,target=target))
                    if len(s['queue'])>32:s['queue'].pop(0)
                eta=.05*max(0.,(s['agree']-s['disagree'])/(s['agree']+s['disagree']+4))
                if self.arm=='ROLE-FIXED':eta=.05
                permutation=list(range(15));random.Random(20261011077).shuffle(permutation)
                coupled=[]
                for i in sorted(set((a,b))):
                    j=permutation[i] if self.arm=='ROLE-PERMUTED' else i
                    if not (s['seen_left'][i] and s['seen_right'][i]):continue
                    left,right=float(s['w'][i]),float(s['w'][15+j]);mid=(left+right)/2;difference=(left-right)/2
                    if eta>0:s['w'][i]=mid+(1.-eta)*difference;s['w'][15+j]=mid-(1.-eta)*difference
                    coupled.append([i,j])
                s['extra']['changed']+=len(coupled) if eta>0 else 0
                event=dict(paired=found is not None,eta=eta,coupled=coupled,agree=s['agree'],disagree=s['disagree'])
        s['updates']+=1
        if self.arm not in config.BASELINES:s['extra']['events']+=1
        if self.arm!='COV-R':
            if gate['eligible']:
                s['history'].append(int(gate['conflict']))
                if len(s['history'])>32:s['history'].pop(0)
            # Re-find the SUPPORT key independently; ROLE's found is a different queue.
            idx=next((i for i,row in enumerate(s['cache']) if row['key']==key),None)
            if idx is None:
                s['cache'].append(dict(key=key,answers=[target]))
                if len(s['cache'])>32:s['cache'].pop(0)
            else:
                s['cache'][idx]['answers'].append(target)
                if len(s['cache'][idx]['answers'])>8:s['cache'][idx]['answers'].pop(0)
        if self.arm in config.BASELINES:
            self.scalar_ops+=11000+(256 if self.arm!='COV-R' else 0)
            self.traffic_bytes+=96000+(1024 if self.arm!='COV-R' else 0)
        else:
            self.scalar_ops+=11256;self.traffic_bytes+=97024
            if self.arm.startswith('RETRO'):
                self.scalar_ops+=11000+2*31**3+len(event['changes'])*3000
                self.traffic_bytes+=192000+len(event['changes'])*24000
            elif self.arm.startswith('COVER'):
                self.scalar_ops+=3000+len(s['buffer'])*200
                self.traffic_bytes+=12000+len(s['buffer'])*1200
            elif self.arm.startswith('ROLE'):
                self.scalar_ops+=1500;self.traffic_bytes+=4096
            elif self.arm.startswith('KCI'):
                self.scalar_ops+=500;self.traffic_bytes+=2048
        result.update(performed=True,gate=gate,mechanism=event);return result
    def require_seal(self,actual):
        if actual!=self.digest():raise AssertionError('independent full-state transition mismatch')
    def invariants(self):
        s=self.s
        for k in ('w','p','h','b'):
            if k in s and not np.isfinite(s[k]).all():raise AssertionError('nonfinite '+k)
        if not np.array_equal(s['p'],s['p'].T):raise AssertionError('P symmetry')
        np.linalg.cholesky(s['p'])
        if np.max(np.abs(s['w']))>4*config.KAPPA+1e-12:raise AssertionError('bounded weight')
        for k in ('cache','history','buffer','ledger','queue'):
            if k in s and len(s[k])>32:raise AssertionError('bounded '+k)
        if self.arm.startswith('RETRO'):
            np.linalg.cholesky(s['h'])
            if np.max(np.abs(s['p']@s['h']-np.eye(31)))>1e-9:raise AssertionError('information inverse')
            if any(not 0<r['effective']<=r['arrival'] for r in s['ledger']):raise AssertionError('nonnegative ledger weights')
