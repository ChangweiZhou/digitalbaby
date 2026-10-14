"""Independent numeric replay of logged bytes; no learner or temporal calls."""
import math
import numpy as np
import settings,data
from representation import SharedEncoder

def require(ok,why):
    if not ok:raise AssertionError(why)
def moved(f,s,dt,mode):
    if mode=='DECAY':return f*np.exp(-dt/3600.),s*np.exp(-dt/(30.*86400.))
    if mode=='STABLE':return f.copy(),s.copy()
    proposed=f*(1.-np.exp(-dt/3600.));room=np.where(proposed>=0,2.-s,-2.-s)
    carried=np.where(proposed>=0,np.minimum(proposed,room),np.maximum(proposed,room))
    return f-carried,s+carried

def verify_equations(receipt):
    enc=SharedEncoder();n=enc.n;checked=teaching=time_events=0;maxerr=0.;peak_bound=0.
    def support(pair):
        ids=np.flatnonzero((enc.code(pair)>0)&enc.mask)
        return ids,np.ones(len(ids))/math.sqrt(len(ids))
    for mode,out in receipt['arms'].items():
        for branch in settings.BRANCHES:
            f=np.zeros(n);s=np.zeros(n);last=np.zeros(n);visits=np.zeros(n,np.uint32)
            t=0.;bytecount=0
            for phase in settings.PHASES:
                if phase.startswith('day'):t+=86400.
                else:
                    for record,trace in zip(receipt['fixture']['records'][phase],out['training'][phase][branch]):
                        pair=tuple(record['pair']);ids,z=support(pair);pending=None
                        for row in trace:
                            t+=settings.DT;bytecount+=1;checked+=1
                            if row['byte']==61:
                                vf,vs=moved(f[ids],s[ids],t-last[ids],mode);pending=float(z@(vf+vs))
                                pred=row['prediction'];err=abs(pred['numeric_value']-pending);maxerr=max(maxerr,err)
                                require(err<3e-11,'equation pre-feedback read')
                                logits=-.5*(np.arange(9)-pending)**2;expected=np.exp(logits-np.logaddexp.reduce(logits))
                                require(np.max(np.abs(expected-pred['probabilities']))<3e-11,'equation emitted probabilities')
                            if row['observed_target'] is None:
                                require(row['time_transform_l1']==0 and row['clipped_rows']==0,'nonanswer time materialization')
                                continue
                            require(pending is not None,'feedback without prior prediction')
                            before_f=f[ids].copy();before_s=s[ids].copy();vf,vs=moved(before_f,before_s,t-last[ids],mode)
                            transform=float(np.abs(vf-before_f).sum()+np.abs(vs-before_s).sum())
                            expected_teach=0.;clipped=0
                            if branch!='N_OLD' or phase=='new':
                                delta=z*np.clip(row['byte']-48-pending,-1.,1.)/np.sqrt(1.+visits[ids])
                                uf=vf+.2*delta;us=vs+.8*delta
                                clipped=int(np.count_nonzero((np.abs(uf)>2)|(np.abs(us)>2)))
                                nf=np.clip(uf,-2,2);ns=np.clip(us,-2,2)
                                expected_teach=float(np.abs(nf-vf).sum()+np.abs(ns-vs).sum())
                                vf,vs=nf,ns;visits[ids]=np.minimum(visits[ids]+1,65535);teaching+=1
                            require(abs(row['time_transform_l1']-transform)<3e-9,'equation time transport ledger')
                            require(abs(row['write_l1'][0]-expected_teach)<3e-9,'equation teaching ledger')
                            require(row['clipped_rows']==clipped,'equation clipping ledger')
                            time_events+=int(transform>0)
                            f[ids]=vf;s[ids]=vs;last[ids]=t
                            peak_bound=max(peak_bound,float(max(np.abs(f[ids]).max(),np.abs(s[ids]).max())))
                snapshot=out['content_snapshots'][phase][branch]
                for name,want in [('fast',f),('slow',s),('last',last),('visits',visits)]:
                    actual=np.asarray(snapshot[name]);require(actual.shape==(n,) and np.isfinite(actual).all(),'actual snapshot schema')
                    err=float(np.max(np.abs(actual-want)));maxerr=max(maxerr,err)
                    require(err<(1e-7 if name=='last' else 3e-10),'actual operative snapshot '+name)
                state=out['states'][phase][branch]
                require(abs(state['model_time']-t)<1e-7 and state['bytes_seen']==bytecount,'equation clock/budget')
                certificate=state['actual_content_certificate']
                require(abs(certificate['fast_l1']-np.abs(f).sum())<3e-9 and abs(certificate['slow_l1']-np.abs(s).sum())<3e-9 and certificate['visits_total']==int(visits.sum()),'actual content certificate linkage')
                qt=t
                for row in out['probes'][phase][branch]['rows']:
                    qt+=settings.DT*4;ids,z=support(tuple(row['pair']));vf,vs=moved(f[ids],s[ids],qt-last[ids],mode)
                    value=float(z@(vf+vs));logits=-.5*(np.arange(9)-value)**2;p=np.exp(logits-np.logaddexp.reduce(logits))
                    require(np.max(np.abs(p-row['probabilities']))<3e-10,'equation held/taught output linkage')
                    require(row['emitted']==48+int(np.argmax(p)),'equation actual output')
                    qt+=settings.DT;checked+=5
    return dict(verdict='PASS',independent_equation_rows=checked,permitted_target_updates=teaching,time_materializations=time_events,max_numeric_difference=maxerr,max_observed_coordinate=peak_bound,model_trajectory_executions=0)
