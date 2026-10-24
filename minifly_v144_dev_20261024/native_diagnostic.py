"""V144 instrumentation of hash-identical V143 science; no update changes."""
import argparse
import hashlib
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from v144_audit import HERE, ROOT, Ledger, write_json, validate
ledger = None
sys.path.insert(0,str(ROOT/'minifly_learned_bank_v1_20261024'))
import numpy as np
import lb1
import data as parent_data
import run as parent_run


def full_state(model):
    # Includes reader/native arrays and scalar state, pending writes, frequency,
    # centring state and fixed matrices omitted by the parent's small state hash.
    values=[model.time,model.records,model.theta,model.freq,model.fe,model.log]
    for bank in model._banks():
        values += [(c.fly,c.brain_t,c.pending_t,c.pending_x,c.teach_seen) for c in bank.comp]
    if model.layer is not None:
        values += [model.layer.W1,model.layer.b1,model.layer.W2,model.layer.G,
                   model.layer.e_bar,model.layer.centre_tau]
    return hashlib.sha256(pickle.dumps(values,protocol=5)).hexdigest()


class TelemetryModel(lb1.Model):
    def train(self, prompt, taught_byte, true_byte):
        n=self.records
        result=super().train(prompt,taught_byte,true_byte)
        # Result contains the pre-update margins actually used by parent train.
        y=lb1.bits_of(taught_byte)
        telemetry={'record':self.records,'target':int(taught_byte),
                   'strength':lb1.strength(n),'S_margin':result['mS'].tolist(),
                   'S_residual':((np.tanh(result['mS']/2)-y)/2).tolist()}
        if 'mL' in result:
            e=(np.tanh(result['mL']/2)-y)/2
            telemetry.update(L_margin=result['mL'].tolist(),L_residual=e.tolist())
        return telemetry


def install_logging(model, sink):
    # Capture existing update inputs; call the exact parent method once.
    original=model.layer.update
    def update(p,a,e,eta,lam):
        gate=a>0
        contributions=(model.layer.G*e[None,:])*gate[:,None]
        d=contributions.sum(axis=1)
        row={'eta':eta,'relu_fraction':float(gate.mean()),
             'projected_error_norm':float(np.linalg.norm(d)),
             'feature_update_l2':float(eta*np.linalg.norm(d)*np.linalg.norm(p)),
             'decay_update_l2':float(eta*lam*np.linalg.norm(model.layer.W1)),
             'bias_update_l2':float(eta*np.linalg.norm(d)),
             'per_bit_update_l2':(eta*np.linalg.norm(contributions,axis=0)*np.linalg.norm(p)).tolist(),
             'nonfinite':not bool(np.isfinite(d).all())}
        value=original(p,a,e,eta,lam)
        sink.append(row)
        return value
    model.layer.update=update


def snapshot(model, world, label, records, out, previous):
    before=full_state(model)
    taught={p:parent_data.answer('PAR',*p) for p in world['train_pairs']}
    acc,outputs=parent_run.accuracy(model,'PAR',world,taught)
    meters,codes=parent_run.meters(model,'PAR',world,previous)
    hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in codes.items()}
    same=[];different=[]
    labels=np.array([parent_data.answer('PAR',*p) for p in parent_run.ALL])
    K=parent_run.jaccard_matrix(codes['L'])
    for i in range(100):
        for j in range(i+1,100):
            (same if labels[i]==labels[j] else different).append(K[i,j])
    p=np.array([lb1.trace_for(parent_data.prompt(*p)) for p in parent_run.ALL])
    hidden=np.maximum(p@model.layer.W1.T+model.layer.b1,0)
    eigen=np.linalg.eigvalsh(np.cov(hidden,rowvar=False))
    positive=np.maximum(eigen,0); total=positive.sum()
    q=positive/total if total else positive
    effective_rank=float(np.exp(-np.sum(q[q>0]*np.log(q[q>0])))) if total else 0
    assert full_state(model)==before,'probe changed persistent state'
    np.savez_compressed(out/(label+'_codes.npz'),**codes,eigenvalues=eigen)
    return dict(label=label,records=records,accuracy=acc,outputs=outputs,meters=meters,
                code_hashes=hashes,within_class_jaccard=float(np.mean(same)),
                between_class_jaccard=float(np.mean(different)),hidden_effective_rank=effective_rank,
                hidden_norm=float(np.linalg.norm(hidden)),state_hash=before),codes


def parity_fixture():
    world=parent_data.make_world('D2')
    original=lb1.Model('LEARN',world['model_seed'],eta0=.03)
    instrumented=TelemetryModel('LEARN',world['model_seed'],eta0=.03)
    updates=[];install_logging(instrumented,updates)
    equal=[]
    for pair in parent_data.schedule(world,1)[:8]:
        p=parent_data.prompt(*pair); y=parent_data.answer('PAR',*pair)
        original.train(p,y,y);instrumented.train(p,y,y)
        equal.append(full_state(original)==full_state(instrumented))
    assert all(equal),'instrumentation changed native state'
    before=full_state(instrumented)
    instrumented.probe(b'2+3=',49)
    assert full_state(instrumented)==before
    return {'record_attempts':16,'native_histories':2,'per_record_state_equal':equal,
            'probe_nonmutating':True,'feature_updates_logged':len(updates)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['LEARN','FROZEN'],required=True)
    a=ap.parse_args(); name='B_'+a.arm
    global ledger
    ledger=Ledger(name,'B');start=time.process_time()
    lock=json.loads((HERE/'cycles/CYCLE1_SOURCE_LOCK.json').read_text())
    check=validate(lock,[],[])
    if not check['valid']:raise RuntimeError(check)
    world=parent_data.make_world('D2');model=TelemetryModel(a.arm,world['model_seed'],eta0=.03)
    updates=[];install_logging(model,updates)
    out=HERE/'results'/name;out.mkdir(parents=True,exist_ok=False)
    snapshots=[];previous=None;events=[]
    perpass=len(world['train_pairs']); targets={1,5,10,20,25,50,75,100,125,150}
    initial,previous=snapshot(model,world,'birth',0,out,previous);snapshots.append(initial)
    for i,pair in enumerate(parent_data.schedule(world,150)):
        if time.process_time()-start>2650:raise RuntimeError('per-arm budget: INCOMPLETE_DIAGNOSTIC')
        ledger.emit('record_start',record=i+1)
        p=parent_data.prompt(*pair); y=parent_data.answer('PAR',*pair)
        telemetry=model.train(p,y,y)
        telemetry['update']=updates[-1] if a.arm=='LEARN' and updates else None
        events.append(telemetry)
        ledger.emit('record_complete',record=i+1,target_tokens=1,input_bytes=len(p))
        if (i+1)%perpass==0 and (i+1)//perpass in targets:
            label='pass'+str((i+1)//perpass)
            cp,previous=snapshot(model,world,label,i+1,out,previous);snapshots.append(cp)
            write_json(out/'PROGRESS.json',{'arm':a.arm,'records':i+1,'checkpoint':cp,'cpu_s':time.process_time()-start})
    model.rest(3600)
    cp,previous=snapshot(model,world,'rest1h',model.records,out,previous);snapshots.append(cp)
    # Event arrays stay local. Compact pass-window aggregates can be independently recomputed.
    (out/'EVENTS.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    windows=[]
    for lo,hi in zip([0,1,5,10,20,25,50,75,100,125],[1,5,10,20,25,50,75,100,125,150]):
        es=events[lo*perpass:hi*perpass]
        row={'passes':[lo,hi],'records':len(es),
             'mean_abs_residual_L':np.mean([np.abs(e['L_residual']) for e in es],axis=0).tolist(),
             'mean_abs_margin_L':np.mean([np.abs(e['L_margin']) for e in es],axis=0).tolist()}
        us=[e['update'] for e in es if e['update']]
        if us:
            for key in ['feature_update_l2','decay_update_l2','bias_update_l2','projected_error_norm','relu_fraction']:
                row[key]=float(np.mean([u[key] for u in us]))
            row['per_bit_update_l2']=np.mean([u['per_bit_update_l2'] for u in us],axis=0).tolist()
        windows.append(row)
    result={'id':name,'complete':True,'metrics':{'snapshots':snapshots,'windows':windows},
            'records':model.records,'cpu_s':time.process_time()-start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'lock_sha256':hashlib.sha256((HERE/'cycles/CYCLE1_SOURCE_LOCK.json').read_bytes()).hexdigest()}
    write_json(out/'RESULT.json',result);ledger.emit('complete',records=model.records)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if ledger:ledger.emit('failure',error=repr(e))
        raise
