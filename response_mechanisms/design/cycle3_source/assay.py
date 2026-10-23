"""Evaluator-only fixture, complete response tensors, and deterministic receipts."""
from __future__ import annotations
import gzip, hashlib, json, os, platform, resource, sys, time
from pathlib import Path
import numpy as np
from mechanism_model import ROOT, A3, ARMS, DEFAULT, System, digest
from fixture import make_world, DT, RECORD_SECONDS


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source_hashes():
    files=list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    out={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    out['a3_source_lock']=sha(A3/'SOURCE_LOCK.json')
    # Immutable imported sources, including package contents, must match original hashes.
    lock=json.loads((A3/'SOURCE_LOCK.json').read_text())
    for name,expected in lock['files'].items():
        if name.startswith(('src/','package/')):
            got=sha(A3/name)
            if got!=expected: raise AssertionError(f'upstream source drift: {name}')
    return out


def runtime_manifest():
    import scipy,numba,pandas
    modules={}
    for name,m in sorted(sys.modules.items()):
        f=getattr(m,'__file__',None)
        if f:
            p=Path(f).resolve()
            if p.suffix=='.py' and A3.parent in p.parents:
                modules[str(p.relative_to(A3.parent))]=sha(p)
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                numba=numba.__version__,pandas=pandas.__version__,imported_repo_modules=modules,
                thread_environment={k:os.getenv(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
                    'MKL_NUM_THREADS','NUMBA_NUM_THREADS')})


def world_doc(world,bouts):
    d=make_world(world)
    for stage, digits in (('old',b'0123'),('new',b'4567')):
        expected=[(b' '*8+bytes((a,43,b,61))).hex() for a in digits for b in digits]
        if d[stage+'_fact']['cues_hex']!=expected:raise AssertionError('probe grid is not row-major')
    records=[]
    for stage in ('old','new'):
        rows=[r for r in d['records'] if r['stage']==stage and r['domain']=='fact'][:16*bouts]
        for i in range(bouts):
            if sorted(r['item'] for r in rows[i*16:(i+1)*16])!=list(range(16)):
                raise AssertionError('incomplete or duplicated bout')
        for r in rows:
            r=dict(r);r['index']=len(records);records.append(r)
    return dict(world=world,bouts=bouts,old_fact=d['old_fact'],new_fact=d['new_fact'],records=records,
                original_fixture_digest=d['digest'],schema='RESPONSE-FACT-LIFE-v1')


def components(v):
    v=np.asarray(v,float).reshape(4,4,4)
    B=v.mean((0,1));F=v.mean(1)-B;G=v.mean(0)-B
    H=v-B-F[:,None,:]-G[None,:,:]
    err=float(np.max(np.abs(v-(B+F[:,None,:]+G[None,:,:]+H))))
    return B,F,G,H,err


def decompose(v,labels):
    v=np.asarray(v,float).reshape(4,4,4)
    B,F,G,H,err=components(v)
    target=2*np.eye(4)[np.array(labels).reshape(4,4)]-1
    tb,tf,tg,th,_=components(target)
    rms=lambda a: float(np.sqrt(np.mean(a*a)))
    g1,g2,g12=map(rms,(F,G,H))
    aligned={}
    for name,c,t in zip(('B','F','G','H'),(B-B.mean(),F,G,H),(tb-tb.mean(),tf,tg,th)):
        norm=rms(t)
        aligned[name]=dict(target_rms=norm,covariance=float(np.mean(c*t)),
                           projection=float(np.mean(c*t))/norm if norm>1e-12 else None)
    return dict(bias_rms=rms(B-B.mean()),g1=g1,g2=g2,g12=g12,
                g12_g1=g12/g1 if g1>1e-12 else None,g12_g2=g12/g2 if g2>1e-12 else None,
                reconstruction_max=err,target_alignment=aligned,
                components=dict(B=B.tolist(),F=F.tolist(),G=G.tolist(),H=H.tolist()))


def probe(system,cues,labels,at):
    before=system.state_digest()
    values=[];raw=[];extra=[];features=[]
    for ch in cues:
        m=system.clone();m.begin_cue(at)
        for k,b in enumerate(bytes.fromhex(ch)):m.cue_byte(b,at+k*DT)
        u,r,phi,j=m.read(at+12*DT)
        values.append(u.tolist());raw.append(r.tolist());extra.append(j.tolist());features.append(phi)
    if system.state_digest()!=before:raise AssertionError('probe changed continuing model')
    features=np.stack(features)
    singular=np.linalg.svd(features,compute_uv=False)
    return dict(probe_at=float(at),response_at=float(at+12*DT),values=values,raw_values=raw,extra_values=extra,homeostasis=system.h.tolist(),
                predictions=np.argmax(values,axis=1).tolist(),
                accuracy=float(np.mean(np.argmax(values,axis=1)==np.array(labels))),
                raw_accuracy=float(np.mean(np.argmax(raw,axis=1)==np.array(labels))),
                decomposition=decompose(values,labels),raw_decomposition=decompose(raw,labels),
                decision_decomposition=decompose(np.array(values)-np.mean(values,axis=1,keepdims=True),labels),
                feature_rank=int(np.sum(singular>max(singular[0]*1e-10,1e-12))),
                feature_singular=singular.tolist(),
                max_phi=float(np.max(np.abs(features))),phi_l2_mean=float(np.linalg.norm(features,axis=1).mean()))


def write_once(path,doc):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=gzip.compress(json.dumps(doc,sort_keys=True,allow_nan=False,separators=(',',':')).encode(),mtime=0)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp');temp.write_bytes(data)
    try:os.link(temp,path)
    finally:temp.unlink()
    return dict(path=str(path.relative_to(ROOT)),bytes=len(data),sha256=sha(path))


def run(arm,world,bouts,params,kind='pilot',destination=None):
    sources=source_hashes()
    resolved_params={**DEFAULT,**params}
    import locked_run
    lock_sha=locked_run.authorize_final(arm,world,bouts,resolved_params) if kind=='final' else None
    t0=time.monotonic();doc=world_doc(world,bouts)
    base=System(arm,resolved_params)
    births=base.births
    predictor_birth=[s.parameter_digest() for s in base.stores]
    systems={b:base.clone() for b in ('W','N_old')}
    if systems['W'].state_digest()!=systems['N_old'].state_digest():raise AssertionError('birth mismatch')
    birth=base.state_digest();del base
    records=[];probes={};nold=16*bouts
    def checkpoint(name,at,sets):
        rows={}
        for branch,s in systems.items():
            rows[branch]={}
            for group in sets:
                f=doc[group+'_fact']
                rows[branch][group]=probe(s,f['cues_hex'],f['labels'],at)
        for group in sets:
            v=np.array(rows['W'][group]['values'])-np.array(rows['N_old'][group]['values'])
            rows.setdefault('paired_delta',{})[group]=dict(values=v.tolist(),
                decomposition=decompose(v,doc[group+'_fact']['labels']),
                decision_decomposition=decompose(v-v.mean(1,keepdims=True),doc[group+'_fact']['labels']))
        probes[name]=rows
    checkpoint('birth',0.,('old','new'))
    for rec in doc['records']:
        idx=rec['index'];stage=rec['stage'];at=idx*RECORD_SECONDS+(86400 if stage=='new' else 0)
        for branch,s in systems.items():
            s.begin_cue(at)
            for k,b in enumerate(bytes.fromhex(rec['cue_hex'])):s.cue_byte(b,at+k*DT)
            when=at+12*DT;h_before=s.h.copy();u,raw,phi,extra=s.read(when)
            sens=s.sensory_digest();dw=s.observe_cue(when,raw)
            write=not(branch=='N_old' and stage=='old')
            ledger=s.teach(rec['answer'],when,write,world,idx,raw,phi)
            if not write and (any(ledger['native_l1']) or ledger['j_write_l1']):raise AssertionError('no-write mutation')
            s.end_record(at+RECORD_SECONDS,at+13*DT)
            records.append(dict(record=idx,branch=branch,stage=stage,cue_hex=rec['cue_hex'],
                answer=rec['answer'],teacher_at=when,end_at=at+RECORD_SECONDS,homeostasis_before=h_before.tolist(),values=u.tolist(),raw_values=raw.tolist(),extra_values=extra.tolist(),
                predicted=int(np.argmax(u)),timing_l1=dw,sensory_before_teacher=sens,**ledger))
        if systems['W'].sensory_digest()!=systems['N_old'].sensory_digest():
            raise AssertionError('teacher-dependent sensory state across branches')
        if idx==nold-1:
            checkpoint('old_end',(idx+1)*RECORD_SECONDS,('old',))
            atday=(idx+1)*RECORD_SECONDS+86400
            for s in systems.values():s.flush(atday)
            checkpoint('old_day',atday,('old',))
        if idx==2*nold-1:
            newend=(idx+1)*RECORD_SECONDS+86400
            checkpoint('new_end',newend,('old','new'))
            for s in systems.values():s.flush(newend+86400)
            checkpoint('final',newend+86400,('old','new'))
    frozen=all([s.parameter_digest() for s in m.stores]==predictor_birth for m in systems.values())
    if not frozen:raise AssertionError('byte predictor changed')
    timing=systems['W'].timing_summary()+systems['N_old'].timing_summary()
    if any(not x['support_unchanged'] or not x['finite_positive'] or x['incoming_mass_max_error']>1e-9 for x in timing):
        raise AssertionError('timing structural certificate failed')
    receipt=dict(schema='RESPONSE-MECHANISMS-RECEIPT-v1',kind=kind,arm=arm,world=world,
                 params=resolved_params,bouts=bouts,fixture=doc,source_hashes=sources,birth_digest=birth,
                 state_budget=systems['W'].state_budget(),births=births,predictor_frozen=frozen,end_timing=timing,lock_sha256=lock_sha,records=records,probes=probes,
                 end_state={b:s.state_digest() for b,s in systems.items()},
                 runtime=runtime_manifest(),
                 resource=dict(wall_seconds=time.monotonic()-t0,process_id=os.getpid(),
                               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    if destination is None:destination=ROOT/'results'/kind/arm/f'{world}.json.gz'
    return {**write_once(destination,receipt),**receipt['resource'], 'arm':arm,'world':world}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--world',type=int,required=True);p.add_argument('--bouts',type=int,required=True)
    p.add_argument('--kind',required=True);p.add_argument('--params',default='{}');p.add_argument('--destination')
    a=p.parse_args();print(json.dumps(run(a.arm,a.world,a.bouts,json.loads(a.params),a.kind,a.destination)),flush=True)
