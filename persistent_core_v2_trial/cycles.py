"""Three explicit development cycles. Technical bugs may be repaired; no performance tuning."""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import bootstrap
from v2_core import TrialCore, ARMS
from centered_core import CenteredCore
from v2_fixture import make_world, SCIENCE_WORLDS, DEVELOPMENT_WORLDS, permitted
from v2_experiment import instrument, record
from v2_integrity import identity, source_map
from v2_audit import audit_receipt, read
from v2_checkpoint import load, seal
from compact_storage import atomic_json

ROOT=bootstrap.ROOT


def reject(f):
    try: f()
    except (ValueError,AssertionError,KeyError,IndexError): return
    raise AssertionError('tamper unexpectedly accepted')


def child(world,arm,assay,folder,*,short=False,stop=None,resume=False):
    args=[sys.executable,'-u',str(ROOT/'worker.py'),'--development','--world',str(world),
          '--arm',arm,'--assay',assay,'--folder',str(folder)]
    if short: args.append('--short')
    if stop: args+=['--stop',str(stop)]
    if resume: args.append('--resume')
    log=folder/f'{world}_{arm}_{assay}-{time.time_ns()}.log'; folder.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:
        result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=3600)
    if result.returncode: raise RuntimeError(f'technical worker failed: {log}')
    return folder/'receipts'/f'{world}_{arm}_{assay}.json.gz'


def cycle1():
    began=time.monotonic(); world=make_world(500101,'lifetime')
    for w in DEVELOPMENT_WORLDS+SCIENCE_WORLDS:
        for assay in ('lifetime','reuse'): make_world(w,assay)
    # V1 wrapper parity against the immutable actual parent, not a toy model.
    control=TrialCore('V1'); original=CenteredCore(); a=instrument(control); b=instrument(original)
    for event in world['events'][:16]:
        ra=record(control,event,'W',a); rb=record(original,event,'W',b)
        if ra['prediction']!=rb['prediction'] or control.bank_digests()!=original.bank_digests():
            raise AssertionError('V1 did not reproduce actual parent')
    for arm in ARMS:
        core=TrialCore(arm); tr=instrument(core)
        assert len({id(m.fly.m.B.data) for m in core.models})==8
        for stage in ('old','new','revision'):
            branch={'old':'N_old','new':'N_new','revision':'N_revision'}[stage]
            e=dict(world['events'][0],stage=stage,index=0)
            c=core.clone(); r=record(c,e,branch,instrument(c))
            assert all(call['write'] is False for call in r['actual_calls'])
            assert r['write']['evidence_before']==r['write']['evidence_after']
        clone=core.clone(); clone.conflicts[0,0]=1
        assert core.conflicts[0,0]==0 and clone.state_digest()!=core.state_digest()
    # Force the declared slow-state conflict, then independently check the
    # patch on the actual native signed transition. This is E0 synthetic state.
    first=world['events'][0]; c=TrialCore('V1'); record(c,first,'W',instrument(c))
    trial=TrialCore('REPLACE')
    for j,m in enumerate(trial.private): m.fly.m.slow[:]=-np.sign(c.private[j].fly.m.slow)
    trial.conflicts[:]=2
    row=record(trial,first,'W',instrument(trial))
    assert sum(rep['changed'] for rep in row['write']['replacements'])>0
    # Real short receipts drive the hostile independent auditor.
    folder=ROOT/'results/development/cycle1'
    path=child(500101,'BOTH','lifetime',folder,short=True); doc=read(path); audit_receipt(doc,identity(),complete=False)
    cases=[]
    for field,value in [('learn',False),('prediction_precedes_outcome',False),('predicted_at',-1.)]:
        d=copy.deepcopy(doc); d['branches']['W']['records'][0][field]=value; cases.append(d)
    d=copy.deepcopy(doc); d['branches']['W']['records'][0]['write']['outcome']=51 if world['events'][0]['outcome']!=51 else 48; cases.append(d)
    d=copy.deepcopy(doc); d['branches']['W']['records'][0]['write']['s'][0]+=.1; cases.append(d)
    d=copy.deepcopy(doc); d['branches']['W']['records'][0]['actual_calls'][4]['write']=False; cases.append(d)
    d=copy.deepcopy(doc); d['branches']['W']['records'][0]['prediction']['emitted']=255; cases.append(d)
    d=copy.deepcopy(doc); d['fixture_sha256']='0'*64; cases.append(d)
    d=copy.deepcopy(doc); d['branches']['W']['births'][1]['fly_id']=d['branches']['W']['births'][0]['fly_id']; cases.append(d)
    for d in cases: reject(lambda d=d:audit_receipt(d,identity(),complete=False))
    # Explicitly reject the historic old-relation N branch writing bug.
    assert permitted('N_old','old') is False
    for branch in ('W','N_old','N_new','N_revision'):
        for stage in ('old','new','revision'):
            expected=branch=='W' or branch!='N_'+stage
            assert permitted(branch,stage)==expected
    result={'cycle':1,'verdict':'PASS','parent_parity_records':16,'actual_birth_independence':True,
            'all_stage_clamps_checked':True,'replacement_activated_and_algebra_checked':True,
            'hostile_receipt_rejections':len(cases),'heldout_exposure_roster_checked':99,
            'review':'E0 checks only. No parameters or endpoints selected from accuracy.',
            'elapsed_s':time.monotonic()-began,'identity_at_trial':identity()}
    atomic_json(ROOT/'results/CYCLE1.json',result); return result


def cycle2():
    began=time.monotonic(); folder=ROOT/'results/development/cycle2'; jobs=[(a,s) for a in ARMS for s in ('lifetime','reuse')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(child,500102,a,s,folder) for a,s in jobs]
        paths=[f.result() for f in futures]
    docs=[read(p) for p in paths]
    for d in docs: audit_receipt(d,identity())
    estimated=sum(d['worker_s'] for d in docs)*96
    sizes=sum(p.stat().st_size for p in paths)*96
    if estimated>288000 or sizes>4294967296 or max(d['peak_rss_bytes'] for d in docs)>1073741824:
        raise RuntimeError('technical resource gate failed; frozen scientific roster not reduced')
    result={'cycle':2,'verdict':'PASS','complete_technical_jobs':8,'complete_lives':24,
            'four_worker_full_life_test':True,'all_receipts_independently_audited':True,
            'resource_trials':[{'arm':d['arm'],'assay':d['assay'],'worker_s':d['worker_s'],'cpu_s':d['cpu_s'],
                                'peak_rss_bytes':d['peak_rss_bytes']} for d in docs],
            'estimated_science_worker_h':estimated/3600,'estimated_receipt_bytes':sizes,
            'review':'Full clocks, all lifecycle stages, actual write flags and prefeedback outputs audited. No performance tuning.',
            'elapsed_s':time.monotonic()-began,'identity_at_trial':identity()}
    atomic_json(ROOT/'results/CYCLE2.json',result); return result


def cycle3():
    began=time.monotonic(); folder=ROOT/'results/development/cycle3'; results={}
    for arm in ARMS:
        uninterrupted=folder/arm/'uninterrupted'; resumed=folder/arm/'resumed'
        a=child(500103,arm,'lifetime',uninterrupted,short=True)
        child(500103,arm,'lifetime',resumed,short=True,stop=32)
        cp=resumed/'checkpoints'/f'500103_{arm}_lifetime.npz'
        core,cursor=load(cp,arm,identity()); assert cursor['next_record']==32
        reject(lambda:load(cp,arm,'0'*64))
        with np.load(cp,allow_pickle=False) as z: arrays={k:z[k].copy() for k in z.files}
        changed={k:v.copy() for k,v in arrays.items()}; changed['conflicts'].flat[0]^=1
        bad=folder/f'{arm}_bad_evidence.npz'; np.savez_compressed(bad,**changed)
        reject(lambda:load(bad,arm,identity()))
        changed={k:v.copy() for k,v in arrays.items()}; meta=json.loads(changed['metadata_json'].tobytes())
        meta.pop('seal'); meta['cursor']['next_record']+=1; meta['seal']=seal(meta)
        changed['metadata_json']=np.frombuffer(json.dumps(meta).encode(),dtype=np.uint8)
        bad=folder/f'{arm}_bad_cursor.npz'; np.savez_compressed(bad,**changed)
        reject(lambda:load(bad,arm,identity()))
        b=child(500103,arm,'lifetime',resumed,short=True,resume=True)
        ad,bd=read(a),read(b)
        # Process CPU timings differ by construction; scientific transitions do not.
        ar=copy.deepcopy(ad['branches']['W']['records']); br=copy.deepcopy(bd['branches']['W']['records'])
        for rows in (ar,br):
            for r in rows:
                r.pop('cpu_input_predict_s'); r.pop('cpu_observe_flush_s')
        assert ar==br and ad['branches']['W']['final_state_digest']==bd['branches']['W']['final_state_digest']
        before=b.read_bytes(); child(500103,arm,'lifetime',resumed,short=True,resume=True); assert b.read_bytes()==before
        results[arm]={'fresh_process_exact_resume':True,'records_compared':64,
                      'tampered_evidence_and_resealed_cursor_rejected':True,'completed_receipt_skip':True}
    # Test inference edge cases before science, including the unresolved rule.
    from v2_analysis import interval
    assert interval([0.]*96)['lower'] is None
    assert interval([i/96 for i in range(96)])['lower'] is not None
    result={'cycle':3,'verdict':'PASS','arms':results,'zero_variance_not_promoted':True,
            'review':'No science run. Cursor/evidence parity and restart protections close execution risks.',
            'elapsed_s':time.monotonic()-began,'identity_at_trial':identity()}
    atomic_json(ROOT/'results/CYCLE3.json',result); return result


def qualify():
    current=identity(); bound={}
    for n in (1,2,3):
        p=ROOT/f'results/CYCLE{n}.json'; d=json.loads(p.read_text())
        if d['verdict']!='PASS' or d['identity_at_trial']!=current: raise ValueError('stale/missing accepted cycle')
        bound[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    import platform,numpy,scipy,numba
    versions=(platform.python_version(),numpy.__version__,scipy.__version__,numba.__version__)
    if versions!=('3.11.5','2.2.6','1.14.1','0.61.2'): raise ValueError('runtime differs from qualified pinned environment')
    atomic_json(ROOT/'SOURCE_LOCK.json',{'identity':current,'files':source_map()})
    atomic_json(ROOT/'results/QUALIFICATION.json',{'verdict':'PASS','identity':current,'cycles_completed':3,
                'cycle_receipts':bound,'runtime':versions,'final_source_frozen':True,'science_not_yet_launched':True})
    return {'verdict':'PASS','identity':current}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--cycle',type=int,choices=(1,2,3)); p.add_argument('--qualify',action='store_true'); a=p.parse_args()
    result=qualify() if a.qualify else {1:cycle1,2:cycle2,3:cycle3}[a.cycle]()
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
