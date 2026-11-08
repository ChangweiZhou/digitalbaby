"""Three short technical cycles. Never call while another producer is active."""
import itertools
import importlib.util
from collections import Counter
from pathlib import Path
from e16wr_common import ROOT,OUT,read,save,verify_lock,require_machine_free,clean

def main():
    gate=require_machine_free();verify_lock()
    from e16wr_reservoir import Reservoir,answers
    # Cycle1: exact enumeration of AlgorithmR, duplicates, no current-stage draws.
    subsets=Counter()
    class Fixed:
        def __init__(self,a,b):self.values=iter((a,b))
        def randrange(self,n):
            v=next(self.values);assert 0<=v<n;return v
    for a,b in itertools.product(range(3),range(4)):
        r=Reservoir(1,2);r.begin(1,'first');r.rng[1]=Fixed(a,b)
        r.admit(1,[([x],[x+1]) for x in range(4)])
        subsets[tuple(sorted(p[0] for p,t in r.rows[1]))]+=1
        assert r.count[1]==4 and r.past(1)==[]
    assert len(subsets)==6 and set(subsets.values())=={2}
    r=Reservoir(1,8);r.begin(1,'first');r.admit(1,[([1],[2])]*100)
    assert len(r.rows[1])==8 and len(set(r.rows[1]))==1 and r.count[1]==100
    r.begin(2,'second');assert r.past(2)==[1]
    assert answers([[1]],[[99]],256)==([[48,48,59,48,48,256]],[True])
    cycles=[dict(cycle=1,status='PASS',algorithmR_all_six_subsets_equally_likely=True,duplicate_presentations_retained=True,current_stratum_excluded=True)]
    # Cycle2: frozen original versus changed source under its unchanged default.
    import os
    os.environ.update(EXP45_HIDDEN='512',E8_SCHEDULE='blocked',E8_REPLAY_SAMPLING='uniform',E8_REPLAY_TARGETS='self',E8_WD='.25',E8_RUN_PREFIX='',
                      E8_BATCH='64',E8_LR='.001',E8_BUFFER_CAP='1024',E8_REPLAY_SHARE='1/2',E8_SAVE_STAGES='0',E8_DIAGNOSTICS='0',E8_COMPOSE_CHECKPOINTS='0')
    import exp8_e16w2 as old
    import exp8_e16wr as new
    import e14_fast_inference as F
    X,T=old.X,old.T
    X.respond=F.respond
    import hashlib
    def digest(x):
        if T.is_tensor(x):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest())
        if isinstance(x,dict):return {str(k):digest(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [digest(v) for v in x]
        return x
    original,score,tsave=X.update,X.score,T.save
    X.score=lambda m,it,store=None:score(m,it[:8],store)
    folder=OUT/'e16wr_technical';folder.mkdir(exist_ok=True)
    def run(mod,name,sampling='uniform',target='self'):
        targetdir=folder/name;targetdir.mkdir(exist_ok=False)
        (targetdir/'e8_dev_73129.pt').symlink_to(OUT/'e8_dev_73129.pt')
        mod.OUT=targetdir;mod.STAGE_UPDATES=2;mod.COMPOSE_UPDATES=4
        mod.SAVE_STAGES=True;mod.REPLAY_SAMPLING=sampling
        mod.REPLAY_TARGETS=target;mod.RESERVOIR_CAP=8
        trace=[];states=[];prompts=[]
        def update(m,opt,items,local,store=None):
            prompts.append(digest([p for p,t in items]))
            loss=original(m,opt,items,local,store)
            trace.append(digest([m.state_dict(),opt.state_dict(),T.get_rng_state(),loss,store.mem if store else None,store.order if store else None]))
            return loss
        X.update=update
        T.save=lambda state,path:states.append(digest(state))
        try:mod.life('C',73129)
        finally:X.update=original;T.save=tsave
        result=read(targetdir/f'{mod.RUN_PREFIX}e8_life_C{mod.TAG}_73129.json')
        save(targetdir/'trajectory.json',dict(trace=trace,states=states,prompts=prompts))
        return dict(trace=trace,states=states,prompts=prompts,scores=clean(result),folder=str(targetdir))
    try:
        a=run(old,'original');b=run(new,'default')
        assert all(a[k]==b[k] for k in ('trace','states','prompts','scores'))
        assert len(a['trace'])==20
        cycles.append(dict(cycle=2,status='PASS',updates=20,original_default_model_optimizer_torch_python_rng_store_buffers_losses_scores_exact=True,
                           checkpoint_states='HASHED_NOT_SAVED_AS_TENSOR_FILES',panels_per_score=8))
        # Cycle3: exercise both target treatments through all9stages, then negative controller fixtures.
        new.SCHEDULE='late_family'
        a=run(new,'reservoir_self','reservoir','self')
        b=run(new,'reservoir_observed','reservoir','observed')
        assert a['prompts']==b['prompts']
        ar=read(next(Path(a['folder']).glob('*.reservoir.json')))
        br=read(next(Path(b['folder']).glob('*.reservoir.json')))
        assert len(ar)==len(br)==9
        for i,(u,v) in enumerate(zip(ar,br),1):
            assert u['reservoirs']==v['reservoirs']
            assert u['replay']['draws']==v['replay']['draws']
            assert sum(u['replay']['draws'].values())==(0 if i==1 else (4 if i==5 else 2)*32)
            assert all(v['replay'][k]['wrong_full']==0 for k in ('composition','operation'))
            assert u['reservoirs'][-1]['presentations']==(128 if i in (1,5) else 64)
            assert u['reservoirs'][-1]['slots']==8
        assert sum(x['replay']['composition']['n'] for x in ar)>0
        import e16wr_common as C
        tmp=folder/'validator';tmp.mkdir()
        job=dict(receipt='exact_job.json',seed=73129,role='self1024')
        save(tmp/job['receipt'],a['scores']);(tmp/'unfinished_sibling.json').write_text('{')
        prior=C.OUT;C.OUT=tmp
        try:
            C.validate_receipt(job,short=True,require_meta=False)
            bad=dict(a['scores']);bad['stages']=bad['stages'][:-1]
            save(tmp/job['receipt'],bad)
            try:C.validate_receipt(job,short=True,require_meta=False)
            except AssertionError:pass
            else:raise AssertionError('Incomplete receipt accepted')
        finally:C.OUT=prior
        cycles.append(dict(cycle=3,status='PASS',short_updates_each=20,nine_stage_loops=True,
                           input_order_and_reservoirs_and_draw_counts_matched=True,observed_target_error_zero=True,
                           current_observations_only=True,incomplete_receipt_rejected=True,unfinished_sibling_ignored=True))
    finally:X.update=original;X.score=score;T.save=tsave
    save(ROOT/'E16WR_CHECKS.json',dict(status='PASS',gate=gate,cycles=cycles,
                                     scope='Short mechanics/state parity only; full default reproduction still required'))

if __name__=='__main__':main()
