"""Cycle 3: audit and test listening; preserve prerequisite failures, then revise.

No oracle/arithmetic efficacy run can start until this qualification passes.
"""
from v51_common import *
from representation import Codes,Sequence
REPLIES=[bytes.fromhex(s) for s in json.loads((HERE/'PARAMETER_LOCK.json').read_text())['fixture_replies_hex']]
from v51_core import Learner
import importlib.util,resource,pickle,gc,math
# Unique module name avoids the inherited v3 data.py path.
_s=importlib.util.spec_from_file_location('v51_task_evaluator',HERE/'data.py')
data=importlib.util.module_from_spec(_s);_s.loader.exec_module(data)

def exposure(records,byte_count):
    sizes=np.array([len(p)+len(a) for p,a in records]);ends=np.cumsum(sizes)
    full=int(np.searchsorted(ends,byte_count,side='right'));used=records[:full]
    prev=int(ends[full-1]) if full else 0
    answer_bytes=sum(len(a) for p,a in used)
    if full<len(records):answer_bytes+=max(0,byte_count-prev-len(records[full][0]))
    return dict(payload_bytes=byte_count,complete_record_presentations=full,
        partial_record_presentations=int(byte_count>prev),answer_payload_bytes=answer_bytes,
        unique_oriented_prompts=len({p for p,a in used}),
        distinct_semantic_orbits=len({tuple(sorted(map(int,p[:-1].split(b'+')))) for p,a in used}))

def class_diagnostic(codes,dev):
    tab=codes.table();symbols=sorted(set(tab[np.flatnonzero(codes.counts)].tolist()))
    classes=sorted(set(int(codes.classes[u]) for u in symbols));A=len(symbols);C=len(classes)
    trans=codes.right[np.ix_(symbols,symbols)];N=float(trans.sum())
    byclass=np.array([sum((trans[i] for i,u in enumerate(symbols) if codes.classes[u]==c),start=np.zeros(A)) for c in classes])
    pc=(byclass+.5)/(byclass.sum(1,keepdims=True)+.5*A)
    pi=(trans+.5)/(trans.sum(1,keepdims=True)+.5*A)
    one=(trans.sum(0)+.5)/(trans.sum()+.5*A)
    ix={u:i for i,u in enumerate(symbols)};cx={c:i for i,c in enumerate(classes)}
    nll_c=nll_i=nll_0=0.;events=0;boundaries=0;runs=[]
    for p,a in dev:
        seq=[int(tab[b]) for b in p+a];run=1
        for u,v in zip(seq[:-1],seq[1:]):
            if u not in ix or v not in ix:continue
            nll_c-=math.log2(pc[cx[int(codes.classes[u])],ix[v]])
            nll_i-=math.log2(pi[ix[u],ix[v]])
            nll_0-=math.log2(one[ix[v]]);events+=1
            if codes.classes[u]!=codes.classes[v]:boundaries+=1;runs.append(run);run=1
            else:run+=1
        runs.append(run)
    # Prelocked approximate two-part/BIC description lengths. Count both
    # left/right active prototype degrees of freedom, class predictor and labels.
    class_params=3*C*(A-1);identity_params=A*(A-1)
    penalty_c=.5*class_params*math.log2(max(N,2))+A*math.log2(6)+C*math.log2(6)
    penalty_i=.5*identity_params*math.log2(max(N,2))
    return dict(active_byte_types=A,occupied_classes=C,classes=classes,training_transitions=N,dev_transitions=events,
        class_nll_bits=nll_c,identity_nll_bits=nll_i,collapsed_nll_bits=nll_0,
        class_description_bits=nll_c+penalty_c,identity_description_bits=nll_i+penalty_i,
        class_parameter_penalty_bits=penalty_c,identity_parameter_penalty_bits=penalty_i,
        boundary_density=boundaries/max(1,events),singleton_run_fraction=sum(n==1 for n in runs)/max(1,len(runs)),
        mean_run_length=float(np.mean(runs)),noncollapsed_predictive_gain=bool(nll_c<nll_0),
        compressed_predictive_description=bool(nll_c+penalty_c<nll_i+penalty_i))

def integration(codes,seed,selected):
    """Bounded arbitrary-byte E1 after the actual representation qualifies."""
    model=Learner(codes,65536,seed,selected['a_max'],selected['W'])
    fixtures=[(bytes([31+i,255-i]),a) for i,a in enumerate(REPLIES)]
    # Threshold homeostasis is calibrated without target loss, then frozen.
    for p,a in fixtures:
        s=Sequence(codes)
        for b in p+a:s.feed(b);model.geometry.calibrate(s)
    model.geometry.frozen=True;rows=[];records=0
    for rnd in range(1,257):
        for p,a in fixtures:model.train(p,a);records+=1
        if rnd in (1,4,16,64,256):
            scored=data.score(model,fixtures);rows.append(dict(round=rnd,**scored))
            print(json.dumps(dict(stage='integrated_port',seed=seed,round=rnd,exact=scored['exact'])),flush=True)
    return dict(full_record_applications=records,qualified=rows[-1]['exact']==1,curves=rows,native=model.engine.stats())

def run():
    start=time.process_time();save(Path('cycles/cycle3/SOURCE_BEFORE.json'),source_lock())
    q0=json.loads((HERE/'cycles/cycle2/AUDIT.json').read_text());assert q0['accepted']
    records=0;partial=0;births=[];listening_bytes=0
    if not q0['Q0_passed']:
        out=dict(accepted=True,Q1_passed=False,status='BLOCKED_BY_Q0',full_record_applications=0,listening_payload_bytes=0)
        save(Path('cycles/cycle3/AUDIT.json'),out);return out
    for seed in (202610205101,202610205102):
        train=data.stream(seed,20000,'train');dev=data.stream(seed,512,'dev')
        for p,a in train:
            aa,bb=map(int,p[:-1].split(b'+'));assert data.part(aa,bb,seed)<8 and data.part(bb,aa,seed)<8
        for p,a in dev:
            aa,bb=map(int,p[:-1].split(b'+'));assert data.part(aa,bb,seed)==8
        blob=b''.join(p+a for p,a in train);assert len(blob)>=200000
        candidates=[];qualified=[]
        for eta in (.001,.005):
            for gain in (0.,.01):
                c=Codes(seed,eta,gain);trace=[];maps=[]
                for end in range(10000,200001,10000):
                    c.hear(blob[end-10000:end]);listening_bytes+=10000
                    diag=c.diagnostics();maps.append(diag['winner_map'])
                    if end in (50000,100000,200000):
                        stable=maps[-1]==maps[-2]
                        present0=set(blob[end-20000:end-10000]);present1=set(blob[end-10000:end])
                        covered=set(map(int,diag['winner_map']))<=present0&present1
                        ok=stable and covered and diag['alias_count']==0
                        trace.append(dict(**diag,stable=stable,both_blocks_cover_types=covered,qualified=ok,
                            exposure=exposure(train,end)))
                        print(json.dumps(dict(stage='byte_identity',seed=seed,eta=eta,gain=gain,bytes=end,aliases=diag['alias_count'],stable=stable,qualified=ok)),flush=True)
                        if ok:
                            c.identity_frozen=True;qualified.append((diag['type_weighted_reconstruction'],end,eta,gain,c));break
                item=dict(eta=eta,gain=gain,trace=trace,qualified=c.identity_frozen)
                candidates.append(item);save(Path('cycles/cycle3')/f'M2_{seed}_{eta}_{gain}.json',item)
        b=dict(seed=seed,M2_candidates=candidates,M2_qualified=bool(qualified))
        if not qualified:
            b.update(status='IDENTITY_NOT_QUALIFIED',M3_run=False,integration_run=False)
            births.append(b);continue
        _,_,eta,gain,c=min(qualified,key=lambda v:v[:4]);b['selected_byte_rates']=dict(eta=eta,gain=gain)
        maps=[];context_records=0;context_bytes=0;m3=[];stable=False
        for end in range(10000,200001,10000):
            batch=[]
            while context_records<len(train):
                p,a=train[context_records]
                if context_bytes+len(p)+len(a)>end:break
                batch.append((p,a));context_records+=1;context_bytes+=len(p)+len(a)
            c.hear_context(batch);listening_bytes+=sum(len(p)+len(a) for p,a in batch)
            maps.append(c.classes.copy())
            if end in (50000,100000,200000):
                stable=len(maps)>1 and np.array_equal(maps[-1],maps[-2])
                diagnostic=class_diagnostic(c,dev)
                passed=stable and diagnostic['noncollapsed_predictive_gain'] and diagnostic['compressed_predictive_description']
                row=dict(byte_budget=end,actual_context_bytes=context_bytes,complete_record_presentations=context_records,
                    stable=stable,qualified=passed,diagnostic=diagnostic,assignments=c.classes.tolist())
                m3.append(row)
                print(json.dumps(dict(stage='context_classes',seed=seed,budget=end,stable=stable,qualified=passed,occupied=diagnostic['occupied_classes'])),flush=True)
                if passed:c.context_frozen=True;break
        b.update(M3_run=True,M3=m3,M3_qualified=c.context_frozen)
        with (HERE/'cycles/cycle3'/f'codes_{seed}.pkl').open('wb') as f:pickle.dump(c,f,protocol=5)
        if c.context_frozen:
            result=integration(c,seed,q0['selected']);records+=result['full_record_applications']
            b.update(integration_run=True,integration=result,status='INTEGRATED_PORT_QUALIFIED' if result['qualified'] else 'INTEGRATED_PORT_NOT_QUALIFIED')
        else:b.update(integration_run=False,status='SEGMENTATION_NOT_QUALIFIED')
        births.append(b);save(Path('cycles/cycle3')/f'BIRTH_{seed}.json',b);del c;gc.collect()
    passed=all(b.get('M3_qualified',False) and b.get('integration',{}).get('qualified',False) for b in births)
    out=dict(accepted=True,status='INTEGRATED_PREREQUISITES_QUALIFIED' if passed else 'NO_ADVANCE_PREREQUISITE_FAILED',
        Q1_and_integration_passed=passed,births=births,full_record_applications=records,partial_native_events=partial,
        listening_payload_bytes=listening_bytes,scientific_records=0,
        oracle_run=False,arithmetic_efficacy_run=False,formal_campaign_ready=False,
        CPU_seconds=time.process_time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_after=source_lock())
    save(Path('cycles/cycle3/AUDIT.json'),out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['births','source_after']}),flush=True)
    return out
if __name__=='__main__':run()
