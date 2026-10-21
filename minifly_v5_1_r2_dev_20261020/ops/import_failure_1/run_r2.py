"""Single-process bounded three-cycle controller; never retries a failed run."""
import os,sys,time,json,pickle,copy,subprocess,traceback,collections
from pathlib import Path
START_CPU=time.process_time();START_WALL=time.monotonic()
from r2_common import *
import psutil
from r2_representation import Codes
from r2_core import Learner,calibrate,Frames
from r2_geometry import Geometry
from r2_engine import Engine,DT
from r2_checks import components,geometry_report,response
from r2_data import BIRTHS,build,jsonable,orientations,countermodels,key

class Limit(Exception):pass
def resources():
    return dict(CPU_seconds=time.process_time()-START_CPU,wall_seconds=time.monotonic()-START_WALL,
                RSS_bytes=psutil.Process().memory_info().rss,available_system_bytes=psutil.virtual_memory().available)
def check():
    r=resources()
    if r['CPU_seconds']>=900 or r['wall_seconds']>=1800:raise Limit('total CPU/wall cap')
    if r['RSS_bytes']>2*1024**3 or r['available_system_bytes']<2*1024**3:raise Limit('RSS/headroom admission')
def status(phase,**kw):
    v=dict(status=phase,resources=resources(),**kw);save('STATUS.json',v);print(json.dumps(v),flush=True)
def append(name,v):
    with (HERE/'results'/name).open('a') as f:f.write(json.dumps(v,sort_keys=True,allow_nan=False)+'\n')
def snapshot(cycle):
    p=HERE/'cycles'/cycle;p.mkdir(exist_ok=True);save(p/'SOURCE_LOCK.json',source_lock())
def parent_audit():
    m=json.loads((ROOT/'minifly_v5_1_dev_20261020/PARENT_ASSET_AUDIT.json').read_text())
    failures=[name for name,h in m['files'].items() if sha(ROOT/name)!=h]
    assert not failures,('parent source mismatch',failures)
    p=ROOT/'minifly_v5_1_dev_20261020'
    assert (HERE/'r2_engine.py').read_text()==(p/'engine.py').read_text().replace('from v51_common import','from r2_common import')
    return dict(accepted=True,verified_parent_asset_count=len(m['files']),engine_change='import namespace only',files=m['files'])

def probe(model,records,seed,arm,phase,cohort):
    check();start=time.process_time();before=model.digest();rows=[]
    for p,a in records:
        check();o=model.generate(p);exact=o['payload']==a.hex() and o['ended'] and not o['truncated']
        row=dict(seed=seed,arm=arm,phase=phase,cohort=cohort,prompt=p.hex(),expected=a.hex(),**o,exact=bool(exact))
        rows.append(row);append('PROBES.jsonl',row)
    assert model.digest()==before,'forbidden persistent query write'
    # Mean orientations within an orbit before taking the cohort mean.
    groups=collections.defaultdict(list)
    for r in rows:groups[key(bytes.fromhex(r['prompt']))].append(int(r['exact']))
    result=dict(seed=seed,arm=arm,phase=phase,cohort=cohort,records=len(rows),orbits=len(groups),
                exact=float(np.mean([np.mean(v) for v in groups.values()])),
                ended_fraction=float(np.mean([r['ended'] for r in rows])),
                truncated_fraction=float(np.mean([r['truncated'] for r in rows])),
                correct_payload_length=float(np.mean([len(r['payload'])==len(r['expected']) for r in rows])),
                evaluation_native_events=sum(r['events'] for r in rows),CPU_seconds=time.process_time()-start,readonly=True)
    append('CURVES.jsonl',result);return result

def d0(data):
    seed=data['seed'];t=time.process_time();codes=Codes(seed);ledger={};events=0
    for phase,records in data['listening'].items():
        n=sum(len(p)+len(a) for p,a in records);events+=n+len(records)
        ledger[phase]=dict(full_observed_records=len(records),payload_bytes=n,answer_bytes=sum(len(a) for _,a in records),EOS=len(records),native_clock_events=n+len(records))
        for i,(p,a) in enumerate(records):append('EXPOSURES.jsonl',dict(seed=seed,phase='D0-'+phase,index=i,prompt=p.hex(),observed_answer=a.hex(),native_write=False))
    A=data['listening']['A'];cap=ledger['A']['payload_bytes']
    for p,a in A:check();codes.hear(p+a,cap)
    assert codes.bytes>=2048;codes.identity_frozen=True
    B=data['listening']['B'];cap=ledger['B']['payload_bytes']
    for p,a in B:check();codes.hear_context(p,a,cap)
    codes.select(data['listening']['C'])
    bank=Engine(65536,seed).bank
    assert abs(bank.active_fraction-.03535533905932738)<1e-15
    g=Geometry(65536,seed+17,bank.kc_side,bank.active_fraction)
    before=geometry_report(codes,g,data['listening']['D']);cal,unused=calibrate(codes,g,data['listening']['D'],check)
    after=geometry_report(codes,g,data['listening']['D']);resp=response(codes,g,data['listening']['D'],seed,check)
    # No native values ever existed during D0. BEGIN resets adaptation and the next
    # record is independent of discarded listening adaptation. Component parity
    # checks establish this zero-work clock shortcut; every interval is charged.
    result=dict(seed=seed,listening=ledger,native_clock_events=events,CPU_seconds=time.process_time()-t,
                codes=codes.diagnostics(),geometry_before=before,geometry_after=after,calibration=cal,response=resp,
                skipped_zero_value_native_events=events,source_lock=source_lock(),
                native=dict(N=65536,active_fraction=bank.active_fraction,side_counts={str(s):int(np.sum(bank.kc_side==s)) for s in (0,1)},side_active=bank.side_n),
                model_selection_views=codes.selection['validation_model_views'],charged_per_history=True)
    save(f'results/D0_{seed}.json',result)
    with (HERE/'results'/f'PRESTATE_{seed}.pkl').open('wb') as f:pickle.dump((codes,g,events),f,protocol=5)
    status('CYCLE2_D0_COMPLETE',seed=seed,M2_aliases=result['codes']['alias_count'],selected_classes=codes.selected)
    return result

def main():
    if (HERE/'results/EXPOSURES.jsonl').exists():raise RuntimeError('existing attempt; automatic replay refused')
    parent=parent_audit();save('PARENT_ASSET_AUDIT.json',parent);save('SOURCE_LOCK.json',source_lock())
    check();snapshot('cycle1');status('CYCLE1_COMPONENTS')
    c=components(check)
    assert c['CPU_seconds']<=60,'component CPU cap exceeded'
    save('cycles/cycle1/AUDIT.json',dict(accepted=True,**c));save('cycles/cycle1/REVISION.json',dict(
        changes=['float64 MSE distinctness tolerance locked at 1e-12','unique namespaces avoid prior module collisions','private forward query view shares read-only q/ring; parity checked'],
        scientific_changes=False,continue_bounded_harness=True))
    snapshot('cycle2');datasets={seed:build(seed) for seed in BIRTHS}
    # Prior actual source/manifest roster scan was empty, excluding design reservations.
    save('DATA_LOCK.json',dict(birth_collision_scan='rg over prior JSON/Python rosters: no matches outside design/pointers/new workspace',births=BIRTHS,
        global_partition='unordered pair hash bucket9 held; all D0/OLD/NEW exclude bucket9',datasets={str(s):jsonable(d) for s,d in datasets.items()}))
    d0results=[]
    for seed in BIRTHS:
        status('CYCLE2_REPRESENTATION',seed=seed);d0results.append(d0(datasets[seed]))
    save('cycles/cycle2/AUDIT.json',dict(accepted=True,scope='integrated learned representation and actual frames; behavioral efficacy not implied',
         warnings=[dict(seed=r['seed'],aliases=r['codes']['alias_count'],selected_C=r['codes']['selection']['selected']) for r in d0results],
         component_warnings_block_harness=False,representation_frozen=True))
    save('cycles/cycle2/REVISION.json',dict(changes=['freeze measured learned state and actual thresholds before native teaching'],rates_or_roster_changed=False))
    snapshot('cycle3');completed=[];profile_done=False;model=None;current=None
    try:
        for seed in BIRTHS:
            d=datasets[seed]
            for arm in ('R-W','A-W','R-N_OLD','R-SH_OLD'):
                check();current=dict(seed=seed,arm=arm)
                with (HERE/'results'/f'PRESTATE_{seed}.pkl').open('rb') as f:codes,g,events=pickle.load(f)
                model=Learner(codes,seed,gated=arm!='A-W',geometry=g,d0_events=events)
                assert np.max(abs(model.engine.bank.fast))==np.max(abs(model.engine.bank.slow))==np.max(abs(model.engine.q))==0
                status('CYCLE3_ARM',**current)
                probe(model,d['old'][:8],seed,arm,'BEFORE','OLD');probe(model,orientations(d['held'][:8]),seed,arm,'BEFORE','HELD')
                prefix_start=time.process_time()
                for i,j in enumerate(d['old_order'],1):
                    check();p,a=(d['shuffled'] if arm=='R-SH_OLD' else d['old'])[j]
                    r=model.train(p,a,write=arm!='R-N_OLD')
                    append('TRAINING.jsonl',dict(seed=seed,arm=arm,phase='OLD',index=i,prompt=p.hex(),observed_answer=a.hex(),**r))
                    if i==16 and not profile_done:
                        cpu=time.process_time()-prefix_start
                        # Prefix includes actual 16-record production only, no baseline probes.
                        estimate=cpu/16*5120
                        profile=dict(records=16,CPU_seconds=cpu,estimated_5120_record_production_CPU=estimate,
                                     remaining_total_CPU=900-resources()['CPU_seconds'],evaluation_D0_already_charged=True)
                        save('results/PREFIX_PROFILE.json',profile);profile_done=True
                        # Keep the declared plan intact. Infeasibility is a reported limit,
                        # never an automatic dose, roster or task reduction.
                        if estimate>900-resources()['CPU_seconds']:
                            probe(model,d['old'][:8],seed,arm,'OLD16_RESOURCE_BOUND','OLD')
                            probe(model,orientations(d['held'][:8]),seed,arm,'OLD16_RESOURCE_BOUND','HELD')
                            raise Limit('16-record measured production projection exceeds remaining fixed CPU allowance')
                    if i in (16,64,256,512):
                        full=i==512;label='OLD'+str(i)
                        probe(model,d['old'] if full else d['old'][:8],seed,arm,label,'OLD')
                        probe(model,orientations(d['held'] if full else d['held'][:8]),seed,arm,label,'HELD')
                        status('CYCLE3_OLD_CHECKPOINT',**current,records=i)
                model.rest(3600)
                probe(model,d['old'][:8],seed,arm,'OLD_REST','OLD');probe(model,orientations(d['held'][:8]),seed,arm,'OLD_REST','HELD')
                for i,j in enumerate(d['new_order'],1):
                    check();p,a=d['new'][j];r=model.train(p,a)
                    append('TRAINING.jsonl',dict(seed=seed,arm=arm,phase='NEW',index=i,prompt=p.hex(),observed_answer=a.hex(),**r))
                    if i in (64,128):
                        for cohort,recs in [('OLD',d['old'][:8]),('NEW',d['new']),('HELD',orientations(d['held'][:8]))]:probe(model,recs,seed,arm,'NEW'+str(i),cohort)
                model.rest(3600)
                for cohort,recs in [('OLD',d['old']),('NEW',d['new']),('HELD',orientations(d['held']))]:probe(model,recs,seed,arm,'FINAL',cohort)
                exposures=[r for batch in d['listening'].values() for r in batch]
                old=d['shuffled'] if arm=='R-SH_OLD' else d['old']
                exposures += [old[i] for i in d['old_order']]+[d['new'][i] for i in d['new_order']]
                stats=model.engine.stats();entry=dict(**current,records=model.external,stats=stats,
                    production_CPU_seconds=model.production_cpu,D0_CPU_charged=d0results[BIRTHS.index(seed)]['CPU_seconds'],
                    held_countermodels=countermodels(exposures,orientations(d['held'])))
                save(f'results/ARM_{seed}_{arm}.json',entry)
                with (HERE/'results'/f'FINAL_{seed}_{arm}.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
                completed.append(entry);del model;model=None
        save('cycles/cycle3/AUDIT.json',dict(accepted=True,complete=True,histories=completed))
        return dict(status='THREE_CYCLES_COMPLETE',completed_histories=completed,resources=resources(),formal_jobs=0)
    except Limit as ex:
        partial=None
        if model is not None:
            partial=dict(**current,records=model.external,stats=model.engine.stats(),production_CPU_seconds=model.production_cpu)
            with (HERE/'results/PARTIAL_STATE.pkl').open('wb') as f:pickle.dump(model,f,protocol=5)
        out=dict(status='RESOURCE_LIMIT',reason=str(ex),completed_histories=completed,partial_history=partial,resources=resources(),formal_jobs=0)
        save('cycles/cycle3/AUDIT.json',dict(accepted=True,complete=False,efficacy_claim=False,**out));return out

if __name__=='__main__':
    awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
    try:
        result=main();save('RUN_RESULT.json',result);status(result['status'],result=result)
    except BaseException as ex:
        result=dict(status='FAILED',error=repr(ex),traceback=traceback.format_exc(),resources=resources())
        save('FAILURE.json',result);status('FAILED',error=repr(ex));raise
    finally:
        awake.terminate();awake.wait()
