"""DEV only. Equal raw streams/clock, four histories, matched read-only probes."""
import argparse, gzip, hashlib, itertools, json, time
from pathlib import Path
import numpy as np
import environment, config
from candidate import LocalContentCore

ROOT=Path(__file__).resolve().parent

def inputs(world):
    config.require_dev(world)
    rng=np.random.default_rng(world)
    keys=[b'3'+bytes(x) for x in itertools.product(b'012',repeat=3)]
    keys=[keys[i] for i in rng.permutation(27)[:24]]
    return dict(old=dict(zip(keys[:16],rng.permutation(np.tile(list(config.ALPHABET),4)).tolist())),
        new=dict(zip(keys[16:],rng.permutation(np.tile(list(config.ALPHABET),2)).tolist())))
def fixture(world,repeats=config.REPEATS):
    mp=inputs(world)
    mp['revised']={k:config.ALPHABET[(config.ALPHABET.index(v)+1)%4] for k,v in list(mp['old'].items())[:8]}
    raw={s:environment.native_streams.stream(mp[s],repeats,world+i) for i,s in enumerate(('old','new','revised'))}
    for s in raw:environment.native_streams.assert_contexts(raw[s],mp[s])
    return mp,raw

def allowed(branch,stage):
    return branch=='W' or (branch=='N_OLD' and stage!='old') or (branch=='N_REV' and stage!='revised')
def consume(model,raw):
    rows=[]
    for byte in raw:
        t=model.t+config.DT;model.predict(t);rows.append(model.observe(int(byte),t))
    return rows

def metrics(rows,mapping):
    r=[x for x in rows if bytes.fromhex(x['context']) in mapping]
    if not r:raise AssertionError('no focal rows')
    if any(x['byte']!=mapping[bytes.fromhex(x['context'])] for x in r):raise AssertionError('wrong probe target')
    return dict(focal_accuracy=sum(x['emitted']==x['byte'] for x in r)/len(r),
        focal_bpb=sum(x['loss_bits'] for x in r)/len(r),all_byte_accuracy=sum(x['emitted']==x['byte'] for x in rows)/len(rows),
        all_byte_bpb=sum(x['loss_bits'] for x in rows)/len(rows),focal_count=len(r),all_byte_count=len(rows))

def probe(model,raw,mapping,erase=False):
    before=model.state_digest();c=model.clone();c.plastic=False
    if erase:c.erase_native_content()
    cpu=time.process_time();rows=consume(c,raw);elapsed=time.process_time()-cpu
    assert model.state_digest()==before,'probe changed operative content'
    return dict(raw_hex=raw.hex(),initial_history=model.history.hex(),initial_time=model.t,rows=rows,metrics=metrics(rows,mapping),erase=erase,cpu_seconds=elapsed)

def run(world,quick=False):
    config.require_dev(world)
    mp,raw=fixture(world,2 if quick else 8)
    mp['current_old']={**mp['old'],**mp['revised']}
    mp['unchanged']={k:v for k,v in mp['old'].items() if k not in mp['revised']}
    cores={'A':{b:environment.native_core.ObservationCore() for b in config.BRANCHES},
           'B':{b:LocalContentCore() for b in config.BRANCHES}}
    stages=('old',) if quick else ('old','day1','new','day2','revised','day3')
    out=dict(world=world,evidence='DEV_ONLY_E0_E1',candidate_version=config.VERSION,quick=quick,inputs={s:{k.hex():v for k,v in d.items()} for s,d in mp.items()},
        raw={s:r.hex() for s,r in raw.items()},arms={},actual_training_lives=8)
    cpu=time.process_time();wall=time.monotonic()
    for arm,models in cores.items():
        result=dict(training={},probes={},states={},costs={},birth={})
        for b,m in models.items():
            result['birth'][b]=[s.birth_record for s in m.stores] if arm=='A' else [m.encoder.birth_record]
        for i,stage in enumerate(stages):
            print(f'DEV {world} {arm} {stage}',flush=True)
            if stage.startswith('day'):
                for m in models.values():m.rest(config.DAY)
            else:
                result['training'][stage]={}
                for branch,m in models.items():
                    m.plastic=allowed(branch,stage)
                    start=time.process_time(); rows=consume(m,raw[stage]);dt=time.process_time()-start
                    result['training'][stage][branch]=rows
                    result['costs'][f'{stage}/{branch}']=dict(cpu_seconds=dt,bytes=len(raw[stage]))
            targets=mp['old'] if stage in ('old','day1','new','day2') else mp['current_old']
            # All branches/configurations receive exactly the same full-old probe bytes.
            probe_raw=environment.native_streams.stream(targets,2,world+100+i)
            result['probes'][stage]={}
            for branch,m in models.items():
                pr=probe(m,probe_raw,targets)
                if stage in ('revised','day3'):
                    pr['unchanged_metrics']=metrics(pr['rows'],mp['unchanged'])
                    pr['revised_metrics']=metrics(pr['rows'],mp['revised'])
                if stage in ('new','day2','revised','day3'):
                    newraw=environment.native_streams.stream(mp['new'],2,world+200+i)
                    pr['new_probe']=probe(m,newraw,mp['new'])
                result['probes'][stage][branch]=pr
                result['states'][f'{stage}/{branch}']=dict(digest=m.state_digest(),clock=m.t,bytes_seen=m.bytes_seen,mutable_bytes=m.mutable_bytes())
            result['probes'][stage]['ERASE']=probe(models['W'],probe_raw,targets,True)
        out['arms'][arm]=result
    out['resources']=dict(cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
        training_bytes_per_history=sum(len(raw[s]) for s in result['training']),science_worlds=0)
    return out

def describe(r):
    out={}
    for a,x in r['arms'].items():
        p=x['probes'];stages=list(p);last=stages[-1]
        out[a]=dict(old=p['old']['W']['metrics']['focal_accuracy'],
            final_old=p[last]['W']['metrics']['focal_accuracy'],
            all_train_bpb=sum(z['loss_bits'] for s in x['training'].values() for z in s['W'])/sum(len(s['W']) for s in x['training'].values()),
            W_training_cpu=sum(c['cpu_seconds'] for key,c in x['costs'].items() if key.endswith('/W')),
            mutable_bytes=x['states'][f'{last}/W']['mutable_bytes'])
        if last=='day3':
            out[a].update(day2=p['day2']['W']['metrics']['focal_accuracy'],
                revised=p['day3']['W']['revised_metrics']['focal_accuracy'],
                unchanged=p['day3']['W']['unchanged_metrics']['focal_accuracy'],
                new=p['day3']['W']['new_probe']['metrics']['focal_accuracy'],
                revision_harm=p['day3']['N_REV']['unchanged_metrics']['focal_accuracy']-p['day3']['W']['unchanged_metrics']['focal_accuracy'],
                gain=p['day2']['W']['metrics']['focal_accuracy']-p['day2']['N_OLD']['metrics']['focal_accuracy'])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--world',type=int,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--quick',action='store_true')
    args=ap.parse_args();config.require_dev(args.world)
    if args.out.exists():raise ValueError('receipt already exists; no overwrite')
    r=run(args.world,args.quick);args.out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.out,'wt') as f:json.dump(r,f,sort_keys=True,allow_nan=False)
    print(json.dumps(describe(r),indent=2))
if __name__=='__main__':main()
