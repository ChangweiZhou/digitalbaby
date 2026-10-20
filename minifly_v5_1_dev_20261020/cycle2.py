"""Cycle 2: finite dose grid on target-independent prepared activity frames."""
from v51_common import *
from engine import Engine,DT,decode,residual,targets
import resource,pickle,gc

REPLIES=[b'',b'\0',b'\xff',b'\x80',b'\0\xff',b'\xff\0',b'\x7f\x80\xff',b'\x80\0\x01']

def frames(n):
    rng=np.random.default_rng(5100001);order=rng.permutation(n);active=int(n*.02)
    x=np.zeros((32,n))
    for k in range(32):x[k,order[k*active:(k+1)*active]]=1
    return x

def read_reply(e,x,cue,cap=4):
    query=e.clone();query.begin();out=[]
    for step in range(cap):
        m,_,_=query.margins(x[cue*4+step]);token=decode(m);query.tick(x[cue*4+step])
        if token==256:return bytes(out),True
        out.append(token)
    return bytes(out),False

def probe(e,x):
    rows=[]
    for cue,a in enumerate(REPLIES):
        p,ended=read_reply(e,x,cue)
        rows.append(dict(cue=cue,expected=a.hex(),payload=p.hex(),ended=ended,exact=bool(ended and p==a)))
    return dict(exact=sum(r['exact'] for r in rows)/8,rows=rows)

def run():
    start=time.process_time();save(Path('cycles/cycle2/SOURCE_BEFORE.json'),source_lock())
    assert json.loads((HERE/'cycles/cycle1/AUDIT.json').read_text())['accepted']
    settings=[];records=0;partial=0;selected=None;x=frames(65536)
    fixture=dict(replies=[a.hex() for a in REPLIES],frame_hash=hashlib.sha256(x.tobytes()).hexdigest(),
        frame_address='4*cue+reply_step, never target byte or expected length',active_per_frame=int(x[0].sum()),
        overlap_off_diagonal=0,scientific_records=0)
    save(Path('cycles/cycle2/FIXTURES.json'),fixture)
    for a_max in (1/16,1/4,1.):
        for W in (1024,256):
            births=[];setting_start=time.process_time()
            for seed in (202610205101,202610205102):
                e=Engine(65536,seed,True);n=Engine(65536,seed,True)
                snapshots=[];external=0;native_target_events=0
                for round_no in range(1,257):
                    for cue,a in enumerate(REPLIES):
                        e.begin();n.begin()
                        strength=a_max*(1/16+15/16*min(external/W,1))
                        loss=0.;base_loss=0.;terms=0
                        for step,token in enumerate((*a,256)):
                            xx=x[cue*4+step];m,_,_=e.margins(xx)
                            mask=np.zeros(9,dtype=bool) if token==256 else np.ones(9,dtype=bool)
                            if token==256:mask[8]=True
                            loss+=float(np.mean((np.tanh(m[mask]/2)-targets(token)[mask])**2));base_loss+=1.;terms+=1
                            e.tick(xx,residual(m,token,strength));n.tick(xx)
                            native_target_events+=1
                        # Real prompt identity, not a cue plus future target.
                        e.end(bytes([cue]));n.end(bytes([cue]),write=False)
                        external+=1;records+=2
                    if round_no in (1,4,16,64,256):
                        row=dict(round=round_no,external=external,strength=strength,
                            loss_on_last_record=loss/terms,matched_passive_loss=base_loss/terms,
                            W=probe(e,x),N=probe(n,x),native=e.stats())
                        snapshots.append(row)
                        save(Path('cycles/cycle2/PROGRESS.json'),dict(a_max=a_max,W=W,seed=seed,
                            round=round_no,completed_full_record_applications=records,CPU_seconds=time.process_time()-start,
                            own_exact=row['W']['exact'],no_write_exact=row['N']['exact']))
                        print(json.dumps(dict(a_max=a_max,W=W,seed=seed,round=round_no,
                            exact=row['W']['exact'],CPU_seconds=time.process_time()-start)),flush=True)
                    if time.process_time()-setting_start>1800:raise RuntimeError('technical setting CPU cap')
                immediate=snapshots[-1];before=pickle.dumps(e,protocol=5)
                with (HERE/'cycles/cycle2'/f'port_{a_max}_{W}_{seed}.pkl').open('wb') as f:f.write(before)
                e.rest(3600);rested=probe(e,x)
                qualified=(immediate['W']['exact']==1 and immediate['loss_on_last_record']<immediate['matched_passive_loss']
                    and np.isfinite(e.bank.fast).all() and np.isfinite(e.bank.slow).all())
                row=dict(seed=seed,external_training_records=external,matched_passive_records=external,
                    target_events_per_branch=native_target_events,qualified=bool(qualified),snapshots=snapshots,
                    one_hour_taught_exact=rested,native_after_rest=e.stats())
                births.append(row);save(Path('cycles/cycle2')/f'birth_{a_max}_{W}_{seed}.json',row)
                del e,n;gc.collect()
            setting=dict(a_max=a_max,W=W,births=births,qualified=all(b['qualified'] for b in births),CPU_seconds=time.process_time()-setting_start)
            settings.append(setting)
            if setting['qualified']:
                selected=dict(a_max=a_max,W=W);break
        if selected:break
    out=dict(accepted=True,Q0_passed=bool(selected),selected=selected,settings=settings,
        full_record_applications=records,external_information_records=records//2,
        matched_no_write_presentations=records//2,partial_native_events=partial,
        scope='Generic isolated port E1 only; no learned encoder, arithmetic or E3',scientific_records=0,
        CPU_seconds=time.process_time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_after=source_lock())
    save(Path('cycles/cycle2/AUDIT.json'),out)
    print(json.dumps({k:v for k,v in out.items() if k not in ['settings','source_after']}),flush=True)
    return out
if __name__=='__main__':run()
