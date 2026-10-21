"""Cheap structural/native and real-frame audits, independent of addition scores."""
from r2_common import *
from r2_engine import Engine,DT,targets,decode,residual
from r2_representation import Codes,Competitive,BITS
from r2_geometry import Sequence,Geometry
from r2_core import Frames,query_view
import ast,pickle

def components(check):
    start=time.process_time();out={};events=0
    # Exact observed initialization, float64 distinctness and inactive capacities.
    m=Competitive(64,8,77,256)
    for b in range(256):m.retain(BITS[b])
    m.initialize(1024);assert len(m.w)==64 and len(np.unique(m.w,axis=0))==64
    z=Competitive(6,8,77,64)
    for _ in range(64):z.retain(BITS[0])
    z.initialize(4096);assert len(z.w)==1
    assert m.gate(0,1024)==0 and m.gate(0,2048)==1
    out['observed_distinct_initialization_no_duplicate_competitors']=True
    maximum=0.;referror=0.
    for n in (5177,65536):
        e=Engine(n,202610205399,False);ref=e.bank.clone();rng=np.random.default_rng(99)
        assert abs(e.bank.active_fraction-0.03535533905932738)<1e-15
        for step in range(2):
            check();x=np.zeros(n)
            for side in (0,1):
                ix=np.flatnonzero(e.bank.kc_side==side);x[rng.choice(ix,e.bank.side_n[side],replace=False)]=1
            coeff=np.linspace(-.01,.01,9) if step else np.zeros(9)
            m0,_,_=e.margins(x);neutral=ref.clone();neutral.fast.fill(0);neutral.slow.fill(0)
            actual,_=query_batch(ref.fast,ref.slow,ref.adapt,x,DT,ref.Q,ref.reader_Q,ref.MM,ref.k0,ref.ta,ref.tg,ref.tf,ref.ts)
            baseline,_=query_batch(neutral.fast,neutral.slow,neutral.adapt,x,DT,ref.Q,ref.reader_Q,ref.MM,ref.k0,ref.ta,ref.tg,ref.tf,ref.ts)
            err=float(abs(m0-(actual-baseline).mean(1)).max());referror=max(referror,err);assert err<1e-12
            e.tick(x,coeff);events+=1
            event_batch(ref.fast,ref.slow,ref.adapt,x,DT,coeff,True,ref.Q,ref.T,ref.F,ref.MM,ref.k0,ref.fw0,ref.fwd,ref.ta,ref.tg,ref.tf,ref.ts,ref.fraction,ref.alpha_scale,ref.genes,ref.sf,ref.ss)
            for k in ('fast','slow','adapt'):
                err=float(abs(getattr(e.bank,k)-getattr(ref,k)).max());maximum=max(maximum,err);assert err<1e-9
        # Private-forward view must match full clone exactly, including all values.
        a=e.clone();b=query_view(e);before=pickle.dumps(e,protocol=5)
        a.begin();b.begin();a.tick(x);b.tick(x);events+=2
        assert np.array_equal(a.margins(x)[0],b.margins(x)[0]);assert before==pickle.dumps(e,protocol=5)
    out.update(native_parity=True,max_state_error=maximum,max_reference_error=referror,readonly_view_full_clone_parity=True)
    # During D0 values and q are identically zero. Skipping forward adaptation is
    # equivalent for the next record only because BEGIN resets it to one.
    a=Engine(5177,202610205399);b=Engine(5177,202610205399)
    x=np.zeros(5177);x[:100]=1
    for _ in range(3):a.tick(x);events+=1
    b.time=3*DT;b.events=3;a.begin();b.begin()
    assert np.array_equal(a.bank.adapt,b.bank.adapt)
    assert np.array_equal(a.margins(x)[0],b.margins(x)[0])
    a.tick(x);b.tick(x);events+=2
    for name in ('fast','slow','adapt'):assert np.array_equal(getattr(a.bank,name),getattr(b.bank,name))
    out['zero_value_D0_clock_skip_next_BEGIN_equivalence']=True
    g=Engine(64,202610205399);x=np.ones(64);slow=[]
    for i in range(5):
        g.begin();g.tick(x,np.ones(9)*.1);events+=1;slow.append(g.slow_write_l1);g.end(bytes([i]));g.rest(61)
    assert slow[:4]==[0.]*4 and slow[4]>0
    g.signs.fill(2);before=g.q.copy();g.end(b'mixed');assert np.array_equal(before,g.q)
    out['strict_confirmation_floor_and_mixed_suppression']=True
    assert all(decode(targets(i))==i for i in range(257));assert np.all(residual(np.zeros(9),256,.01)[:8]==0)
    codes=Codes(77)
    for _ in range(16):codes.hear(bytes(range(256)),4096)
    codes.identity_frozen=True
    for _ in range(20):codes.hear_context(bytes(range(128)),bytes(range(128,256)),5120)
    profiles,valid=codes.profiles();assert np.allclose(profiles[valid].sum(1),1)
    codes.select([(bytes(range(128)),bytes(range(128,256)))])
    f=Frames(codes);states=list(f.prompt(b'\x00\xff'))
    assert f.state.reply==0 and f.state.open[-1]==255
    before=f.state.feature().copy();x=f.prediction().feature().copy();assert np.array_equal(before,x)
    f.observed(128);assert f.state.reply==1 and f.state.open[-1]==128
    f.observed(256);assert f.state.reply==1
    out.update(all_byte_EOS_roundtrip=True,normalized_context=True,reply_prefeedback_and_EOS=True,
               synthetic_payload_exposures=4096+5120,synthetic_validation_payload=256,
               synthetic_profile_views=codes.profile_views,synthetic_model_selection=codes.selection)
    for name in ('r2_core.py','r2_engine.py','r2_geometry.py','r2_representation.py'):
        tree=ast.parse((HERE/name).read_text());imports=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imports.extend(n.name for n in node.names)
            if isinstance(node,ast.ImportFrom):imports.append(node.module)
        assert not set(imports)&{'r2_data','data','torch','jax','tensorflow','sklearn'}
    out.update(no_task_or_gradient_imports=True,partial_native_events=events,full_native_records=0,CPU_seconds=time.process_time()-start)
    return out

def geometry_report(codes,g,records):
    xs=[];ties=[];cuts=[];closures=[];family=[]
    for p,a in records[:16]:
        f=Frames(codes)
        for _ in f.prompt(p):pass
        for token in (*a,256):
            x,raw=g.encode(f.prediction());xs.append(x)
            # Contributions from generic identity / raw / presence and clocks.
            feature=f.state.feature();con=feature[g.index]*g.weights
            family.append([float(abs(con[:,j]).sum()) for j in range(4)])
            for side in (0,1):
                ix=np.flatnonzero((g.side==side)&(x>0));cut=float(raw[ix].min()) if len(ix) else 0
                cuts.append(cut);ties.append(dict(selected=int(np.sum(raw[ix]==cut)),available=int(np.sum((g.side==side)&(raw==cut)))))
            f.observed(token)
        closures.append(f.state.closures)
    a=np.array(xs);n=len(a);counts=a.sum(1);pairs=[]
    for i in range(n-1):pairs.append(float(np.dot(a[i],a[i+1])/max(1,min(counts[i],counts[i+1]))))
    f=Frames(codes)
    for _ in f.prompt(records[0][0]):pass
    assert np.array_equal(g.encode(f.prediction())[0],g.encode(f.prediction())[0])
    return dict(frames=n,active_min=float(counts.min()),active_median=float(np.median(counts)),active_max=float(counts.max()),
                adjacent_overlap_mean=float(np.mean(pairs)),common_all=int(np.sum(a.min(0)>0)),
                cutoff_min=float(min(cuts)),cutoff_max=float(max(cuts)),cutoff_ties=ties,closures=closures,
                fan_contribution_L1_mean=np.mean(family,axis=0).tolist(),stable_same_frame=True)

def response(codes,geometry,records,seed,check):
    frames=[]
    for p,a in records[:16]:
        f=Frames(codes)
        for _ in f.prompt(p):pass
        for token in (*a,256):
            x,_=geometry.encode(f.prediction());frames.append(x);f.observed(token)
    ix=np.argsort([x.sum() for x in frames],kind='stable');chosen=[ix[0],ix[len(ix)//2],ix[-1]];rows=[];events=0
    base=Engine(65536,seed)
    for quantile,i in zip(('low','median','high'),chosen):
        x=frames[i]
        for sign in (-1,0,1):
            check();e=query_view(base);e.begin();m=e.margins(x)[0];y=np.ones(9)*sign
            coeff=(1/256)*(np.tanh(m/2)-y)/2 if sign else None
            e.tick(x,coeff);events+=1;after=e.margins(x)[0]
            rows.append(dict(quantile=quantile,active=int(x.sum()),sign=sign,strength=1/256,margin=after.tolist(),
                             target_direction_mean=float(np.mean(after*sign)),passive=sign==0))
    return dict(rows=rows,partial_native_events=events,full_native_records=0,
                scope='one-event actual-frame directional response, not acquisition or safe ceiling qualification')
