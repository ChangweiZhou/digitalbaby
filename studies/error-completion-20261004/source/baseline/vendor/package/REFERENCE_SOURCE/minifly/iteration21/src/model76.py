"""Pair teaching bridge. External pair residual is explicitly NOT a DAN model.

Gamma follows native events. Alpha forcing is replaced, while its decay and
allocation are retained. A pending alpha vector is committed after the pair.
"""
from common76 import *
from model73 import advance73

def step_encoded(m,dt,x=None,pun=0.,alpha_scale=1.,plastic=True):
    """Same frozen EVENT map, bypassing only deterministic PN re-encoding."""
    assert np.isfinite(dt) and dt>=0
    k=m.kernel
    if x is None or not np.any(x):
        m.adapt[:]=1-(1-m.adapt)*np.exp(-dt/m.ta)
        m.fast[:,0]*=np.exp(-dt/m.tg);m.fast[:,1]*=np.exp(-dt/k['fast_tau'])
        m.slow*=np.exp(-dt/k['slow_tau']);active=False;gamma_L1=0.
    else:
        r=advance73(x,float(dt),float(pun),m.Q,m.T,m.F,m.MM,m.k0,m.fw0,m.fwd,m.ta,m.tg,k['fast_tau'],k['slow_tau'],k['fraction'],
          m.fast,m.slow,m.adapt,0,0,m.strength,m.Q4,m.weights,m.side,m.ei,m.ej,m.eq,m.share,m.ET,m.R,m.R2,plastic,float(alpha_scale))
        gamma_L1=float(abs(r[3][:,0]).sum());active=True
    m.elapsed=np.float64(m.elapsed+dt);m.event_count+=np.uint64(1);m.presentation_count+=np.uint64(active)
    return gamma_L1

def pair_events(m,codes,label,alpha_scale=1.,plastic=True):
    total=0.
    for i in range(2):
        total+=step_encoded(m,30.,codes[i],float(i==label),alpha_scale,plastic)
        step_encoded(m,135.)
    return total

def geometry(m,dx):
    diff=dx[0]-dx[1];v=diff*m.T[:,2:].mean(1)
    g=diff*m.Q[:,2:].mean(1)*np.any(m.T[:,2:]!=0,axis=1)
    den=float(g@v);assert den>=-1e-14
    return v,max(0.,den)

def calibrate():
    """Only unlabeled PN activity determines the one fixed geometric gain."""
    with np.load(ROOT/'data/inputs.npz') as z:pn=z['CALIB'].copy()
    m=Fly();codes=np.array([m.encode(p) for p in pn]);rng=np.random.default_rng(CFG['calibration_seed']+9)
    rows=[]
    for t in range(CFG['calibration_trials']):
        idx=int(rng.integers(len(codes)//2));cc=codes[2*idx:2*idx+2]
        _,den=geometry(m,observed_activity(m,cc));rows.append(dict(trial=t,denominator=den))
        pair_events(m,cc,0,plastic=False)
    ar=np.array([r['denominator'] for r in rows[CFG['warmup']:]])
    assert np.all(ar>CFG['epsilon']) and np.isfinite(ar).all()
    value=float(np.median(ar));out=dict(denominator=value,source='unlabeled CALIB geometry; no output/label fit',
        n=len(ar),quantiles={str(q):float(np.quantile(ar,q)) for q in [0,.01,.1,.5,.9,.99,1]},
        native_fraction=float(m.kernel['fraction']),target_rule='primary R0=2 * frozen 1Hz threshold; sensitivities1/4Hz',
        target_primary_Hz=2.,input_sha=sha(ROOT/'data/inputs.npz'))
    pd.DataFrame(rows).to_csv(ROOT/'data/calibration_geometry.csv',index=False)
    atomic_json(ROOT/'data/calibration.json',out);return out

class PairLearner:
    def __init__(self,arm,denominator=None,prestate=None):
        self.arm=arm;self.spec=CFG['arms'][arm].copy();self.m=Fly() if prestate is None else prestate.clone()
        self.reader=FrozenReader(self.m.Q,READER)
        self.den0=float(denominator if denominator is not None else json.loads((ROOT/'data/calibration.json').read_text())['denominator'])
        self.legacy=None
        if self.spec['kind']=='legacy':
            assert prestate is None,'V75 bridge has no common native-prestate interpretation'
            self.legacy=Bridge75(self.m,'ADAPTED','CLIPPED','ROUTE_ERROR')
        self.reset_stats()

    def reset_stats(self):
        self.stats=dict(updates=0,zero_geometry=0,denominator_clamps=0,bridge_alpha_L1=0.,bridge_slow_write_L1=0.,bridge_fast_write_L1=0.,gamma_write_L1=0.,abs_signal=0.)
        self.denominators=[];self.gains=[]

    def clone(self):
        n=copy.copy(self);n.m=self.m.clone();n.spec=self.spec.copy();n.reset_stats()
        if self.legacy is not None:
            n.legacy=copy.copy(self.legacy);n.legacy.s=self.legacy.s.copy()
        return n

    def read(self,codes,h=0,erase_alpha_fast=False):
        m=self.m
        if erase_alpha_fast:
            m=m.clone();m.fast[:,1]=0.
        return unsigned_observe(m,codes,self.reader,h)

    def pending(self,codes,label):
        """Only present activity, own observed output, revealed label and constants."""
        assert label in (0,1)
        dx=observed_activity(self.m,codes)
        observed=self.m.expression(dx)-self.reader.predict(dx)
        score=float(observed.mean(1)@[1.,-1.])
        v,den=geometry(self.m,dx);s=self.spec
        target=(2*int(label)-1)*s['target_Hz']
        signal=target-score if s['signal']=='error' else target
        used=self.den0;clamped=False
        if s['normalization']=='adaptive':
            used=float(np.clip(den,self.den0*CFG['normalizer_range'][0],self.den0*CFG['normalizer_range'][1]));clamped=used!=den
        gain=CFG['eta']/(CFG['epsilon']+used)
        u=gain*signal*v if den>CFG['epsilon'] else np.zeros_like(v)
        assert np.isfinite(u).all()
        return u,dict(score=score,signal=float(signal),denominator=den,gain=gain,clamped=int(clamped))

    def learn_pair(self,codes,label,write=True):
        """One physical 330s pair; no cue identity, task identity or clean label."""
        assert codes.shape==(2,len(self.m.slow)) and label in (0,1)
        kind=self.spec['kind'];u=None;info=None
        if write and kind=='hybrid':u,info=self.pending(codes,label)
        if kind=='legacy':
            if write:
                self.legacy.predict(observed_activity(self.m,codes),self.reader)
                self.legacy.learn(label) # V75 commits then applies one330s slow decay.
            else:self.legacy.s*=np.exp(-330/self.m.kernel['slow_tau'])
            pair_events(self.m,codes,label,plastic=False)
            self.m.slow=self.legacy.s.copy()
        else:
            gamma=pair_events(self.m,codes,label,alpha_scale=1. if kind=='native' else 0.,plastic=write)
            self.stats['gamma_write_L1']+=gamma
            if u is not None:
                fraction=self.m.kernel['fraction'] if self.spec['allocation']=='native' else 1.
                self.m.fast[:,1]+=(1-fraction)*u;self.m.slow+=fraction*u
                mass=float(abs(u).sum());self.stats['bridge_alpha_L1']+=mass
                self.stats['bridge_slow_write_L1']+=fraction*mass;self.stats['bridge_fast_write_L1']+=(1-fraction)*mass
                self.stats['zero_geometry']+=int(info['denominator']<=CFG['epsilon'])
                self.stats['denominator_clamps']+=info['clamped'];self.stats['abs_signal']+=abs(info['signal'])
                self.denominators.append(info['denominator']);self.gains.append(info['gain'])
        self.stats['updates']+=int(write)
        assert np.isfinite(self.m.fast).all() and np.isfinite(self.m.slow).all() and np.isfinite(self.m.adapt).all()
        return info

    def rest(self,dt):
        step_encoded(self.m,dt)
        if self.legacy is not None:self.legacy.s=self.m.slow.copy()

    def diagnostic(self):
        out=self.stats.copy();n=max(1,out['updates'])
        out.update(clamp_fraction=out['denominator_clamps']/n,zero_geometry_fraction=out['zero_geometry']/n,
            max_abs_slow=float(abs(self.m.slow).max()),max_abs_alpha_fast=float(abs(self.m.fast[:,1]).max()),
            state_L2=float(np.linalg.norm(state_arrays(self.m))),mutable_bytes=self.m.mutable_bytes()+(self.legacy.s.nbytes if self.legacy is not None else 0),
            legacy_extra_state_bytes=self.legacy.s.nbytes if self.legacy is not None else 0,
            pending_vector_bytes=self.m.slow.nbytes if self.spec['kind']=='hybrid' else 0)
        for name,vals in [('denominator',self.denominators),('gain',self.gains)]:
            if vals:
                for q in [0,.5,1]:out[f'{name}_q{q:g}']=float(np.quantile(vals,q))
        return out
