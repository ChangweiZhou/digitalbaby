from pathlib import Path
import sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration11/src'))
from v66_common import np,pd,SmallFly,FrozenReader,events,train,state_error,state,rest,fingerprint,balanced,analytic_baseline
from ports import PortFly
CFG=json.loads((ROOT/'config.json').read_text());DATA=ROOT/'data';OUT=ROOT/'results';MODEL=BASE/CFG['parent'];READER=BASE/CFG['observer']
ARMS=['INTACT','G0','A0','GA0','NO_WRITE','HALF_WRITE']
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def measure(m,A,B,role,reader):
    obs=[];score=[];memory=[];baseline=[];active=[]
    for p in (A,B):
        n=m.clone();r=SmallFly.step(n,5.,p);dx=r['sensory_activity'];o=r['rates'][6:8]
        pred=reader.predict(dx);zero=analytic_baseline(dx@n.Q,n.k0,n.MM)
        obs.append(o);score.append(float((o-pred).mean()));memory.append(float((o-zero).mean()));baseline.append(float(zero.mean()));active.append(r['active_KCs'])
    direction=2*role-1;actual=direction*(memory[0]-memory[1]);reader_margin=direction*(score[0]-score[1])
    return dict(actual=actual,reader=reader_margin,both=bool(actual>=1 and reader_margin>=reader.pair_threshold),
        actual_correct=bool(actual>=1),reader_correct=bool(reader_margin>=reader.pair_threshold),
        A_left=float(obs[0][0]),A_right=float(obs[0][1]),B_left=float(obs[1][0]),B_right=float(obs[1][1]),
        A_memory=memory[0],B_memory=memory[1],A_naive=baseline[0],B_naive=baseline[1],
        clipped=float(np.mean((np.asarray(obs)<=-11.2+1e-10)|(np.asarray(obs)>=19.96-1e-10))),active_KCs=float(np.mean(active)))
def trained(m,A,B,role,bouts=6):
    raw=np.zeros(3);p_sum=np.zeros((4,4));interaction=0.;bound_excess=0.
    bound=abs(m.fw0)*31.16*np.abs(m.F[2:]).sum(0)
    for dt,p,u in events(A,B,role,bouts):
        r=m.step(dt,p,u);raw+=np.abs(r['raw_increment']).sum(0)
        ps=r['all_payloads'];p_sum+=ps
        interaction=max(interaction,float(abs(ps[0]-ps[1]-ps[2]+ps[3]).max()))
        bound_excess=max(bound_excess,float(np.max(abs(ps[0]-ps[2])-bound)))
    return dict(raw_gamma_L1=raw[0],raw_alpha_fast_L1=raw[1],raw_slow_L1=raw[2],
        payload_interaction_max=interaction,alpha_bound_excess=max(0.,bound_excess),payloads_sum=p_sum.tolist())
def parent_sources():
    old=json.loads((BASE/'iteration11/checks/lock.json').read_text())['parent_assets']
    refs=list(old)+['iteration11/REPORT.md','iteration11/PROTOCOL.md','iteration11/config.json','iteration11/data/input_banks.npz',
        'iteration11/src/v66_common.py','v67_preparation/DESIGN.md','v67_preparation/protocol_draft.json','v67_preparation/STATIC_EVIDENCE.json']
    return {p:sha(BASE/p) for p in refs}
def code_hashes():return {n:sha(ROOT/'src'/n) for n in ('ports.py','v67_common.py','prepare.py','campaign.py')}
def ci(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    if not len(a):return dict(mean=None,low=None,high=None,n_streams=0)
    rng=np.random.default_rng(CFG['analysis_seed']);b=a[rng.integers(0,len(a),(2000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n_streams=len(a))
