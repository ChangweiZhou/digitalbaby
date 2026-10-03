"""V66 instrumentation only; imports immutable parent dynamics and observer."""
from pathlib import Path
import sys,json,hashlib,csv,time
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration10/src'))
from common import np,pd,events,train,state_error,patterns,RetentionLearner,analytic_baseline
from smallfly import SmallFly,FrozenReader
from experiment import measure_pair
sys.path.insert(0,str(BASE/'v66_design_review'))
from feasibility_check import state,cosine,rest,sensitivity,linear_projection,probe_parts
CFG=json.loads((ROOT/'config.json').read_text())
MODEL=BASE/CFG['parent'];READER=BASE/CFG['observer'];DATA=ROOT/'data';OUT=ROOT/'results'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
    Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def scalar_cos(a,b):
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a.ravel(),b.ravel())/d) if d else 0.
def ratio(a,b):return 1. if a==b==0 else (float(a/b) if b else float('inf'))
def balanced(seed,n):
    rng=np.random.default_rng(seed+77)
    return np.concatenate([rng.permutation([0]*(min(12,n-k)//2)+[1]*(min(12,n-k)//2)) for k in range(0,n,12)])
def training(m,A,B,role,bouts=6):
    teachers=[];signals=[]
    for dt,p,u in events(A,B,role,bouts):
        r=m.step(dt,p,u)
        if p is not None:teachers.append(r['teaching_payload']);signals.append(r['sensory_activity'])
    t=np.asarray(teachers);x=np.asarray(signals)
    return dict(teacher=(2*role-1)*(t[::2].mean(0)-t[1::2].mean(0)),teacher_norm=float(np.sqrt(np.mean(t*t))),
                sensory=(2*role-1)*(x[::2].mean(0)-x[1::2].mean(0)),active=float(np.mean(np.count_nonzero(x,axis=1))))
def twins(m,A,B,role,bouts=6):
    w=m.clone();n=m.clone();n.mode='no_learning'
    sig=training(w,A,B,role,bouts);training(n,A,B,role,bouts)
    assert np.array_equal(w.adapt,n.adapt)
    assert (w.elapsed,w.event_count,w.presentation_count)==(n.elapsed,n.event_count,n.presentation_count)
    n.mode=m.mode
    return w,n,sig,state(w)-state(n)
def fingerprint(m):
    h=hashlib.sha256()
    for k in ('fast','slow','adapt','elapsed','event_count','presentation_count'):h.update(np.asarray(getattr(m,k)).tobytes())
    return h.hexdigest()
def checkpoint(template,b,roles):
    m=template.clone();signatures=[]
    for j in range(4):signatures.append(training(m,*b[2*j:2*j+2],int(roles[j])))
    footprints=[];reader=FrozenReader(m.Q,READER);focals=[]
    for j in range(4):
        omit=template.clone()
        for k in range(4):
            omit.mode='no_learning' if k==j else 'full'
            train(omit,*b[2*k:2*k+2],int(roles[k]))
        footprints.append(state(m)-state(omit))
        q=measure_pair(m,*b[2*j:2*j+2],int(roles[j]),reader)
        focals.append(dict(focal=j,eligible=bool(q['correct'] and q['oracle_correct']),**q))
    return m,footprints,signatures,focals
def metrics(w,n,A,B,role,reader,h):
    wh=rest(w,h);nh=rest(n,h)
    mw=measure_pair(wh,A,B,role,reader);mn=measure_pair(nh,A,B,role,reader)
    return dict(E_Hz=mw['oracle_margin_Hz']-mn['oracle_margin_Hz'],reader_E_Hz=mw['margin_Hz']-mn['margin_Hz'],
        W_actual=mw['oracle_margin_Hz'],N_actual=mn['oracle_margin_Hz'],W_reader=mw['margin_Hz'],N_reader=mn['margin_Hz'],
        W_both=bool(mw['correct'] and mw['oracle_correct']),N_both=bool(mn['correct'] and mn['oracle_correct']),
        clip_fraction=mw['clipped_fraction'],no_write_clip=mn['clipped_fraction'])
def sources():
    paths=[MODEL,READER,BASE/'iteration10/data/input_banks.npz',BASE/'iteration10/data/screen/reference.csv',
           BASE/'iteration10/src/common.py',BASE/'iteration10/src/experiment.py',BASE/'iteration10/src/smallfly.py',
           BASE/'iteration5/src/hybrid.py',BASE/'iteration5/src/prepare_interface.py',BASE/'iteration6/src/retention_model.py',
           BASE/'iteration8/src/readout.py',BASE/'iteration4/results/fitted_kernels.json',
           BASE/'v66_design_review/feasibility_check.py',BASE/'v66_design_review/feasibility_summary.json',
           BASE/'physiology_source_audit_20260913/REPORT.md']
    return {str(p.relative_to(BASE)):sha(p) for p in paths}
def code_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src').glob('*.py')) if p.name in ('v66_common.py','prepare.py','campaign.py')}
