from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration21/src'))
from common76 import np,pd,json,time,copy,sha,atomic_json,Fly,FrozenReader,READER,observed_activity,state_arrays,state_error,sign_p,holm
from model76 import PairLearner as ParentLearner,geometry,pair_events,step_encoded
sys.path.insert(0,str(ROOT/'src'))
CFG=json.loads((ROOT/'config.json').read_text())
DEN=float(json.loads((BASE/'iteration21/data/calibration.json').read_text())['denominator'])
def bank(seed):
    with np.load(ROOT/'data/inputs.npz') as z:return z[f'F_{seed}'].copy(),z[f'roles_{seed}'].copy()
def confidence(x):
    x=np.asarray(x,float);assert len(x) and np.isfinite(x).all()
    r=np.random.default_rng(CFG['bootstrap_seed']);b=x[r.integers(len(x),size=(CFG['bootstrap_draws'],len(x)))].mean(1)
    return dict(mean=float(x.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),n=len(x))
