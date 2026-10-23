"""Shared frozen MiniFly model; this extension owns all search state and config."""
from pathlib import Path
import sys, os
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
sys.path.insert(0,str(BASE/'iteration24/src'))
from common79 import (np,pd,json,time,copy,sha,atomic_json,Fly,FrozenReader,READER,
    observed_activity,state_arrays,state_error,pair_events,step_encoded,ParentLearner,measured)
from model79 import raw_event
sys.path.insert(0,str(ROOT/'src'))
CFG=json.loads((ROOT/'config.json').read_text())


def bank(seed,panel):
    seeds=CFG['development_seeds'] if panel=='development' else CFG['confirmation_seeds'] if panel=='confirmation' else []
    if seed not in seeds:raise ValueError('Seed outside declared panel')
    with np.load(ROOT/'data/inputs.npz',allow_pickle=False) as z:
        pn=z[f'F_{seed}'].copy(); roles=z[f'roles_{seed}'].copy()
    return np.asarray(pn),roles


def encode_bank(model,pn):
    """Encode raw PN patterns through THIS candidate's own interface.

    V81E relaxes PN->KC weights, topology and KC sparsity, so the encoded bank is
    candidate-specific and can no longer be precomputed from a reference model.
    """
    from model_evo import encode_sparse
    return np.asarray([encode_sparse(model,p) for p in np.asarray(pn)])
