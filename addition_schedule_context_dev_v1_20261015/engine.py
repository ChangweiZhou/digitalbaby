"""Record application and actual write evidence. No policy receives task metadata."""
import copy
import time
import models
import numpy as np


def versions(model):
    return dict(parameters={k:p._version for k,p in model.p.items()},moments={k:m._version for k,m in model.mom.items()},actor_steps=model.actor_steps)


def native_birth_evidence(model):
    stores=[s for bank in model.banks for s in bank]
    arrays=[a for s in stores for a in (s.fly.m.fast,s.fly.m.slow,s.fly.m.adapt)]
    independent=len({id(s.fly) for s in stores})==32 and all(not np.shares_memory(a,b) for i,a in enumerate(arrays) for b in arrays[i+1:])
    if not independent:raise AssertionError('native newborn mutable alias')
    return dict(births=[dict(bank=k,store=j,**s.birth) for k,bank in enumerate(model.banks) for j,s in enumerate(bank)],mutable_stores_independent=True)


def apply_addition(model,row,*,capture=False):
    before=versions(model);record=model.train(bytes.fromhex(row['prompt']),bytes.fromhex(row['answer']),row['allowed'],capture=capture)
    after=versions(model)
    if not row['allowed'] and before!=after:raise AssertionError('forbidden parameter or RMS write')
    return dict(id=row['id'],kind=row['kind'],source_index=row['source_index'],prompt=row['prompt'],answer=row['answer'],
                versions_before=before,versions_after=after,**record)


def apply_native(model,item,allowed):
    start=model.time;record=model.train(item['cue'],item['answer'],allowed)
    elapsed=model.time-start
    if elapsed>165:raise AssertionError('native frame exceeds fixed cadence')
    model.rest(165-elapsed)
    return dict(id=item['id'],cue=item['cue'].hex(),answer=item['answer'].hex(),**record)
