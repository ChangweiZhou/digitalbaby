"""Whole-record checkpoints with hashes; no training or arithmetic logic."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from core import Learner, Limits
from artifacts import sha, atomic_json


def save(model, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:getattr(model.native,k) for k in ('fast','slow','adapt')}
    arrays.update(theta=model.theta,freq=model.freq,h=model.h,fe=model.fe.p)
    for group in ('params','eligibility','pending'):
        arrays.update({group+'__'+k:v for k,v in getattr(model,group).items()})
    meta=dict(seed=model.seed,plastic=model.plastic,gradual=model.gradual,source_parent=model.source_parent,
              limits=asdict(model.limits),time=model.time,fe_time=model.fe.t,records=model.records,
              parameter_steps=model.parameter_steps,version=model.version,stats=model.stats,
              native_elapsed=model.native.elapsed,native_events=model.native.events,digest=model.digest())
    arrays['metadata']=np.array(json.dumps(meta,sort_keys=True))
    np.savez_compressed(path,**arrays)
    atomic_json(path.with_suffix('.certificate.json'),dict(sha256=sha(path),digest=meta['digest'],records=model.records,
                                                          time=model.time,whole_record_boundary=True))
    return meta['digest']


def load(path):
    path=Path(path);certificate=json.loads(path.with_suffix('.certificate.json').read_text())
    if sha(path)!=certificate['sha256']:raise ValueError('checkpoint hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']))
        m=Learner(meta['seed'],plastic=meta['plastic'],gradual=meta['gradual'],
                  source_parent=meta['source_parent'],limits=Limits(**meta['limits']))
        for k in ('fast','slow','adapt'):setattr(m.native,k,z[k].copy())
        for k in ('theta','freq','h'):setattr(m,k,z[k].copy())
        m.fe.p=z['fe'].copy();m.fe.t=meta['fe_time']
        for group in ('params','eligibility','pending'):
            setattr(m,group,{k:z[group+'__'+k].copy() for k in getattr(m,group)})
        for k in ('time','records','parameter_steps','version','stats'):setattr(m,k,meta[k])
        m.native.elapsed=meta['native_elapsed'];m.native.events=meta['native_events']
        assert m.digest()==meta['digest']==certificate['digest'],'restored checkpoint differs'
        return m
