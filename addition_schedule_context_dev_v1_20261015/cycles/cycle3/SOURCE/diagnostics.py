"""Offline read-only views of a single already-applied parameter proposal."""
import copy
import config
import models
import probes
import numpy as np


def views(capture,examples):
    if capture is None:raise ValueError('an actual captured update is required')
    pre=copy.deepcopy(capture['pre']);post=capture['post'];answer={}
    allkeys=set(pre['p'])
    sets=dict(NONE=set(),ALL=allkeys,E_ONLY=set(models.GROUPS['E']),R_ONLY=set(models.GROUPS['R']),
        C_ONLY=set(models.GROUPS['C']),V_ONLY=set(models.GROUPS['V']),
        INTERNAL_ONLY=allkeys-set(models.GROUPS['V']))
    for name in config.VIEWS:
        state=copy.deepcopy(pre)
        for k in sets[name]:state['p'][k]=post['p'][k].copy()  # exact applied post value, no rounded delta readdition
        model=models.Keep(0);model.set_state(state)
        for k in allkeys:
            np.testing.assert_array_equal(model.p[k].detach().numpy(),post['p'][k] if k in sets[name] else pre['p'][k])
            np.testing.assert_array_equal(model.mom[k].numpy(),pre['mom'][k])
        answer[name]=dict(own=probes.own(model,examples),conditional=probes.conditional(model,examples))
    reference=models.Keep(0);reference.set_state(post)
    actual=probes.own(reference,examples)
    if actual['rows']!=answer['ALL']['own']['rows']:raise AssertionError('ALL functional output differs from actual post')
    nll=lambda n:sum(r['mean_NLL'] for r in answer[n]['conditional']['rows'])/len(examples)
    return dict(views=answer,NLL_interaction=nll('ALL')-nll('V_ONLY')-nll('INTERNAL_ONLY')+nll('NONE'),
        training_applications=0,recomputed_gradients=0,interpretation='Read-only functional diagnostics; teacher prefixes are not autonomous outputs.')
