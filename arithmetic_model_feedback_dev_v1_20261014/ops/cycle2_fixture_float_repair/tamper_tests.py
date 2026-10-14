"""Adversarial receipt copies only; never modifies accepted jobs or states."""
import copy,gzip,json
from pathlib import Path
import numpy as np
from protocol import ROOT,identity,atomic
from auditor import audit
from analysis import staircase,metrics,reduce_stage
def run():
    folder=ROOT/'cycles/cycle2';p=next((folder/'receipts').glob('*_R0.json.gz'))
    with gzip.open(p,'rt') as f:r=json.load(f)
    checks=[]
    def reject(name,mutate):
        t=copy.deepcopy(r);mutate(t)
        try:audit(t);raise RuntimeError('auditor admitted tamper '+name)
        except AssertionError:checks.append(dict(name=name,rejected=True))
    reject('reward flip',lambda t:t['prefixes']['TRUE']['segment']['records'][0].update(reward=1-t['prefixes']['TRUE']['segment']['records'][0]['reward']))
    reject('hidden scalar answer',lambda t:t['branches']['FADING']['segments'][-1]['records'][0].update(observed=t['branches']['FADING']['segments'][-1]['records'][0]['expected']))
    reject('scalar learns with dense mode',lambda t:t['branches']['FADING']['segments'][-1]['records'][0].update(learning_mode='DENSE'))
    reject('no warm actor write',lambda t:t['prefixes']['NO_WARM']['segment']['records'][0]['update'].update(actor_steps=1))
    reject('frozen critic write',lambda t:t['branches']['FREEZE_SCALAR']['segments'][0]['records'][0]['update'].update(critic_steps=1))
    reject('changed operation',lambda t:t['prefixes']['TRUE']['segment']['records'][0].update(operation='*'))
    reject('false expected answer',lambda t:t['prefixes']['TRUE']['segment']['records'][0].update(expected='ffff'))
    reject('duplicate held representative',lambda t:t['branches']['FADING']['curve'][-1]['scores']['binary'].__setitem__(0,t['branches']['FADING']['curve'][-1]['scores']['binary'][1]))
    reject('changed dataset',lambda t:t.update(data_digest='bad'))
    reject('wrong total count',lambda t:t.update(executed_records=t['executed_records']+1))
    reject('wrong scalar fork',lambda t:t['branches']['FREEZE_SCALAR'].update(fork_state='bad'))
    reject('bad output shortcut',lambda t:t['branches']['FADING']['curve'][-1]['scores']['mixed'][0]['shortcuts'].update(global_='bad'))
    with np.load(ROOT/r['archives']['BIRTH.npz'],allow_pickle=False) as old:data={k:old[k].copy() for k in old.files}
    data['p_E'][0,0]+=1;fake=ROOT/'ops/TAMPER_ARCHIVE.npz'
    with fake.open('wb') as f:np.savez_compressed(f,**data)
    reject('archive array tamper',lambda t:t['archives'].update({'BIRTH.npz':str(fake.relative_to(ROOT))}))
    fake.unlink()
    assert staircase([(0,.2),(5,.6),(10,.9)],10)==.4
    assert staircase([(3,.2),(5,.6)],10)==.34
    checks.extend([dict(name='left continuous sample area',accepted=True),dict(name='birth query charged before ability',accepted=True)])
    result=reduce_stage(folder);atomic(folder/'SUMMARY.json',result)
    return dict(accepted=True,checks=checks,check_count=len(checks),scope='13 rejected saved-copy tamper cases and two curve arithmetic fixtures; zero training applications')
if __name__=='__main__':
    r=run();atomic(ROOT/'cycles/cycle2/TAMPER_TESTS.json',r);print(json.dumps(r))
