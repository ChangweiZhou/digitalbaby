"""Additional independent state/derivative and auditor tamper fixtures."""
import copy, gzip, hashlib, json
from pathlib import Path
import torch
from core import Learner
from runner import model_seed
from auditor import audit,independent_answer

def run(cycle):
    out=Path(f'cycles/cycle{cycle}');checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    a=Learner('P',model_seed(9040,'P'));b=Learner('P',model_seed(9040,'P'))
    a.observe(b'a',b'b',True);b.observe(b'a',b'b',True)
    before=b.digest(); b.observe(b'never observed permitted',b'forbidden answer',False);b.generate(b'scratch',2)
    check('pending learning graph/state freeze',b.digest()==before)
    for _ in range(3):a.observe(b'c',b'd',True);b.observe(b'c',b'd',True)
    check('forbidden stream has no effect on later permitted gradients',a.digest()==b.digest())
    check('modulation explicitly bounded',bool((a.m.abs()<=1).all()))
    # Central differences through two real records into an earlier modulation write.
    c=Learner('P',model_seed(9042,'P'))
    def policy_loss():
        c.m=torch.zeros_like(c.m)
        first,_=c.loss_record(b'aa',b'x',True);second,_=c.loss_record(b'bb',b'y',True)
        return (first+second)/2
    l=policy_loss();l.backward();name,p=next((n,p) for n,p in c.net.named_parameters() if n.startswith('policy.') and p.grad is not None)
    i=int(p.grad.abs().reshape(-1).argmax());analytic=float(p.grad.reshape(-1)[i]);old=float(p.detach().reshape(-1)[i]);vals=[]
    for sign in (1,-1):
        with torch.no_grad():p.reshape(-1)[i]=old+sign*.003
        vals.append(float(policy_loss().detach()))
    with torch.no_grad():p.reshape(-1)[i]=old
    numerical=(vals[0]-vals[1])/.006
    check('future policy credit finite difference',abs(analytic-numerical)<.001+abs(analytic)*.06 and abs(analytic)>1e-8)
    c=Learner('G1',model_seed(9043,'G1'))
    permitted=[]
    for i in range(70):
        pp=bytes([i,255]);rr=bytes([i+1]);permitted.append((pp,rr));c.observe(pp,rr,True)
    before=c.digest();c.observe(b'forbidden',b'target',False)
    check('reservoir provenance',all(item in permitted for item in c.buffer) and len(c.buffer)==64)
    check('reservoir no-write RNG/state freeze',c.digest()==before)
    check('one prior replay maximum',c.resource['replay_records']==69 and c.steps==70)
    before=c.digest();counter=dict(c.resource)
    try:c.observe(b'input',b'x'*65,True)
    except ValueError:pass
    else:raise AssertionError('credit cap missing')
    check('invalid record rejected before mutation',before==c.digest() and counter==c.resource)
    # Pending credit and fast state are exposed in state checks, not just weights.
    c=Learner('P',model_seed(9044,'P'));c.observe(b'a',b'b',True)
    try:c.clone()
    except RuntimeError:pass
    else:raise AssertionError('forked unresolved graph')
    check('fork refuses unresolved learning window',True)
    paths=sorted(Path('cycles/cycle1/receipts').glob('*.json.gz'))
    if cycle>2:paths=sorted(Path('cycles/cycle2/receipts').glob('*.json.gz'))
    with gzip.open(paths[0],'rt') as f:receipt=json.load(f)
    synthetic_fixture=False
    if cycle==2:
        # Preserve original cycle1 receipts. This copy is a synthetic auditor fixture,
        # not a rewritten trajectory or a native repaired experimental receipt.
        for q in receipt['branches']['SH_BINARY']['before_mixed']:
            if q['label']=='old_taught':
                for x in q['rows']:
                    x['expected']=independent_answer(x['prompt'],receipt['world'])
                    x['exact']=x['actual']==x['expected'] and x['status']=='EOS'
        synthetic_fixture=True
    snapshot=Path(f'cycles/cycle{1 if cycle==2 else 2}/source')
    expected={name:hashlib.sha256((snapshot/name).read_bytes()).hexdigest() for name in ('core.py','worlds.py','runner.py','auditor.py')}
    audit(receipt,expected)
    mutations=[
      ('science envelope',lambda r:r.update(technical=False)),
      ('total count',lambda r:r.update(executed_external_records=1)),
      ('source identity',lambda r:r['source'].update({'core.py':'0'*64})),
      ('N old branch permission',lambda r:r['branches']['N_BINARY']['segments'][0].update(allowed=True)),
      ('N old record permission',lambda r:r['branches']['N_BINARY']['segments'][0]['records'][0].update(allowed=True)),
      ('N old state parity',lambda r:r['branches']['N_BINARY']['segments'][0].update(after='broken')),
      ('N mixed permission',lambda r:r['branches']['N_MIXED']['segments'][0].update(allowed=True)),
      ('actual target label',lambda r:r['branches']['W']['segments'][0]['records'][0].update(reply='ff')),
      ('actual prediction target',lambda r:r['branches']['W']['segments'][0]['records'][0]['predictions'][0].update(target=256)),
      ('own output scoring',lambda r:r['branches']['W']['final'][0]['rows'][0].update(exact=not r['branches']['W']['final'][0]['rows'][0]['exact'])),
      ('query state purity',lambda r:r['branches']['W']['final'][0].update(after='changed')),
      ('bounded update',lambda r:r['branches']['W']['last_update'].update(applied_RMS=.1)),
      ('output cap',lambda r:r['branches']['W']['final'][0]['rows'][0].update(actual='00'*17)),
      ('N optimizer count',lambda r:r['branches']['N_BINARY']['segments'][0]['counts_after'].update(optimizer_steps=10))]
    for name,mutate in mutations:
        bad=copy.deepcopy(receipt);mutate(bad)
        try:audit(bad,expected)
        except (AssertionError,ValueError,KeyError,UnicodeError):checks.append('tamper rejected: '+name)
        else:raise AssertionError('tamper accepted: '+name)
    result={'accepted':True,'checks':checks,'count':len(checks),'policy_analytic':analytic,'policy_numeric':numerical,'scientific_jobs':0,'generic_fixture_permitted_records':70+9,'synthetic_corrected_legacy_auditor_fixture':synthetic_fixture,'scope':'state contracts and selected derivative/auditor tamper fixtures; synthetic fixture is not a native receipt'}
    (out/'ADDITIONAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'checks':len(checks),'accepted':True}),flush=True)
if __name__=='__main__':
    import sys
    run(int(sys.argv[1]))
