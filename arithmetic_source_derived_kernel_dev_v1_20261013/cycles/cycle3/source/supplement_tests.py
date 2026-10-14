"""Receipt/reducer tamper fixtures only; no new model forward or training."""
import copy
import gzip
import json
import math
from pathlib import Path
from auditor import audit
from reducer import staircase, select, holm


def run():
    rows=[]
    def check(name):rows.append(dict(check=name,accepted=True))
    assert abs(staircase([(0,.1),(2,.9),(4,.5)],4)-.5)<1e-12
    assert abs(staircase([(0,.1),(2,.9),(4,.5)],3)-(1.1/3))<1e-12
    assert abs(staircase([(1,.8),(3,.2)],5)-.4)<1e-12
    for points in ([(-1,.1)],[(2,.3),(1,.2)],[(0,1.1)]):
        try:staircase(points,5)
        except ValueError:pass
        else:raise AssertionError('invalid curve admitted')
    check('left-continuous, paid-birth, hold-last, common-cost clipped integration')
    assert holm([.001,.1,.005])==[True,False,True]
    assert holm([1.]*8)==[False]*8
    check('Holm full family and blocked tail')
    def value(js,gc,old=.5,new=.6,cpu=10):
        operators={o:old for o in ('+','-','*','/')};pairs={a+b:new for a in ('+','-','*','/') for b in ('+','-','*','/')}
        w=dict(J_sample=js,J_compute=gc,production_cpu_end=cpu,final=dict(old_held=operators,new_held=pairs),shortcut=dict(old_held={o:.1 for o in operators},new_held={p:.1 for p in pairs}))
        n=copy.deepcopy(w);n['final']=dict(old_held={o:.1 for o in operators},new_held={p:.1 for p in pairs})
        return dict(W=w,N_BINARY=copy.deepcopy(n),SH_BINARY=copy.deepcopy(n),N_MIXED=copy.deepcopy(n))
    fixture={name:[value(js,gc)]*4 for name,js,gc in [('B16',.1,.1),('B24',.11,.1),('B69',.12,.12),('E6',.3,.3),('E6_FIX',.31,.31)]}
    x=select(fixture);assert x['selected']=='E6_FIX' and x['reference']=='B69' and x['twin']=='E6'
    check('fixed candidate/reference/twin selection before fresh confirmation')
    for arm in ('E6','E6_FIX'):
        fixture[arm]=[value(.01,.01)]*4
    assert select(fixture)['selected'] is None
    check('negative/no-advance is completed fixed-roster outcome')
    base=Path(__file__).parent/'cycles/cycle2'
    r=json.load(gzip.open(next((base/'receipts').glob('*B16.json.gz')),'rt'))
    expected=json.loads((base/'SOURCE_LOCK.json').read_text())['source']
    assert audit(r,expected)['accepted']
    mutations=[]
    mutations.append(('old-disabled-permission',lambda x:x['branches']['N_BINARY']['segments'][0].__setitem__('allowed',True)))
    mutations.append(('old-disabled-write',lambda x:x['branches']['N_BINARY']['segments'][0]['records'][0].__setitem__('optimizer_steps',1)))
    mutations.append(('old-disabled-state',lambda x:x['branches']['N_BINARY']['segments'][0].__setitem__('after','0'*64)))
    mutations.append(('source',lambda x:x['source'].__setitem__('core.py','0'*64)))
    mutations.append(('true-label',lambda x:x['branches']['W']['segments'][0]['records'][0].__setitem__('reply','00')))
    def held(x):return x['branches']['W']['binary_curve'][0]['probes'][0]['rows']
    mutations.append(('held-label',lambda x:held(x)[0].__setitem__('expected','00')))
    mutations.append(('held-output',lambda x:held(x)[0].__setitem__('exact',not held(x)[0]['exact'])))
    mutations.append(('operator-reweight',lambda x:held(x)[0].__setitem__('operation','?')))
    mutations.append(('duplicate-held',lambda x:held(x).__setitem__(1,copy.deepcopy(held(x)[0]))))
    mutations.append(('held-countermodel',lambda x:held(x)[0]['shortcuts'].__setitem__('global','00')))
    mutations.append(('cost-coordinate',lambda x:x['branches']['W']['binary_curve'][1].__setitem__('production_cpu',-1)))
    mutations.append(('external-record-count',lambda x:x.__setitem__('executed_external_records',1)))
    mutations.append(('prefix-inheritance',lambda x:x['branches']['N_MIXED']['segments'][0].__setitem__('before','0'*64)))
    mutations.append(('data-manifest-identity',lambda x:x.__setitem__('data_digest','0'*64)))
    rejected=[];admitted=[]
    for name,mutate in mutations:
        x=copy.deepcopy(r);mutate(x)
        try:audit(x,expected)
        except (AssertionError,ValueError,KeyError):rejected.append(name)
        else:admitted.append(name)
    result=dict(accepted=not admitted,checks=rows,rejected_tampers=rejected,admitted_tampers=admitted,
                native_arithmetic_records=0,model_forward_passes=0,scientific_jobs=0)
    return result


if __name__=='__main__':
    r=run();(Path(__file__).parent/'SUPPLEMENT_TESTS.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r))
    if not r['accepted']:raise SystemExit(1)
