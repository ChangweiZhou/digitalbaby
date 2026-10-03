"""Independent final aggregate check; never repeats learner trajectories."""
import json
import statistics as st
import bootstrap
from v2_fixture import SCIENCE_WORLDS, ASSAYS, make_world
from v2_core import ARMS
from v2_audit import read
from v2_integrity import verify
from compact_storage import atomic_json


def main():
    source=verify(); result=json.loads((bootstrap.ROOT/'results/SUMMARY.json').read_text())
    if result['identity']!=source or (result['jobs'],result['lives'])!=(768,2304): raise ValueError('summary identity')
    counted=0; births=0
    for a in ARMS:
        values={k:[] for k in ('old','new','revision','e3')}
        for w in SCIENCE_WORLDS:
            for assay in ASSAYS:
                d=read(bootstrap.ROOT/f'results/science/receipts/{w}_{a}_{assay}.json.gz')
                if d['identity']!=source or d['engineering_only']: raise ValueError('receipt source')
                counted+=1; births+=8*len(d['branches'])
                final={b:next(p for p in h['probes'] if p['name']=='final')['rows'] for b,h in d['branches'].items()}
                if assay=='lifetime':
                    intact=set(make_world(w,assay)['intact_old_items'])
                    for stage in ('old','new','revision'):
                        rs=[r for r in final['W'] if r['stage']==stage and (stage!='old' or r['item'] in intact)]
                        values[stage].append(sum(r['correct'] for r in rs)/len(rs))
                else:
                    acc={b:st.mean(r['correct'] for r in rs if r['stage']=='heldout') for b,rs in final.items()}
                    values['e3'].append(acc['W']-acc['N_old'])
        for k,vals in values.items():
            if abs(st.mean(vals)-result['summaries'][a][k]['mean'])>1e-12: raise ValueError('independent aggregate mismatch')
    if counted!=768 or births!=18432: raise ValueError('roster/birth count')
    qualified=[a for a,d in result['decisions'].items() if all(x['pass_gate'] for x in d['tests'].values()) and all(d['guards'].values())]
    if bool(qualified)!=(result['adopted_candidate'] is not None): raise ValueError('adoption gates')
    atomic_json(bootstrap.ROOT/'results/FINAL_AUDIT.json',{'verdict':'PASS','identity':source,
                'committed_jobs':counted,'lives':2304,'canonical_birth_calls':births,
                'independent_all_cue_aggregate_match':True,'adoption_verdict':result['verdict']})
    print('FINAL_AUDIT PASS',flush=True)


if __name__=='__main__': main()
