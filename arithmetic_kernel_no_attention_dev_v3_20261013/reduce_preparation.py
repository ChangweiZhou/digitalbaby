"""Read-only reduction of committed technical receipts. Imports no learner."""
import collections, gzip, hashlib, importlib.util, inspect, json
from pathlib import Path

def accuracy(q):
    ops=collections.defaultdict(list)
    for x in q['rows']:ops[x['operation']].append(int(x['exact']))
    return sum(sum(v)/len(v) for v in ops.values())/len(ops)

def shortcut(q):
    if not q['rows'] or 'shortcuts' not in q['rows'][0]:return None
    ops=collections.defaultdict(list)
    for x in q['rows']:ops[x['operation']].append(x)
    return sum(max(sum(x['shortcuts'][k]==x['expected'] for x in rows)/len(rows) for k in ('exact','canonical','global','length','suffix','bag')) for rows in ops.values())/len(ops)

def run():
    jobs=[];checks=[]
    for cycle in (1,2,3):
        snapshot=Path(f'cycles/cycle{cycle}/source')
        expected={n:hashlib.sha256((snapshot/n).read_bytes()).hexdigest() for n in ('core.py','worlds.py','runner.py','auditor.py')}
        spec=importlib.util.spec_from_file_location(f'locked_auditor_{cycle}',snapshot/'auditor.py');aud=importlib.util.module_from_spec(spec);spec.loader.exec_module(aud)
        for p in sorted(Path(f'cycles/cycle{cycle}/receipts').glob('*.json.gz')):
            with gzip.open(p,'rt') as f:r=json.load(f)
            assert r['source']==expected
            certificate=aud.audit(r,expected) if 'expected_source' in inspect.signature(aud.audit).parameters else aud.audit(r)
            assert certificate['accepted']
            rows={}
            for name,b in r['branches'].items():
                rows[name]={q['label']:{'macro_accuracy':accuracy(q),'correct':sum(x['exact'] for x in q['rows']),'queries':len(q['rows']),'shortcut_envelope':shortcut(q),
                             'per_operation':{op:sum(x['exact'] for x in q['rows'] if x['operation']==op)/sum(x['operation']==op for x in q['rows']) for op in sorted({x['operation'] for x in q['rows']})}}
                            for q in b['final']}
            item={'cycle':cycle,'world':r['world'],'arm':r['arm'],'external_record_applications':r['executed_external_records'],'parameters':r['parameters'],
                  'binary_records_per_history':r['binary_n'],'mixed_records_per_history':r['mixed_n'],'process_cpu':r['total_process_cpu'],'wall_seconds':r['total_wall'],'final':rows,
                  'W_resource':r['branches']['W']['resource'],'receipt':str(p),'source_snapshot':str(snapshot),
                  'historical_metric_caveat':'cycle1 SH_BINARY before_mixed old_taught expected labels provisional; final/W/held probes unaffected' if cycle==1 else None}
            item['causal_differences']={label:{'W_minus_N_BINARY':rows['W'][label]['macro_accuracy']-rows['N_BINARY'][label]['macro_accuracy'],
                                            'W_minus_SH_BINARY':rows['W'][label]['macro_accuracy']-rows['SH_BINARY'][label]['macro_accuracy']}
                                       for label in ('old_held','new_held')}
            item['curves']= {name:{phase:[{'coordinate':entry['external_W_coordinates'],
                          'scores':{q['label']:accuracy(q) for q in entry['probes']},
                          'shortcuts':{q['label']:shortcut(q) for q in entry['probes']},'paid_model_cpu':entry['resource']['cpu_training']+entry['resource']['cpu_generation']}
                          for entry in b.get(phase+'_curve',[])] for phase in ('binary','mixed')}
                          for name,b in r['branches'].items()}
            jobs.append(item)
        checks.append(f'cycle{cycle} sources, original-version auditor and counts')
    assert len(jobs)==31 and sum(x['external_record_applications'] for x in jobs)==58368
    full=[x for x in jobs if x['binary_records_per_history']==8192];assert len(full)==1
    summary={'schema':'KERNEL_V3_PREPARATION_SUMMARY','status':'THREE_CYCLES_COMPLETE_NO_EFFICIENCY_ADOPTION','technical_jobs':len(jobs),'external_record_applications':58368,
             'correlated_final_histories':124,'scientific_jobs':0,'goal_unchanged':True,'attention_or_transformers_used':False,
             'core_adoption':'NONE','population_or_family_exclusion_claim':'NONE','jobs':jobs,
             'limitations':['Technical worlds only; no statistical architecture comparison','Candidates have shorter exposure than full-dose G0; do not compare their raw accuracies as equal-dose outcomes','Short jobs probe fixed slices; full-dose job uses one representative per group, not every presentation','CPU/circuit-visit measurements are not a verified FLOP or tensor peak-memory ledger','Cycle1 shuffled-control pre-mixed taught diagnostic remains explicitly provisional','Formal allocation, all-candidate full-job costs and complete compute metric implementation remain unqualified']}
    Path('SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    audit={'accepted':True,'scope':'technical preparation and truthful receipt reduction, not adoption or formal science','checks':checks,
           'known_historical_issue':summary['limitations'][4],'jobs':31,'external_record_applications':58368,'scientific_jobs':0,'final_model_source_unchanged_from_cycle3':hashlib.sha256(Path('core.py').read_bytes()).hexdigest()==json.loads(Path('cycles/cycle3/source/../CONTRACT_TESTS.json').read_text()).get('core_source_sha256',hashlib.sha256((Path('cycles/cycle3/source')/'core.py').read_bytes()).hexdigest())}
    assert audit['final_model_source_unchanged_from_cycle3']
    Path('FINAL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'technical_jobs':31,'record_applications':58368,'adoption':'NONE','full_G0_final':full[0]['final']['W'],'full_G0_cpu':full[0]['process_cpu']},indent=2))
    return summary
if __name__=='__main__':run()
