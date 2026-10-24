"""Recompute recorded outputs with stdlib logic; no learner execution."""
import csv
import hashlib
import json
from pathlib import Path
from v144_audit import HERE,ROOT,write_json


def balanced_parity(outputs,held,k):
    groups={48:[],49:[]}
    for a,b in held:
        target=48+(a+b)%2
        groups[target].append(outputs[str(a)+str(b)][k]==target)
    return sum(sum(v)/len(v) for v in groups.values())/2


def main():
    result={'native':{},'reference':{},'formal_science_jobs':0,'holdback_opened':False}
    native_rows=[]
    # Published development split only; no parent model import or replay.
    parent=json.loads((HERE/'cycles/PARENT_D2_EVIDENCE.json').read_text())
    orbits=parent['held_orbits'];held=[p for a,b in orbits for p in ([(a,b)] if a==b else [(a,b),(b,a)])]
    for arm in ['LEARN','FROZEN']:
        p=HERE/'results'/('B_'+arm)/'RESULT.json'
        if not p.exists():continue
        value=json.loads(p.read_text());native_rows.append(value)
        snapshots=value['metrics']['snapshots'];fin=snapshots[-1]
        expected=parent['arms'][arm]['outputs']
        match=fin['outputs']==expected
        frozen_invariant=len({s['code_hashes']['L'] for s in snapshots})==1
        result['native'][arm]={'complete':value['complete'],'records':value['records'],'cpu_s':value['cpu_s'],
           'final_balanced_bank':balanced_parity(fin['outputs'],held,2),
           'final_balanced_system':balanced_parity(fin['outputs'],held,0),
           'final_taught_bank':fin['accuracy']['taught']['L'],
           'matches_published_D2_final_outputs':match,
           'frozen_code_hash_invariance':frozen_invariant if arm=='FROZEN' else None,
           'birth_overlap':snapshots[0]['meters']['L']['mean_offdiag_jaccard'],
           'final_overlap':fin['meters']['L']['mean_offdiag_jaccard'],
           'birth_rank':snapshots[0]['hidden_effective_rank'],'final_rank':fin['hidden_effective_rank'],
           'final_within_class_overlap':fin['within_class_jaccard'],
           'final_between_class_overlap':fin['between_class_jaccard'],
           'windows':value['metrics']['windows']}
        if not match:raise AssertionError('native D2 outputs differ from published parent')
        if arm=='FROZEN' and not frozen_invariant:raise AssertionError('FROZEN codes moved')
    write_json(HERE/'cycles/B_VALIDITY_INPUT.json',{'rows':native_rows,'required':['B_LEARN','B_FROZEN'],'closure_ok':True})
    table=[]
    for p in sorted((HERE/'results').glob('C2_*/RESULT.json')):
        v=json.loads(p.read_text());settings=v['settings'];summaries=[]
        for snap in v['metrics']['snapshots']:
            for split in ['held','taught']:
                groups={}
                rows=snap[split]['rows']
                for row in rows:
                    a,b=row['pair'];output=bytes.fromhex(row['output_hex'])
                    hit=output==str(a+b).encode() and row['status']=='EOS'
                    assert hit==row['exact']
                    groups.setdefault(tuple(sorted((a,b))),[]).append(hit)
                exact=sum(sum(g)/len(g) for g in groups.values())/len(groups)
                assert abs(exact-snap[split]['orbit_exact'])<1e-12
                if split=='held':
                    table.append(dict(mode=settings['mode'],checkpoint=snap['label'],records=snap['records'],
                                      target_tokens=snap['observed_target_tokens'],cpu_s=snap['cpu_s'],
                                      held_orbit_exact=exact,own_eos=snap['held']['own_eos']))
        fin=v['metrics']['snapshots'][-1]
        result['reference'][settings['mode']]={'complete':v['complete'],'records':v['records'],
          'target_tokens':v['observed_target_tokens'],'cpu_s':v['cpu_s'],
          'held_orbit_exact':fin['held']['orbit_exact'],'taught_orbit_exact':fin['taught']['orbit_exact'],
          'held_own_eos':fin['held']['own_eos'],'held_payload_only':fin['held']['payload_only'],
          'teacher_forced_token_accuracy':fin['teacher_forced_diagnostic']['teacher_forced_token_accuracy'],
          'countermodels':v['metrics']['countermodels']}
    if table:
        with (HERE/'SCREEN_TABLE.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
    ledger_rows=[];attempts={}
    for p in sorted((HERE/'ops').glob('*.jsonl')):
        rows=[json.loads(x) for x in p.read_text().splitlines()]
        ledger_rows.extend(dict(source_log=p.name,**x) for x in rows)
        starts=sum(x['event']=='record_start' for x in rows)
        completes=[x for x in rows if x['event']=='record_complete']
        attempts[p.name]={'record_starts':starts,'record_completions':len(completes),
                         'target_tokens':sum(x.get('target_tokens',0) for x in completes),
                         'process_cpu_last_s':rows[-1]['process_cpu_s'],
                         'last_event':rows[-1]['event'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (HERE/'ATTEMPT_LEDGER.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in ledger_rows))
    result['attempts']=attempts
    result['accounting_limitations']=['Some pre-ledger import/fixture setup CPU unmeasured; do not call retained CPU all-in campaign CPU.',
      'Fixture aggregate counts are additional to per-record worker logs; consult cycle notes.',
      'Record counters do not measure all internal native events or bytes copied.']
    write_json(HERE/'SUMMARY.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='native' and k!='attempts'},indent=2))


if __name__=='__main__':main()
