"""Read-only rescoring. No learning; labels are used only after policy emission."""
import gzip,json,hashlib
from pathlib import Path
from collections import defaultdict
from authority_policy import Observation,emit,POLICIES
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'next_core_budget_v2_20261006/science/confirm/receipts'
LATIN=ROOT.parents[1]/'DIGITALBABY_GITHUB_UPDATES_20261005/studies'

def score(row):
    predictions=row.get('predictions') or [row.get('prediction',row)]
    obs=[Observation(tuple(p['shared']),tuple(p['private'])) for p in predictions]
    result={p:emit(obs,p) for p in POLICIES}
    assert result['ERROR']['emitted']==row['emitted'], 'ERROR arithmetic parity'
    if 'Q_HALF_emitted' in row:assert result['Q_HALF']['emitted']==row['Q_HALF_emitted']
    if 'Q_HALF' in row.get('policies',{}):assert result['Q_HALF']['emitted']==row['policies']['Q_HALF']['emitted']
    if 'alpha_half' in row.get('policies',{}):assert result['Q_HALF']['emitted']==row['policies']['alpha_half']['emitted']
    return result

def main(output_cycle=1):
    rows=[];files=[];parity=0
    # Deliberately freeze the single rule before inspecting these scored outcomes.
    from bridge_runtime import make_world
    for path in sorted(BASE.glob('*_ERROR_*.json.gz')):
        d=json.loads(gzip.decompress(path.read_bytes()))
        assert d['complete'] and not d['development']
        files.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        intact=set(make_world(d['world'],'lifetime')['intact_old_items']) if d['assay']=='lifetime' else None
        for branch,bd in d['branches'].items():
            for probe in bd['probes']:
                for row in probe['rows']:
                    results=score(row);parity+=1
                    if d['assay']=='lifetime' and row['stage']=='old' and row['item'] not in intact:continue
                    q=results['Q_EARNED']
                    rows.append({'source':'binary_confirmation' if d['assay']=='reuse' else 'lifetime_confirmation',
                                 'world':d['world'],'branch':branch,'checkpoint':probe['name'],'group':row['stage'],
                                 'item':row.get('item',row.get('pair')),'order':row.get('order'),
                                 'alpha':q['alpha'],'trigger':q['trigger'],'shared_margin':q['shared_margin'],
                                 'private_margin':q['private_margin'],'agreement':q['shared_winner']==q['private_winner'],
                                 'correct':{p:int(r['emitted']==row['target']) for p,r in results.items()}})
    # Raw Latin publication is incomplete: one official R_center world, plus one
    # ERROR technical pilot. Never call these 64 ERROR confirmation worlds.
    for source,world,directory,prefix in (
        ('latin_R_center_single_official',310001,LATIN/'rcenter-survivor-rounds-20261003/receipts/science/310001',''),
        ('latin_ERROR_technical_pilot',320200,LATIN/'error-completion-20261004/receipts/cycle3/pilot-320200','ERROR__')):
        for path in sorted(directory.glob('probe-*.json')):
            d=json.loads(path.read_text())
            if prefix and not d['branch'].startswith(prefix):continue
            files.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            for row in d['rows']:
                results=score(row);parity+=1;q=results['Q_EARNED']
                rows.append({'source':source,'world':world,'branch':d['branch'].removeprefix(prefix),
                             'checkpoint':d['when'],'group':row['set'],'item':row['item'],
                             'alpha':q['alpha'],'trigger':q['trigger'],'shared_margin':q['shared_margin'],
                             'private_margin':q['private_margin'],'agreement':q['shared_winner']==q['private_winner'],
                             'correct':{p:int(r['emitted']==row['target']) for p,r in results.items()}})
    groups=defaultdict(list)
    for r in rows:groups[(r['source'],r['world'],r['branch'],r['checkpoint'],r['group'])].append(r)
    per_world=[]
    for k,rs in sorted(groups.items()):
        per_world.append(dict(zip(('source','world','branch','checkpoint','group'),k),n=len(rs),
                              accuracy={p:sum(r['correct'][p] for r in rs)/len(rs) for p in POLICIES},
                              trigger_rate=sum(r['trigger'] for r in rs)/len(rs),
                              helped=sum(r['correct']['Q_EARNED']>r['correct']['ERROR'] for r in rs),
                              hurt=sum(r['correct']['Q_EARNED']<r['correct']['ERROR'] for r in rs)))
    aggregates=defaultdict(list)
    for r in per_world:aggregates[(r['source'],r['branch'],r['checkpoint'],r['group'])].append(r)
    summary=[]
    for k,rs in sorted(aggregates.items()):
        summary.append(dict(zip(('source','branch','checkpoint','group'),k),worlds=len(rs),
                            accuracy={p:sum(r['accuracy'][p] for r in rs)/len(rs) for p in POLICIES},
                            trigger_rate=sum(r['trigger_rate'] for r in rs)/len(rs)))
    out=ROOT/f'cycles/cycle{output_cycle}';out.mkdir(parents=True,exist_ok=True)
    (out/'OFFLINE_SUMMARY.json').write_text(json.dumps({'status':'EXPLORATORY_ONLY','parity_rows':parity,
        'new_learning_records':0,'rule_chosen_before_scoring':True,'threshold_search':False,
        'latin_raw_limitation':'Only R_center world 310001 and ERROR technical pilot 320200 are accessible. Full 64-world Latin per-option vectors are not published.',
        'sources':files,'summary':summary,'per_world':per_world},indent=2)+'\n')
    with gzip.open(out/'OFFLINE_ROWS.json.gz','wt') as f:json.dump(rows,f)
    for s in summary:
        if s['checkpoint']=='final' and s['group'] in ('heldout','old','new','revision'):
            print(json.dumps(s))
    print('Exact ERROR/Q_HALF baseline parity rows:',parity)
if __name__=='__main__':
    import sys
    main(int(sys.argv[1]) if len(sys.argv)>1 else 1)
