from pathlib import Path
from collections import Counter, defaultdict
import json, csv, hashlib, statistics, gzip

ROOT = Path('/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929')
BASE = ROOT/'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007'
OUT = BASE/'FULL_REVIEW_POST_U034_20261011'
OUT.mkdir(exist_ok=True)
fingerprints = {}

def read(p):
    p=Path(p).resolve()
    raw=p.read_bytes()
    fingerprints[str(p)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    return raw.decode('utf-8')

def jread(p): return json.loads(read(p))
def canon(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def unique(xs): return sorted({canon(x) for x in xs})
def csvread(p): return list(csv.DictReader(read(p).splitlines()))

reg=jread(BASE/'VERSION_REGISTER_20261011/VERSIONS.json')
versions=reg['versions']
hist=jread(BASE/'HISTORICAL_108_SCREEN.json')
cat=jread(BASE/'CATALOGUE_136_SCREEN.json')
ledger=jread(ROOT.parent/'FULL_MECHANISM_LEDGER_SOURCE_20261002/MECHANISM_LEDGER.json')
checks=[]
def check(name, passed, details=None):
    checks.append({'check':name,'pass':bool(passed),'details':details})

check('historical roster M001-M108', [x['id'] for x in hist]==[f'M{i:03d}' for i in range(1,109)])
check('historical original fields preserved', all({k:x[k] for k in original}==original for x,original in zip(hist,ledger)) and len(hist)==len(ledger))
check('catalogue source rows 1-136', [x['source_row'] for x in cat]==list(range(1,137)))
catalogue_text={p:read(p).splitlines() for p in {x['source_path'] for x in cat}}
cat_mismatch=[x['source_row'] for x in cat if catalogue_text[x['source_path']][x['source_line']-1].strip()!=x['source_text'].strip()]
check('catalogue quoted source lines unchanged', not cat_mismatch,cat_mismatch)
counts=dict(Counter(x['series'] for x in versions))
check('version series counts match register', counts==reg['coverage_counts'], counts)
check('254 unique version IDs', len(versions)==254 and len({x['version_id'] for x in versions})==254)

paths=defaultdict(list)
def scan(x, vid, location=''):
    if isinstance(x,dict):
        for k,v in x.items():scan(v,vid,location+'.'+k)
    elif isinstance(x,list):
        for i,v in enumerate(x):scan(v,vid,location+f'[{i}]')
    elif isinstance(x,str) and (x.startswith('/Users/') or x.startswith('../')):
        paths[x].append({'version_id':vid,'field':location})
for v in versions:scan(v,v['version_id'])
path_checks=[]
for raw,refs in sorted(paths.items()):
    p=(Path(raw) if raw.startswith('/') else BASE/'VERSION_REGISTER_20261011'/raw).resolve()
    path_checks.append({'recorded_path':raw,'resolved_path':str(p),'exists':p.exists(),'references':refs})
    if p.is_file():read(p)
check('all 22 version source/operation/evidence path targets exist', all(x['exists'] for x in path_checks), {'distinct_targets':len(path_checks)})

record_checks=[]
for series in ['V23','V24','V25']:
    xs=[v for v in versions if v['series']==series]
    source=jread(xs[0]['source'])
    rows=csvread(Path(xs[0]['source']).parent/'SUMMARY.csv')
    if series=='V23':source_names={x['name'] for x in source}
    elif series=='V24':source_names={x['memory']['name'] for x in source}
    else:source_names={x['name'] for x in source}
    check(series+' manifest name coverage',source_names=={x['name'] for x in xs},{'source_names':len(source_names),'register_names':len(xs)})
    for v in xs:
        name=v['name']; failures=[]
        if series=='V23':
            operation=next(x for x in source if x['name']==name)
            if v['operation']!=operation:failures.append('operation differs from manifest')
            rs=[x for x in rows if x['arm']==name and x['status']=='completed']
            expected={'n':len(rs)}
            mapping={'A_after_B_percent':'A_after_B','B_learned_percent':'B_acquired','revision_after_C_percent':'first_revision_after_C','second_revision_percent':'second_revision','terminal_all_percent':'terminal_all_current'}
            for k,f in mapping.items():expected[k]=round(statistics.mean(float(x[f]) for x in rs)*100,2)
            expected['mutable_KiB']=round(statistics.mean(float(x['peak_mutable_numeric_bytes']) for x in rs)/1024,1)
            if expected!=v['evidence']['summary']:failures.append({'summary_expected':expected,'summary_recorded':v['evidence']['summary']})
        elif series=='V24':
            ss=[x for x in source if x['memory']['name']==name]
            if unique(x['memory'] for x in ss)!=unique(v['operation']['memory_specs']):failures.append('memory_specs differ')
            contexts=[{k:x[k] for k in ['encoder','encoder_variant','order','panel','task','teaching']} for x in ss]
            if unique(contexts)!=unique(v['operation']['contexts']):failures.append('contexts differ')
            for summary in v['evidence']['summaries']:
                matching=[r for r in rows if r['arm']==name and all(r.get(k)==summary[k] for k in ['panel','encoder','encoder_variant','task','order','teaching','view'])]
                if not any(all(r.get(k)==val for k,val in summary.items()) for r in matching):failures.append({'summary_not_in_source':summary})
        else:
            ss=[x for x in source if x['name']==name]
            configs=[{k:val for k,val in x.items() if k not in ['id','seed','comparisons']} for x in ss]
            if unique(configs)!=unique(v['operation']['configurations']):failures.append('configurations differ')
            if unique(c for x in ss for c in x['comparisons'])!=unique(v['operation']['declared_parent_comparisons']):failures.append('declared comparisons differ')
            for summary in v['evidence']['summaries']:
                if not any(all(r.get(k)==val for k,val in summary.items()) for r in rows if r['name']==name):failures.append({'summary_not_in_source':summary})
        record_checks.append({'version_id':v['version_id'],'operation_and_summary_match':not failures,'failures':failures})
    check(series+' exact manifest operations and stored summary values',all(r['operation_and_summary_match'] for r in record_checks if r['version_id'].startswith(series+'/')))

for series in ['SAMPLE_GEN_V2','FEEDBACK_EFFICIENCY_V1','OBSERVATION_SUPPORT_V1','SUPPORT_DIRECTION_V1']:
    xs=[v for v in versions if v['series']==series]
    raw=xs[0]['operation']['design_contract']
    path=Path(raw) if raw.startswith('/') else BASE/'VERSION_REGISTER_20261011'/raw
    contract=jread(path)
    roster={x['id'] if isinstance(x,dict) else x for x in contract['roster']}
    check(series+' contract roster coverage',roster=={x['name'] for x in xs})

u034=ROOT/'addition_uncertainty_restoration_dev_v1_20261011'
u034_contract=jread(ROOT/'addition_uncertainty_restoration_v1_design_20261011/DESIGN_CONTRACT.json')
summary=jread(u034/'results/SUMMARY.json')
source_changed=[]
for raw,expected in summary['source'].items():
    text=read(ROOT/raw)
    if hashlib.sha256(text.encode()).hexdigest()!=expected:source_changed.append(raw)
check('U034 current source hashes match final summary',not source_changed,{'files':len(summary['source']),'changed':source_changed})
index=jread(u034/'results/COMMIT_INDEX.json')['entries']
u034_rows=[];receipt_hash_failures=[]
for entry in index:
    path=u034/entry['receipt'];raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:receipt_hash_failures.append(str(path))
    receipt=json.loads(gzip.decompress(raw))
    kernel=receipt['kernel_receipt'];w=kernel['branches']['W'];fork=w['forks']['320']
    curve=[fork['new_checkpoints'][str(n)]['delayed_new_held']['accuracy'] for n in u034_contract['new_checkpoints']]
    # J's prespecified weights read directly from decision.py; no learner runs.
    j=.25*curve[0]+.5*curve[1]+.25*curve[2]
    records=[r for b in kernel['branches'].values() for group in [b['old_records']]+[f['new_records'] for f in b['forks'].values()] for r in group]
    records += [r for f in kernel['extra_forks'].values() for r in f['new_records']]
    cpu=sum(r['production_cpu_seconds'] for r in records)
    payload=max(f['final_checkpoint']['payload_bytes'] for b in kernel['branches'].values() for f in b['forks'].values())
    payload=max(payload,*[f['final_checkpoint']['payload_bytes'] for f in kernel['extra_forks'].values()])
    u034_rows.append({'world':receipt['world'],'arm':receipt['arm'],'stage':receipt['stage'],'noise':kernel['noise'],'family':kernel['family'],'J':j,'old_final':fork['final_old_held']['accuracy'],'cpu':cpu,'payload':payload})
check('U034 all 60 receipt hashes match commit index',len(index)==60 and not receipt_hash_failures,receipt_hash_failures)
check('U034 12 paired DEV worlds and five complete arms',len({x['world'] for x in u034_rows})==12 and all(x['stage']=='DEV' for x in u034_rows) and all(len([x for x in u034_rows if x['world']==w])==5 for w in {x['world'] for x in u034_rows}))
u034_metrics={}
for v in [x for x in versions if x['series']=='SUPPORT_DIRECTION_V1']:
    rows=[x for x in u034_rows if x['arm']==v['name']];blurry=[x for x in rows if x['noise']=='BLURRY']
    m={'BLURRY_worlds':len(blurry),'BLURRY_NEW_J':statistics.mean(x['J'] for x in blurry),'BLURRY_final_old_held_accuracy':statistics.mean(x['old_final'] for x in blurry),'training_CPU_seconds_sum':sum(x['cpu'] for x in rows),'max_payload_bytes':max(x['payload'] for x in rows)}
    u034_metrics[v['name']]=m
for v in [x for x in versions if x['series']=='SUPPORT_DIRECTION_V1']:
    m=u034_metrics[v['name']];m['CPU_vs_SUPPORT']=m['training_CPU_seconds_sum']/u034_metrics['COV-SUPPORT']['training_CPU_seconds_sum']
    stored=v['evidence']['descriptive_metrics']
    check(v['version_id']+' metrics independently reduced from receipts',all(abs(m[k]-stored[k])<=1e-10*max(1,abs(m[k])) for k in stored),m)
check('U034 verdict DEV only, no selected recipe or confirmation',summary['verdict']=='NO_ADVANCE_RETAIN_USER_CORE' and summary['selected'] is None and summary['confirmation_entropy_created'] is False)

for filename, originals in [('HISTORICAL_108_COMPARISON.json',hist),('CATALOGUE_136_COMPARISON.json',cat),('VERSION_254_COMPARISON.json',versions)]:
    rows=jread(OUT/filename)
    check(filename+' complete original records preserved',len(rows)==len(originals) and [x['original'] for x in rows]==originals and len({x['id'] for x in rows})==len(originals))
    check(filename+' no family-wide exclusion or new run inferred',all(x['family_wide_exclusion'] is False and x['new_run_authorized'] is False for x in rows))

result={'audit_date':'2026-10-11','scope':'Document/source reconciliation of 108 historical ledger entries, 136 catalogue entries, 254 version records; targeted independent U034 receipt hash and descriptive-metric reduction. No experiment rerun; not an independent validation of every original scientific result.',
        'checks':checks,'counts':{'historical':len(hist),'catalogue':len(cat),'versions':len(versions),'series':counts},
        'version_path_checks':path_checks,'historical_version_checks':record_checks,'u034_independent_metrics':u034_metrics,'u034_receipt_rows':u034_rows,'source_fingerprints':fingerprints,
        'interpretation_limits':['These overlapping rosters are not independent mechanisms or a scientific survival denominator.','A path-existing check alone does not certify scientific validity.','The stored V24/V25 summary is a selection of source columns; full source cost tables remain necessary for matched cost claims.','No stopped exact recipe closes a mechanism family. Missing/void causal branches remain unresolved rather than scientific negatives.']}
(OUT/'ASSISTANT_SOURCE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'checks':checks,'failed_historical_versions':[r['version_id'] for r in record_checks if not r['operation_and_summary_match']],'fingerprinted_files':len(fingerprints)},ensure_ascii=False,indent=2))
