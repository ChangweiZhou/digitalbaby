import gzip,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]
d=json.loads(gzip.decompress((ROOT/'data/development_history.json.gz').read_bytes()))
p=json.loads((ROOT/'data/PROVENANCE.json').read_text())
assert hashlib.sha256((ROOT/'data/development_history.json.gz').read_bytes()).hexdigest()==p['bundle_sha256']
assert d['provenance']==p['source_receipts']
byworld={r['world']:r for r in d['worlds']}
def subset(a,b):
    if isinstance(a,dict):
        for k,v in a.items():subset(v,b[k])
    else:assert a==b
count=0
for source in p['source_receipts']:
    f=BASE/source['source_path'];raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==source['sha256'];original=json.loads(gzip.decompress(raw))
    extracted=byworld[source['world']][source['arm']]
    for key,val in extracted.items():
        if key=='records':
            rows=[r for r in original['records'] if r['branch']=='W'];assert len(rows)==len(val)
            for small,full in zip(val,rows):subset(small,full)
        else:subset(val,original[key])
    count+=1
r=json.loads((ROOT/'results/CYCLE3_RESULT.json').read_text())
repairs=breaks=rawcorrect=corrected=0
for row in r['worlds']:
    def argmax(x):return max(range(4),key=lambda i:x[i])
    v=row['raw_values'];b=row['baseline_predictions'];c=row['corrected_values'];ys=row['true_labels_evaluator_only'];rr=bb=0
    for vi,bi,ci,y in zip(v,b,c,ys):
        assert max(abs(ci[j]-(vi[j]-bi[j])) for j in range(4))<1e-12
        old=argmax(vi)==y;new=argmax(ci)==y
        rawcorrect+=old;corrected+=new;rr+=not old and new;bb+=old and not new
    assert rr==row['wrong_to_right'] and bb==row['right_to_wrong'];repairs+=rr;breaks+=bb
assert corrected==rawcorrect+repairs-breaks
assert (rawcorrect,corrected,repairs,breaks)==(202,186,38,54)
replays={}
for n in (1,3):
    a=json.loads((ROOT/f'results/CYCLE{n}_RESULT.json').read_text());b=json.loads((ROOT/f'results/CYCLE{n}_PORTABLE_REPLAY.json').read_text())
    a.pop('resources',None);b.pop('resources',None);assert a==b;replays[str(n)]='exact equality excluding resources'
out={'source_receipts_verified':count,'extraction_exact':True,'raw_correct':rawcorrect,'B_correct':corrected,'repairs':repairs,'breaks':breaks,'portable_replays':replays}
(ROOT/'design/FINAL_EVIDENCE_CHECK.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
