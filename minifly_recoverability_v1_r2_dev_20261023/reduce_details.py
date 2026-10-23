"""Saved-emission diagnostics only; no native calls or fitting."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;rows=[]
for f in sorted((P/'traces').glob('*_outputs.json')):
 d=json.loads(f.read_text())
 for r in d['results']:
  vals=d['labels'][d['pools'].index(r['pool'])];out=r['metrics']['outputs'];ok=[o['EOS'] and o['bytes']==v for o,v in zip(out,vals)]
  row={k:d[k] for k in ('seed','life','STD','arm')};row.update({k:r[k] for k in ('pool','head','lesion')})
  row.update(per_length={str(L):dict(n=sum(len(v)==L for v in vals),correct=sum(c for c,v in zip(ok,vals) if len(v)==L)) for L in range(9)},early_EOS=sum(o['EOS'] and len(o['bytes'])<len(v) for o,v in zip(out,vals)),late_EOS=sum(o['EOS'] and len(o['bytes'])>len(v) for o,v in zip(out,vals)),truncation=sum(not o['EOS'] for o in out),empty_emission=sum(o['EOS'] and not o['bytes'] for o in out));rows.append(row)
(P/'DETAIL_METRICS.json').write_text(json.dumps(dict(status='ARTIFACT_ONLY',rows=rows),indent=2)+'\n');print(len(rows))
