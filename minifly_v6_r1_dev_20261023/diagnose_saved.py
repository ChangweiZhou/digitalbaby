"""Posthoc localization from saved outputs only; never a new primary endpoint."""
import json,collections
from pathlib import Path
P=Path(__file__).resolve().parent
out=[]
for p in sorted((P/'results/probes').glob('*_FINAL.json')):
 v=json.loads(p.read_text())
 for pool in ('old','new','held'):
  rs=[r for r in v['rows'] if r['pool']==pool];total=len(rs);c=collections.Counter(bytes(r['output']).hex() for r in rs);f=[]
  for r in rs:
   L=len(r['target']);m=r['margins'];f.append(dict(first_byte=bool(r['output'] and r['output'][0]==r['target'][0]),oracle_length_prefix=r['output'][:L]==r['target'],EOS_at_true_length=bool(len(m)>L and m[L][8]>0),first_EOS_margin=m[0][8]))
  out.append(dict(seed=v['seed'],arm=v['arm'],pool=pool,queries=total,first_byte=sum(z['first_byte'] for z in f)/total,oracle_target_length_payload_prefix_upper_bound=sum(z['oracle_length_prefix'] for z in f)/total,native_EOS_positive_at_true_target_length=sum(z['EOS_at_true_length'] for z in f)/total,first_EOS_margin_mean=sum(z['first_EOS_margin'] for z in f)/total,most_common_outputs_hex=c.most_common(3)))
(P/'results/SAVED_OUTPUT_DIAGNOSTICS.json').write_text(json.dumps(dict(status='POSTHOC_LOCALIZATION_ONLY',note='True target length is used offline only for this generous payload-prefix upper bound; no own-output score uses it. Margins after incorrect prefixes do not measure teacher-forced EOS recognition. No learner queries or feedback are rerun.',rows=out),indent=2)+'\n')
