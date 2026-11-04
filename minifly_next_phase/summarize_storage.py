import json
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t

root=Path(__file__).resolve().parent
plan=json.loads((root/'STORAGE_PLAN.json').read_text())
summary=[]
for load,units,k in plan['cells']:
    paired=[]
    for seed in plan['seeds']:
        rows=[json.loads((root/f'{seed}-{load}-{units}-{k}-{dtype}.json').read_text())
              for dtype in plan['dtypes']]
        assert rows[0]['stream_hash']==rows[1]['stream_hash']
        paired.append(rows)
    def values(r):
        revision=next(row['revision_unseen'] for name,row in r['metrics'].items()
                      if name.endswith('_8') and 'revision_unseen' in row)
        return dict(old=r['metrics']['after_clutter']['old'],
                    unseen=r['metrics']['after_clutter']['unseen'], revision=revision,
                    one_shot=r['one_shot'])
    result=dict(load=load,units=units,active=k,formats={},contrasts={})
    for i,dtype in enumerate(plan['dtypes']):
        rows=[p[i] for p in paired]
        result['formats'][dtype]=dict(
            **{name:mean(values(r)[name] for r in rows) for name in values(rows[0])},
            readout_bytes=rows[0]['costs']['store_bytes'],
            worker_lifetime_peak_rss_bytes=max(r['peak_rss_bytes'] for r in rows),
            mean_seconds_per_byte=mean(r['seconds']/r['byte_events'] for r in rows))
    for name in values(paired[0][0]):
        d=[values(b)[name]-values(a)[name] for a,b in paired]
        m=mean(d)
        h=float(t.ppf(.975,len(d)-1))*stdev(d)/len(d)**.5
        result['contrasts'][name]=[m,m-h,m+h]
    summary.append(result)
r=summary[-1]
g=plan['gate']
gate=(r['formats']['float16']['old']>=g['retention']
      and all(r['contrasts'][k][0]>=-g['max_drop'] for k in ('old','unseen','revision'))
      and r['formats']['float16']['readout_bytes']<r['formats']['float64']['readout_bytes'])
(root/'STORAGE_RESULTS.json').write_text(json.dumps(dict(gate=gate,cells=summary),indent=2)+'\n')
