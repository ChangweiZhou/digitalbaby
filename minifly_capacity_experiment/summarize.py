"""Summarize completed cells and paired contrasts from the frozen plan."""
import json
from pathlib import Path
from statistics import mean, stdev
from scipy.stats import t

root = Path(__file__).resolve().parent
plan = json.loads((root/'plan.json').read_text())
seeds = plan['seeds']
groups = {}
lines = ['# MiniFly sparse store capacity test', '',
         'Four paired seeds per cell. The recovered V40 SUPER arm uses the V39-derived stream,',
         'lag encoder and local error update. Only layer dimensions and active counts vary within',
         'a load. All 12 streams were audited for absence of alias-2 contexts in training windows.',
         'The alias retains the same taught key and label. This is E2 presentation transfer;',
         'a canonical key table could solve it too.', '',
         '| Facts | Units | Active | Seeds | Old after D | Unseen after D | Old after clutter | Unseen after clutter | One-shot unseen | Revision @8 unseen | Store MiB |',
         '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
def pct(value):
    return f'{100*value:.1f}%'
def metric(row, checkpoint, name):
    return row['metrics'][checkpoint][name]
for load, units, active in plan['cells']:
    cell = (load, units, active)
    files = [root/f'{seed}-{load}-{units}-{active}.json' for seed in seeds]
    rows = [json.loads(file.read_text()) for file in files if file.exists()]
    groups[cell] = rows
    if rows:
        measures = [pct(mean(metric(row, cp, name) for row in rows))
                    for cp in ('after_D', 'after_clutter') for name in ('old', 'unseen')]
        revision = [values['revision_unseen'] for row in rows
                    for cp, values in row['metrics'].items()
                    if cp.endswith('_8') and 'revision_unseen' in values]
        measures += [pct(mean(row['one_shot'] for row in rows)),
                     pct(mean(revision)) if revision else '—',
                     f"{rows[0]['costs']['store_bytes']/2**20:.0f}"]
        lines.append('| ' + ' | '.join(map(str, [*cell, len(rows), *measures])) + ' |')
lines += ['', '## Paired effects on retention after wave D', '',
          'Intervals are descriptive paired 95% t intervals across four seeds.']
for a, b in [((128,4096,205),(128,16384,205)), ((512,16384,205),(512,65536,205)),
             ((128,16384,205),(128,16384,410)), ((512,65536,205),(512,65536,410))]:
    left, right = groups[a], groups[b]
    if len(left) != len(seeds) or len(right) != len(seeds):
        continue
    assert all(x['stream_hash'] == y['stream_hash'] for x, y in zip(left, right))
    for name in ('old', 'unseen'):
        delta = [100*(metric(y,'after_D',name)-metric(x,'after_D',name)) for x,y in zip(left,right)]
        avg = mean(delta)
        half = float(t.ppf(.975,len(seeds)-1))*stdev(delta)/len(seeds)**.5
        lines.append(f'- {a} → {b}, {name}: {avg:+.1f} pp [{avg-half:+.1f}, {avg+half:+.1f}].')
lines += ['', 'The store uses the recovered float64 arithmetic. Code arrays are additional, and',
          'the encoder scans all units for top-k selection. These are workload-specific capacity',
          'measurements, with no byte-efficiency or V181-fit claim.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
