import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
V190=HERE.parent.parent/'minifly_v190_cued_integrated_reliability_design_20261103/campaign_v190r1'
raw=HERE.joinpath('relations_instrumented.log').read_text().splitlines()
round4=[json.loads(line.split(' ',1)[1]) for line in raw if line.startswith('PAIR_METRICS ')]
assert [r['seed'] for r in round4]==list(range(6))
v190=[json.loads((V190/f'{i}_RECEIPT.json').read_text())['stages']['B'] for i in range(3)]
summary={}
for j,name in enumerate(('comparison','parity')):
    entries=[row for r in round4 for row in r[name]]
    assert len(entries)==144
    byte=sum(y==out for _,_,y,out,_ in entries)
    with_newline=sum(y==out and stop==10 for _,_,y,out,stop in entries)
    new=[v['relations']['PREQ'][j] for v in v190]
    summary[name]=dict(round4={'n':len(entries),'byte_correct':byte,
        'byte_plus_predicted_newline':with_newline,'byte_plus_native_eos':None},
        v190={'n':sum(r['n'] for r in new),'byte_correct':sum(r['byte'] for r in new),
        'byte_plus_native_eos':sum(r['exact'] for r in new)})
facts={}
for name,form in [('new_template',2),('question',3)]:
    rows=[v['facts'][f'{form}_old'] for v in v190]
    facts[name]={'n':sum(r['n'] for r in rows),'byte_correct':sum(r['byte'] for r in rows),
                 'byte_plus_native_eos':sum(r['exact'] for r in rows)}
report={'relations':summary,'v190_facts':facts,'round4_familiarity':{
    p.stem:json.loads(p.read_text()) for p in sorted(HERE.glob('familiarity_*.json')) if p.stat().st_size}}
(HERE/'COMPARISON.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
