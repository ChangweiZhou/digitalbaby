"""Posthoc frequency explanation, evaluator only. No output replacement or fitting in fly."""
import json,collections
from pathlib import Path
P=Path(__file__).resolve().parent;out=[]
for seed in (202610236101,202610236102):
 d=json.loads((P/f'manifests/{seed}.json').read_text())
 for arm in ('LEARN','FROZEN','NO_OLD','SH_OLD'):
  counts=collections.Counter();EOS=0;tokens=0
  for phase in ('old','new'):
   if arm=='NO_OLD' and phase=='old':continue
   for entry in d[phase+'_order']:
    r=d['shuffled' if arm=='SH_OLD' and phase=='old' else phase][entry['index']]
    p=bytes(r['prompt'])
    if entry['reverse']:
     a,b=p[:-1].split(b'+');p=b+b'+'+a+b'='
    counts.update(p+bytes(r['answer']));EOS+=1;tokens+=len(p)+len(r['answer'])+1
  payload=sum(counts.values());bitpositive=[sum(n for b,n in counts.items() if b&(1<<i))/payload for i in range(8)];mode=sum(1<<i for i,v in enumerate(bitpositive) if v>.5)
  path=P/f'results/probes/{seed}_{arm}_FINAL.json'
  if not path.exists():continue
  rows=json.loads(path.read_text())['rows'];pred=[mode]*16
  out.append(dict(seed=seed,arm=arm,eligible_native_feedback_payloads=payload,eligible_EOS_on_targets=EOS,all_EOS_targets=tokens,EOS_on_fraction=EOS/tokens,payload_bit_positive_fractions=bitpositive,independent_bit_mode=mode,bit_mode_hex=bytes([mode]).hex(),predicted_generation_hex=bytes(pred).hex(),saved_final_outputs_matching_unconditional_bit_mode_fraction=sum(r['output']==pred and r['truncated'] for r in rows)/len(rows)))
(P/'results/MARGINAL_DIAGNOSTICS.json').write_text(json.dumps(dict(status='POSTHOC_EXPLANATORY_COUNTERMODEL_NOT_PRIMARY',note='Fits bit/EOS priors from observed tokens with enabled native writes, including prompts, using no held labels or learned model parameters. It ignores order, physiology and conditional state, so matching does not prove the exact causal algorithm. It never supplies fly outputs or changes the endpoint.',rows=out),indent=2)+'\n')
