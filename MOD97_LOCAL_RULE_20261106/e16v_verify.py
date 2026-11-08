"""Receipt/source verification only. No torch import, model execution or training."""
import json,csv,hashlib,statistics,re
from pathlib import Path
from random import Random
from collections import Counter
from fractions import Fraction
R=Path(__file__).resolve().parent;S=R/'E16V_SOURCE_SNAPSHOT';D=S/'MOD97_LOCAL_RULE_20261106';O=D/'exp45'
def load(p):return json.loads(p.read_text())
def pct(v,k='exact'):return 100*v[k]/v['n']
def ops(st,mod=97,k='12',silence=False):
 sc=st['scores']['silenced_k12'] if silence else st['scores']
 return statistics.mean(pct(sc[f'{op}{mod}'] if silence else sc[f'{op}{mod}'][k]) for op in ('add','sub','max','min'))
def comp(st):return statistics.mean(pct(st['scores']['compose'][k],'final') for k in ('+-','-+','><','<>'))
def tab(h,rows):return ['|'+'|'.join(map(str,h))+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rows]
def apply(o,a,b):return {'+':(a+b)%97,'-':(a-b)%97,'>':max(a,b),'<':min(a,b)}[o]
def triples(pair,seed):
 rng=Random(seed*31+ord(pair[0])*7+ord(pair[1]));a=[];seen=set()
 while len(a)<600:
  t=tuple(rng.randrange(97) for _ in range(3))
  if t not in seen:seen.add(t);a.append(t)
 return a
def main():
 receipts={p.stem.removeprefix('e16_life_'):load(p) for p in sorted(O.glob('e16_life_*.json'))}
 seeds=(73129,93683,116089);wanted=[f'{a}_{s}' for a in ['T','D','T_oracle','T_balanced','T_all_balanced','T_share','M'] for s in seeds]
 assert set(wanted)<=set(receipts)
 lines=['# E16-V — independent receipt and source verification','',f'Pinned source commit: {(S/"COMMIT.txt").read_text().strip()}. No lives rerun; no network imported or executed. Expected exposure is distinguished from realized exposure, which is not logged.','']
 metrics=[]
 for key in wanted:
  d=receipts[key];seed=d['seed'];a=d['stages'];bp=receipts[f'D_{seed}']['stages'][-1]
  metrics.append(dict(arm=key,seed=seed,F1_block2=ops(a[2]),F1_block4=ops(a[4]),F1_block6=ops(a[6]),F3_block6=ops(a[6],73),comp_block4=comp(a[4]),comp_block6=comp(a[6]),F1_loss_2_to6=ops(a[2])-ops(a[6]),comp_loss_4_to6=comp(a[4])-comp(a[6]),F1_ratio=ops(a[6])/ops(bp),F3_ratio=ops(a[6],73)/ops(bp,73),comp_ratio=comp(a[6])/comp(bp),within_final=pct(a[6]['scores']['compose']['within'],'final')))
 lines+=['## 1 Report numbers','']+tab(['Arm/seed','F1 b2→b4→b6','Comp b4→b6','F3 b6','F1/comp/F3 ratio'],[[x['arm'],f"{x['F1_block2']:.2f}→{x['F1_block4']:.2f}→{x['F1_block6']:.2f}",f"{x['comp_block4']:.2f}→{x['comp_block6']:.2f}",f"{x['F3_block6']:.2f}",f"{x['F1_ratio']:.3f}/{x['comp_ratio']:.3f}/{x['F3_ratio']:.3f}"] for x in metrics])+['','Verdict: canonical means and ratios recomputed for every seed. Full quoted-table/prose crosswalk remains to be completed; see pending scope below.','']
 ceilings=[]
 for seed in seeds:
  rec=receipts[f'T_{seed}']['stages'][0]['shortcut_ceilings']
  for pair in ('+-','-+','><','<>'):
   counts=Counter();hits=Counter()
   for a,b,c in triples(pair,seed+1):
    i=apply(pair[0],a,b);f=apply(pair[1],i,c);counts[f]+=1
    for name,pred in [('ignore_o1_use_a',apply(pair[1],a,c)),('ignore_o1_use_b',apply(pair[1],b,c)),('ignore_o2',i),('answer_is_c',c),('copy_intermediate',i)]:hits[name]+=pred==f
   for k in rec[pair]:assert abs(hits[k]/600-rec[pair][k])<1e-14,(seed,pair,k)
   ceilings.append(dict(seed=seed,pair=pair,**{k:100*v/600 for k,v in hits.items()},modal_answer=max(counts.values())/6))
 lines+=['## 2 Independently rebuilt shortcut panels','', 'The intermediate-copy predictor is mathematically identical to the existing ignore_o2 predictor in this two-step task. Modal answer is the best constant on the evaluated panel, an upper bound rather than a trained predictor.','']+tab(['Seed','Pair','Ignore1/a','Ignore1/b','Ignore2','Copy c','Copy intermediate','Modal answer'],[[x['seed'],x['pair'],*[f"{x[k]:.2f}" for k in ['ignore_o1_use_a','ignore_o1_use_b','ignore_o2','answer_is_c','copy_intermediate','modal_answer']]] for x in ceilings])+['','Verdict: CONFIRMED, all four recorded ceilings match independent reconstruction exactly (fraction precision); two additional ceilings computed.','']
 exposure=[]
 for arm in ['uniform','balanced','all_balanced','share','oracle','D']:
  for block in range(1,7):
   fam=(block-1)//2;v=[Fraction(0)]*3
   if arm in ['oracle','D']:
    for i in range(fam+1):v[i]=Fraction(1,fam+1)
   elif block==1 or (arm=='share' and block==2):v[fam]=1
   else:
    v[fam]=Fraction(1,2)
    previous=[(i-1)//2 for i in range(1,block)]
    if arm=='uniform':
     for x in previous:v[x]+=Fraction(1,2*len(previous))
    else:
     fs=sorted(set(previous)-({fam} if arm=='share' else set()))
     for x in fs:v[x]+=Fraction(1,2*len(fs))
   assert sum(v)==1
   exposure.append(dict(arm=arm,block=block,F1=str(v[0]),F2=str(v[1]),F3=str(v[2])))
 lines+=['## 3 Exposure accounting','', 'Fractions below are EXPECTED whole-batch exposure from source branches. Uniform/stratum draws are random; receipts have no realized draw counters. Buffer rows corroborate eligible family/entry counts, not realized frequencies.','']+tab(['Arm','Block','F1','F2','F3'],[[x[k] for k in ['arm','block','F1','F2','F3']] for x in exposure])+['','Verdict: DISCREPANCY in claims of doubled exposure. Share changes F1 in F2 from 1/2,1/3 to 1/2,1/2 and F2 in F3 from1/4,1/5 to1/4,1/4. The first loss-bearing block is unchanged. Realized exposure is NOT_MEASURED. Buffer-field crosscheck is pending.','']
 lines+=['## 4 Stream identity','']+tab(['Comparison','Source conditions','Remaining differences'],[['T_oracle vs D','Same entry_stream/joint_stream constructors, seeds and seen-family ordering','Update rule, decay0.25/0.3, Store presence'],['M vs T_uniform','Same Random(seed+9), current generators, buffer sample/draw/render order; replay labels recomputed from operands','M uses BP at0.25 and no Store; T uses sign trace and Store']])+['','Verdict: CONFIRMED at source-construction level, not actual-batch digest attestation. No per-run batch hashes or frozen runtime source are present in these receipts. Store affects scoring; confirm training helper RNG use separately before final verdict.','']
 cell=[]
 for key in wanted:
  st=receipts[key]['stages'][6]
  for op,v in st['scores']['silenced_k12'].items():cell.append(dict(arm=key,op=op,n=v['n'],exact=v['exact'],percent=pct(v)))
 lines+=['## 5 Silenced cell','']+tab(['Arm/seed','Largest silenced operation score','All operations≤10%'],[[key,f"{max(x['percent'] for x in cell if x['arm']==key):.2f}",max(x['percent'] for x in cell if x['arm']==key)<=10] for key in wanted])+['','Verdict: '+('CONFIRMED' if all(x['percent']<=10 for x in cell) else 'DISCREPANCY')+'; every per-operation score retained in E16V_CELL.csv. This is a joint network/state ablation, not a proof that window computations contribute nothing.','']
 paths=load(S/'REPOSITORY_PATHS.json');prov=[]
 for seed in seeds:
  name=f'MOD97_LOCAL_RULE_20261106/exp45/e11_dev_M_{seed}.pt';prov.append([name,'PRESENT' if name in paths else 'MISSING'])
 dry=[p for p in paths if p.startswith('MOD97_LOCAL_RULE_20261106/exp45/archive_e16_dry/')];prov.append(['archive_e16_dry/ files',len(dry)])
 for key in wanted:prov.append([key+' settings',json.dumps(receipts[key]['stages'][0].get('settings'))])
 lines+=['## 6 Provenance','']+tab(['Artifact/settings','Status'],prov)+['','Verdict: standing E11 memory-birth weights are absent from this pinned repository tree; dry records exist as listed. Receipt settings/source-at-run linkage and gaps require the remaining source review.','', '## Pending verification scope','', 'This is a WORKING report, not final signoff. Remaining: every SUMMARY_E16 table cell and prose quote crosswalk, buffer field crosscheck, helper RNG/source-version evidence, exact silenced composition tables, settings deviations, SHA manifest and final six verdicts. No experiment has been rerun.']
 (R/'E16V_REPORT.md').write_text('\n'.join(lines)+'\n')
 for name,rows in [('E16V_METRICS.csv',metrics),('E16V_SHORTCUTS.csv',ceilings),('E16V_EXPOSURES.csv',exposure),('E16V_CELL.csv',cell)]:
  with (R/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (R/'E16V_STATUS.json').write_text(json.dumps(dict(status='WORKING_NOT_FINAL',life_reruns=0,receipts=len(wanted),shortcut_checks=36,remaining=['summary/prose crosswalk','buffer corroboration','helper RNG','settings/provenance finalization']),indent=2)+'\n')
 print('Working audit generated; no lives or networks executed.')
if __name__=='__main__':main()
