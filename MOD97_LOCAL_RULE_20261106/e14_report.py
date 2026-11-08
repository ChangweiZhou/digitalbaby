"""Read-only E14 live scorecard; fresh and standing seeds kept separate."""
import json,statistics
from pathlib import Path
import e14_campaign as C
R=C.R;O=C.O;N=('add97','sub97','max97','min97','add73','sub73','max73','min73');K=('+-','-+','><','<>')
def load(p):return json.loads(p.read_text())
def avg(sc,keys=N):return sum(100*sc[k]['exact']/sc[k]['n'] for k in keys)/len(keys)
def comp(sc,key='final'):return sum(100*sc[k][key]/sc[k]['n'] for k in K)/4

def e8(key,seed):
 arm='D' if key in ('A_D','C_BP') else 'C'
 d=load(O/f'e14_{key}_e8_life_{arm}_{seed}.json')
 assert d['stages'][-1]['task']=='compose'
 pre=d['stages'][-2]['scores'];end=d['stages'][-1]['scores']
 return pre,end,comp(end['compose'])
def e7(key,seed,arm='SCR',batch=64):
 tag=f'_b{batch}' if batch!=64 else ''
 d=load(O/f'e14_{key}_e7_life_{arm}_h512_d16384{tag}_{seed}.json')
 return d['curve'][-1]['compose']
def main():
 s=load(O/'E14_STATUS.json') if (O/'E14_STATUS.json').exists() else dict(status='prepared',completed=[])
 lines=['# E14 decay confirmation','',f'Status: **{s["status"]}**; completed subprocesses {len(s.get("completed",[]))}/46 (6births,39lives,1extra default profile).','', 'E14_LOCK.json/E14_PREDICTIONS.md are prospective. Three technical cycles precede launch; full baseline and nine logged-life parity checks are required. Fresh-seed confirmation is not an independent-code replication.','', '## A: fresh E8 seeds','','|Seed|Arm|Stage8 mean|Final mean|Held composition|','|---|---|---:|---:|---:|']
 detail=[];verdict=[]
 for seed in C.FRESH:
  for key in ('A_C','A_D'):
   try:a,b,c=e8(key,seed)
   except (FileNotFoundError,AssertionError,json.JSONDecodeError):lines.append(f'|{seed}|{key}|Pending|Pending|Pending|');continue
   lines.append(f'|{seed}|{key}|{avg(a):.2f}|{avg(b):.2f}|{c:.2f}|')
   if key=='A_C':verdict.append(f'A E8 seed{seed}: retention>=90 and composition>=75: {avg(b)>=90 and c>=75}.')
   detail += [f'\n### Fresh {key} seed{seed}','|Operation|Stage8|Final|','|---|---:|---:|']+[f'|{k}|{100*a[k]["exact"]/a[k]["n"]:.2f}|{100*b[k]["exact"]/b[k]["n"]:.2f}|' for k in a]+[str(b['compose'])]
 lines += ['','## A: fresh E7 seeds','','|Seed|Arm|Held final mean|Full scratchpad mean|Within final|Signed BP-minus-local gap|','|---|---|---:|---:|---:|---:|']
 for seed in C.FRESH:
  for key,arm,batch in [('A_batch256','SCR',256),('A_decay025','SCR',64),('A_BP','SCRbp',64)]:
   try:d=e7(key,seed,arm,batch)
   except (FileNotFoundError,json.JSONDecodeError):lines.append(f'|{seed}|{key}|Pending|Pending|Pending|Pending|');continue
   try:bp=comp(e7('A_BP',seed,'SCRbp'));gap=bp-comp(d)
   except (FileNotFoundError,json.JSONDecodeError):gap=None
   gs=f'{gap:+.2f}' if gap is not None else 'Pending'
   lines.append(f'|{seed}|{key}|{comp(d):.2f}|{comp(d,"exact"):.2f}|{100*d["within"]["final"]/d["within"]["n"]:.2f}|{gs}|')
   if key!='A_BP':verdict.append(f'A E7 {key} seed{seed}: held>=80 {comp(d)>=80}; absolute BP gap<=10 {abs(gap)<=10 if gap is not None else "Pending"}.')
   detail += [f'\n### Fresh {key} seed{seed}',str(d)]
 lines += ['','## B: standing-seed E8 curve','','|Seed|Decay|Stage8 mean|Final mean|Held composition|Origin|','|---|---:|---:|---:|---:|---|']
 for seed in C.STANDING:
  metrics={}
  for wd,key in [(0.1,'B_E8_wd010'),(.25,None),(.5,'B_E8_wd050'),(1.,None)]:
   try:
    if key:a,b,c=e8(key,seed)
    else:
     p=O/(f'e13_A1_e8_life_C_{seed}.json' if wd==.25 else f'e8_life_C_{seed}.json');d=load(p);a=d['stages'][-2]['scores'];b=d['stages'][-1]['scores'];c=comp(b['compose'])
   except (FileNotFoundError,AssertionError,json.JSONDecodeError):lines.append(f'|{seed}|{wd}|Pending|Pending|Pending|new|');continue
   v=(avg(a),avg(b),c);metrics[wd]=v;lines.append(f'|{seed}|{wd}|{v[0]:.2f}|{v[1]:.2f}|{v[2]:.2f}|{"new" if key else "reused"}|')
   if key:detail += [f'\n### Standing {key} seed{seed}',str(a),str(b)]
  if .1 in metrics and .25 in metrics:verdict.append(f'B E8 seed{seed}: plateau endpoints within3 {all(abs(x-y)<=3 for x,y in zip(metrics[.1],metrics[.25]))}.')
  if .5 in metrics and .25 in metrics:verdict.append(f'B E8 seed{seed}: wd0.5 composition >=5 below0.25 {metrics[.25][2]-metrics[.5][2]>=5}.')
 lines += ['','## B: standing-seed E7 curve','','|Seed|Decay|Held final mean|Within final|Full scratchpad mean|Origin|','|---|---:|---:|---:|---:|---|']
 for seed in C.STANDING:
  metrics={}
  for wd,tag in ((.1,'010'),(.25,'025'),(.5,'050'),(1.,None)):
   try:d=e7(f'B_E7_wd{tag}',seed) if tag else load(O/f'e7_life_SCR_h512_d16384_{seed}.json')['curve'][-1]['compose']
   except (FileNotFoundError,json.JSONDecodeError):lines.append(f'|{seed}|{wd}|Pending|Pending|Pending|new|');continue
   v=(comp(d),100*d['within']['final']/d['within']['n'],comp(d,'exact'));metrics[wd]=v;lines.append(f'|{seed}|{wd}|{v[0]:.2f}|{v[1]:.2f}|{v[2]:.2f}|{"new" if tag else "reused"}|');detail += [f'\n### Standing E7 wd{wd} seed{seed}',str(d)]
  if .1 in metrics and .25 in metrics:verdict.append(f'B E7 seed{seed}: plateau endpoints within3 {all(abs(x-y)<=3 for x,y in zip(metrics[.1],metrics[.25]))}.')
 lines += ['','## C: standing-seed diagnostic summaries','','Raw-gradient cosine and actual Adam descent cosine are different. Weight tensors below; bias rows and full trajectories are in diagnostics.jsonl. Output raw cosine1 is expected by construction. True gradient is on the actual batch, including replay pseudo-labels. Cross-arm means are descriptive.','','|Seed|Arm|Layer|Logged points|Mean raw cosine|Mean Adam-descent cosine|Mean Adam-step/weight|First weight norm|Last weight norm|','|---|---|---|---:|---:|---:|---:|---:|---:|']
 for seed in C.STANDING:
  for key in ('C_wd1','C_wd025','C_BP'):
   arm='D' if key=='C_BP' else 'C';p=O/f'e14_{key}_e8_life_{arm}_{seed}.diagnostics.jsonl'
   if not p.exists():continue
   logs=[]
   for line in p.read_text().splitlines():
    try:logs.append(json.loads(line))
    except json.JSONDecodeError:continue # A live writer may not have flushed its last line yet.
   for layer in ('l1','l2','l3'):
    values=[x['parameters'][layer+'.weight'] for x in logs]
    if not values:continue
    def mean_field(name):
     v=[x[name] for x in values if x[name] is not None];return statistics.mean(v) if v else None
    raw=mean_field('raw_gradient_cosine_to_true');adam=mean_field('adam_descent_cosine_to_true');ratio=mean_field('adam_step_over_weight_norm')
    show=lambda x:'null' if x is None else f'{x:.6f}'
    lines.append(f'|{seed}|{key}|{layer}|{len(values)}|{show(raw)}|{show(adam)}|{ratio:.6f}|{values[0]["weight_norm_before"]:.6f}|{values[-1]["weight_norm_after"]:.6f}|')
 lines += ['','## Prediction verdicts','']+verdict
 lines += ['','## Compute and limits','',f'Completed subprocess wall sum: {sum(j.get("wall_seconds",0) for j in s.get("completed",[])):.2f}s; parallel elapsed is separate. Actual complete CPU/diagnostic costs pending final recount. PeakRSS:NOT_MEASURED. No tuning/seed substitution on scientific negatives. Fresh births and extra observer backward passes are compute, not free resources. Log hooks are default-off and their logged/unlogged parity is verified per completed logged job.']
 (R/'E14_REPORT.md').write_text('\n'.join(lines+detail)+'\n')
if __name__=='__main__':main()
