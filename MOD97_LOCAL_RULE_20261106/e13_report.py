"""Read-only scorecard generation for E13; no selection or tuning."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'exp45';S=(73129,93683,116089);N=('add97','sub97','max97','min97','add73','sub73','max73','min73')
def load(p):return json.loads(p.read_text())
def avg(sc,keys=N):return sum(100*sc[k]['exact']/sc[k]['n'] for k in keys)/len(keys)
def life(p):
 d=load(p);assert d['stages'][-1]['task']=='compose';a=d['stages'][-2]['scores'];b=d['stages'][-1]['scores'];comp=b['compose'];return a,b,sum(100*comp[k]['final']/comp[k]['n'] for k in ('+-','-+','><','<>'))/4

def main():
 status=load(O/'E13_STATUS.json') if (O/'E13_STATUS.json').exists() else {'status':'prepared','completed':[]}
 lines=['# E13 batch-package experiment','',f'A–C status: **{status["status"]}**, completed subprocesses {len(status.get("completed",[]))}/20 (18 candidates plus2 full profiles).','', 'Three technical cycles passed; see E13_CYCLES.json and E13_CONTROLLER_REHEARSAL.json. Source and command lock: E13_LOCK.json. No independent implementation confirmation or formal checker seal is claimed.','', '## Part A','','|Arm|Seed|Stage8 mean|Final mean|Held composition|','|---|---:|---:|---:|---:|']
 details=[];verdict=[]
 for seed in S:
  refs={64:life(O/f'e8_life_C_{seed}.json'),256:life(O/f'e12_B_e8_life_C_b256_{seed}.json')}
  for arm,b in [('A1',64),('A2',256),('A3',256),('A4',1024)]:
   tag='' if b==64 else f'_b{b}';p=O/f'e13_{arm}_e8_life_C{tag}_{seed}.json'
   try:a,z,c=life(p)
   except (FileNotFoundError,AssertionError,json.JSONDecodeError):lines.append(f'|{arm}|{seed}|Pending|Pending|Pending|');continue
   lines.append(f'|{arm}|{seed}|{avg(a):.2f}|{avg(z):.2f}|{c:.2f}|')
   if arm in ('A1','A2'):verdict.append(f'{arm} seed{seed}: final retention{avg(z):.2f}; predicted decay-threshold {avg(z)>=90 if arm=="A1" else avg(z)<=80}. Delta vs batch64{avg(z)-avg(refs[64][1]):+.2f}, vs batch256{avg(z)-avg(refs[256][1]):+.2f}.')
   if arm=='A4':verdict.append(f'A4 seed{seed}: within3 retention and composition no lower: {abs(avg(z)-avg(refs[256][1]))<=3 and c>=refs[256][2]}.')
   details += [f'\n### {arm} seed{seed}','|Operation|Stage8|Final|','|---|---:|---:|']+[f'|{k}|{100*a[k]["exact"]/a[k]["n"]:.2f}|{100*z[k]["exact"]/z[k]["n"]:.2f}|' for k in a]+[str(z['compose'])]
 lines += ['','## Part B','','|Seed|Held final mean|Full scratchpad mean|Within final|Last interval held delta|','|---:|---:|---:|---:|---:|']
 for seed in S:
  p=O/f'e13_B_e7_life_SCR_h512_d16384_b256_{seed}.json'
  if not p.exists():lines.append(f'|{seed}|Pending|Pending|Pending|Pending|');continue
  d=load(p);cur=d['curve'];last=cur[-1]['compose'];keys=('+-','-+','><','<>');mean=lambda c,k:sum(100*c[x][k]/c[x]['n'] for x in keys)/4
  final=mean(last,'final');delta=final-mean(cur[-2]['compose'],'final')
  lines.append(f'|{seed}|{final:.2f}|{mean(last,"exact"):.2f}|{100*last["within"]["final"]/last["within"]["n"]:.2f}|{delta:+.2f}|');verdict.append(f'B seed{seed}: held>=80 {final>=80}.')
  details += [f'\n### B seed{seed}',str(last)]
 lines+=['','## Part C','','|Seed|Batch64 final old mean|Batch256 final old mean|Delta|Held composition|','|---:|---:|---:|---:|---:|']
 for seed in S:
  p=O/f'e13_C_e8_life_D_b256_{seed}.json'
  try:a,z,c=life(p)
  except (FileNotFoundError,AssertionError,json.JSONDecodeError):lines.append(f'|{seed}|Pending|Pending|Pending|Pending|');continue
  ref=life(O/f'e8_life_D_{seed}.json');delta=avg(z)-avg(ref[1]);lines.append(f'|{seed}|{avg(ref[1]):.2f}|{avg(z):.2f}|{delta:+.2f}|{c:.2f}|');verdict.append(f'C seed{seed}: final abs delta<2 {abs(delta)<2}; stage8 delta{avg(a)-avg(ref[0]):+.2f}.')
  details += [f'\n### C seed{seed}','|Operation|Stage8|Final|','|---|---:|---:|']+[f'|{k}|{100*a[k]["exact"]/a[k]["n"]:.2f}|{100*z[k]["exact"]/z[k]["n"]:.2f}|' for k in a]+[str(z['compose'])]
 lines+=['','## Part D','','Steps1–3 delivered in ../output/minifly-connectome-navigation/PI_FOLLOWUP_REPORT.md. All three native clamped-heading return medians are0.7099–0.7170, above0.30; prescribed scientific STOP. No refit, no new cell types/panels. Initial invalid nominal-phase clamp preserved and corrected by fixed-observer inversion; accepted clamp error below1e-6degree. This does not establish a working integrator.','', '## Prediction checks','']+verdict
 lines+=['','## Costs and limitations','','Live costs in E13_JOB_*.json; completed summed process wall seconds '+str(sum(j.get('wall_seconds',0) for j in status.get('completed',[])))+'. Final CPU/checkpoint totals and verification pending until campaign end. Existing E9/E12 and E8 BP controls are reused; the historical E7 BP ceiling is not identically matched in width/dose. Batch interventions change cumulative decay, update count and optimization. PartD diagnostics are read-only and its prescribed stop is a negative result, not a pass.']
 (R/'E13_REPORT.md').write_text('\n'.join(lines+details)+'\n')
if __name__=='__main__':main()
