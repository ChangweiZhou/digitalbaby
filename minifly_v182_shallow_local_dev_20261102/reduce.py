"""Read saved states/metrics only; reconstruct all decisions without learning."""
import json,hashlib,math,time,resource
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;OUT=HERE/'results';OLD=ROOT/'minifly_v181_mini_local_plasticity_dev_20261102/campaign_20261102'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 t=time.process_time();lock=read(HERE/'SOURCE_LOCK.json')
 for p,h in lock['files'].items():assert sha(ROOT/p)==h,p
 counts=read(OLD/'COUNT_CONTROLS.json')['worlds'];decisions={};parity={};totalr=totalt=cpcount=sitechecks=0
 generations={};panels={}
 for w in ('M1','M2'):
  panels[w]={};generations[w]={};parity[w]={}
  for arm in ('NORMAL','SHALLOW'):
   job=read(OUT/f'{w}_{arm}_METRICS.json');panels[w][arm]=job;r=read(OUT/f'{w}_{arm}_RECEIPT.json');totalr+=r['records'];totalt+=r['targets'];assert r['records']==560
   stages=['birth','A1','A2','B1','B2','B3','EDIT1','EDIT2','NOEDIT1','NOEDIT2'];assert set(job)==set(stages) and set(r['checkpoints'])==set(stages)
   for stage in stages:
    cp=r['checkpoints'][stage];f=OUT/'checkpoints'/f'{w}_{arm}_{stage}.pt';assert sha(f)==cp['checkpoint_sha256'] and f.stat().st_size==cp['checkpoint_bytes']<1048576;cpcount+=1
    sf=OUT/'sites'/f'{w}_{arm}_{stage}.npz';assert sha(sf)==cp['sites_sha256'];z=np.load(sf)
    for name,m in job[stage].items():
     if name.startswith('_'):continue
     bits=z[name+'__bits'];y=z[name+'__y'];assert len(bits)==m['targets'];assert abs(float(bits.mean())-m['bits'])<1e-10
     assert abs(float(bits[y==256].mean())-m['EOS_NLL'])<1e-10
     assert sum(y==256)==len(m['record_target_counts']);off=np.cumsum([0]+m['record_target_counts']);assert all(y[i-1]==256 for i in off[1:]);sums=[float(bits[a:b].sum()) for a,b in zip(off[:-1],off[1:])];assert np.allclose(sums,m['record_bit_sums'],rtol=0,atol=1e-9);sitechecks+=len(bits)
     if name=='EDIT_held':
      changed=y!=z['A_held__y'];same=z[name+'__same_KC'];masks={'SAME_KC':~changed&same,'ALTERED_KC':~changed&~same,'CHANGED':changed}
      for k,mask in masks.items():assert int(mask.sum())==m['partitions'][k]['n'] and abs(float(bits[mask].mean())-m['partitions'][k]['bits'])<1e-10
   assert r['checkpoints']['EDIT2']['clock']==r['checkpoints']['NOEDIT2']['clock']
   generations[w][arm]={stage:{'n':len(job[stage]['_generation']),'own_EOS':sum(v['own_EOS'] for v in job[stage]['_generation']),'truncated':sum(v['truncated'] for v in job[stage]['_generation'])} for stage in ('A2','B3','EDIT2')}
  n=panels[w]['NORMAL'];s=panels[w]['SHALLOW'];gain=n['A2']['A_held']['bits']-s['A2']['A_held']['bits']
  components=lambda x:[x['B3']['B_held']['bits'],x['EDIT2']['B_held']['bits'],x['EDIT2']['EDIT_held']['partitions']['SAME_KC']['bits'],x['EDIT2']['EDIT_held']['partitions']['CHANGED']['bits']]
  nc=components(n);sc=components(s);eos=[s[st][p]['EOS_NLL']-n[st][p]['EOS_NLL'] for st in ('A2','B3','EDIT2') for p in ('A_held','B_held')]
  fb=s['NOEDIT2']['EDIT_held']['partitions']['CHANGED']['bits']-s['EDIT2']['EDIT_held']['partitions']['CHANGED']['bits']
  checks={'acquisition_gain':gain>=.05,'competence':s['A2']['A_held']['bits']<=counts[w]['A_unigram']['bits']-.05,'lifetime_guards':all(a<=b+.1 for a,b in zip(sc,nc)),'feedback_gain':fb>=.1,'EOS_guards':max(eos)<=.25}
  decisions[w]={'acquisition_gain_bits':gain,'A2_bits':{a:panels[w][a]['A2']['A_held']['bits'] for a in ('NORMAL','SHALLOW')},'unigram_bits':counts[w]['A_unigram']['bits'],'components_NORMAL':nc,'components_SHALLOW':sc,'composite_NORMAL':sum(nc)/4,'composite_SHALLOW':sum(sc)/4,'feedback_gain_bits':fb,'EOS_regressions':eos,'checks':checks}
  old=read(OLD/f'{w}_RESULTS_COMPACT.json')['panels']['NORMAL'];errors=[abs(n[st][p]['bits']-old[st][p]['bits']) for st in stages for p in old[st]];parity[w]={'max_bits_difference':max(errors),'within_1e4':max(errors)<=1e-4}
 assert totalr==2240 and totalt==113574 and cpcount==40
 overall=all(all(x['checks'].values()) for x in decisions.values());relative=all(all(v for k,v in x['checks'].items() if k!='competence') for x in decisions.values())
 verdict='SHALLOW_DEV_GAIN' if overall else 'SHALLOW_RELATIVE_GAIN_BASELINE_GAP' if relative else 'SHALLOW_HYPOTHESIS_NOT_SUPPORTED'
 result={'verdict':verdict,'worlds':decisions,'parent_scalar_parity':parity,'generation':generations,'scientific_records':totalr,'scientific_targets':totalt,'checkpoints':cpcount,'saved_site_values_checked':sitechecks,'reducer_CPU_seconds':time.process_time()-t,'cross_family_review':'PENDING','formal_checker':'UNAVAILABLE','adoption':False,'E3_claim':False};save(OUT/'DECISION.json',result)
 lines=['# V182 complete-life shallow-architecture development result','',f'**{verdict}**','', '| World | NORMAL A2 | SHALLOW A2 | Acquisition gain | NORMAL composite | SHALLOW composite |','|---|---:|---:|---:|---:|---:|']
 for w,d in decisions.items():lines.append(f"| {w} | {d['A2_bits']['NORMAL']:.6f} | {d['A2_bits']['SHALLOW']:.6f} | {d['acquisition_gain_bits']:.6f} | {d['composite_NORMAL']:.6f} | {d['composite_SHALLOW']:.6f} |")
 lines+=['', 'All four paired lives completed A→B→corrected-A and matched NOEDIT forks. No checkpoint selection, dose change, warmup or gate rescue occurred. The candidate removes T and moves the existing pre-output dropout onto H; the result concerns that whole declared architecture package.', '', f'Physical scientific learning: {totalr:,} full-record presentations / {totalt:,} byte/EOS targets. Forty full mutable checkpoints and sufficient site panels were saved and hash-audited. Technical updates and resource costs are separately recorded in TECHNICAL.json and RUN_COMPLETE.json.', '', 'M1/M2 are opened development worlds. No deterministic E1, semantic correction, arithmetic E3, sealed confirmation, model adoption or cross-family certification is claimed. Existing V181 historical design/source-manifest discrepancies are documented separately; current source closure is fully locked.', '', 'NORMAL published-scalar parity: '+json.dumps(parity), '', 'Full frozen gate vector and signed components: DECISION.json. Genuine own EOS/truncation is reported separately; arbitrary literary exactness is not a competence gate.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
