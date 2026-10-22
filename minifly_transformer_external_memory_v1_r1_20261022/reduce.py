"""Artifact-only audit and reduction; never imports native engine or reruns learning."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
rows=[];total=0;cpu=0
for c in (1,2,3):
 d=load(P/'cycles'/f'CYCLE{c}_RESULT.json');plan=load(P/'cycles'/f'CYCLE{c}_PLAN.json')
 assert hashlib.sha256((P/'cycles'/f'CYCLE{c}_SOURCE.py').read_bytes()).hexdigest()==plan['runner_sha256']
 assert hashlib.sha256((P/'DESIGN.md').read_bytes()).hexdigest()==plan['design_sha256']
 assert d['seed']==plan['seed'];assert d['CPU_seconds']<240
 total+=d['records'];cpu+=d['CPU_seconds']
 if c==1:
  assert d['no_write_zero'] and d['readonly'] and d['codec_all257'];continue
 values=np.array(d.get('values',np.random.default_rng(d['seed']+90).integers(0,256,(16,4),dtype=np.uint8)),dtype=np.uint8)
 groups=[('W',d['curves'])] if c==2 else [(a['arm'],a['stages']) for a in d['arms']]
 for arm,stages in groups:
  for st in stages:
   r=st['score'];m=np.array(r['margins']);assert m.shape==(16,9,9) and np.isfinite(m).all()
   emitted=[]
   for row in m:
    seq=[];ended=False
    for z in row:
     if z[8]>0:ended=True;break
     seq.append(sum((1<<k) for k in range(8) if z[k]>0))
    emitted.append({'payload':seq,'eos':ended,'truncated':not ended})
   assert emitted==r['outputs']
   exact=float(np.mean([x['payload']==v.tolist() and x['eos'] for x,v in zip(emitted,values)]));assert exact==r['exact']
   bittruth=((values[:,:,None]>>np.arange(8))&1)
   bits=float(np.mean((m[:,:4,:8]>0)==bittruth));assert bits==r['bit_accuracy']
   payload=np.sum((m[:,:4,:8]>0)*(1<<np.arange(8)),axis=2)
   eosfirst=[next((i for i,z in enumerate(row) if z[8]>0),9) for row in m]
   rows.append(dict(cycle=c,arm=arm,checkpoint=st.get('stage',str(st.get('passes'))),exact=exact,bit_accuracy=bits,payload_all4_if_length_forced_diagnostic_only=float(np.mean(np.all(payload==values,axis=1))),early_EOS=float(np.mean(np.array(eosfirst)<4)),own_EOS_at4=float(np.mean(np.array(eosfirst)==4)),late_or_missing_EOS=float(np.mean(np.array(eosfirst)>4)),target_aligned_payload_margin=float(np.mean(m[:,:4,:8]*(2*bittruth.astype(float)-1))),EOS_slot4_margin_mean=float(m[:,4,8].mean()),shuffled_assignment_exact=float(np.mean([x['payload']==v.tolist() and x['eos'] for x,v in zip(emitted,np.roll(values,1,axis=0))])) ))
assert total==1842
summary=dict(status='THREE_TECHNICAL_CYCLES_COMPLETE_HYBRID_NOT_QUALIFIED',technical_record_applications=total,native_events=total*5,CPU_seconds=cpu,scientific_jobs=0,host_training_updates=0,evidence='E1 component diagnostics; no hybrid or arithmetic efficacy result',rows=rows)
(P/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
(P/'FINAL_AUDIT.json').write_text(json.dumps(dict(integrity='ACCEPTED_ARTIFACT_REDUCTION',independent_process_not_independent_reviewer=True,training_rerun=False,source_hashes_checked=True,outputs_and_metrics_recomputed=True,exact_storage_qualified=False,hybrid_qualified=False,formal_launch_ready=False,records=total),indent=2)+'\n')
print(json.dumps(summary,indent=2))
