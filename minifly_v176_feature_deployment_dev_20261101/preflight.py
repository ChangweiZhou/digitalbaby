"""V176 authenticated corpus and label-blind context manifest. No model fitting."""
import hashlib, importlib.util, json, platform, shutil, time
from pathlib import Path
H=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def save(n,v):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def run():
 old=H/'history/preflight_001'
 if not old.exists():
  old.mkdir(parents=True)
  for n in ('STATUS.json','FINAL_AUDIT.json','REPORT.md','REVISED_DESIGN.md','CYCLE_0_AUDIT.json','ASSET_INVENTORY.json','AUTHORIZATION.json'):
   if (H/n).exists():shutil.copy2(H/n,old/n)
 save('CONTINUATION_AUTHORIZATION.json',dict(request='Do the 3 cycle exercise.',epoch=time.time(),asset_recovery_authorized=True,previous_stop_preserved=True,scientific_retry=False,prior_training_records=0))
 spec=json.loads((H.parent/'minifly_v176_feature_deployment_design_20261101/SPEC.json').read_text())
 path=H/'assets/source/minifly_v148_pretrain_probe_dev_20261025/corpus.py'
 loader=importlib.util.spec_from_file_location('v176_corpus',path);c=importlib.util.module_from_spec(loader);loader.loader.exec_module(c)
 corp={};meta={}
 for fn,d in spec['corpus_sha256'].items():
  raw=(H/'assets/corpora'/fn).read_bytes();assert sha(raw)==d
  rec=c.records(raw);cut=next((i for i,r in enumerate(rec) if r.startswith(b'End of Project Gutenberg')),len(rec))
  role='A' if fn in ('chesterton-thursday.txt','austen-emma.txt') else 'B'
  sp=c.split(rec[:cut],*spec['data_protocol'][role+'_split_KiB']);corp[fn]=sp
  meta[fn]=dict(sha256=d,footer_records_dropped=len(rec)-cut,offsets=sp['offsets'],splits={k:dict(records=len(sp[k]),targets=sum(len(r)+1 for r in sp[k]),sha256=sha(b''.join(len(r).to_bytes(4,'big')+r for r in sp[k]))) for k in ('train','calib','held')})
 manifest={};source_records=0;source_targets=0
 for pi,(an,bn) in enumerate(spec['pairs'],1):
  pair='P'+str(pi);a,b=corp[an],corp[bn]
  source_records+=len(a['train'])*9;source_targets+=sum(len(r)+1 for r in a['train'])*9
  exposure={r[:j] for k in ('train','calib') for r in a[k] for j in range(len(r)+1)}
  groups={}
  for k in ('train','calib','held'):
   for ri,r in enumerate(b[k]):
    for j,y in enumerate(r):groups.setdefault(r[:j],[]).append((k,ri,j,y))
  out={};diagnostics={}
  for k in ('train','held'):
   eligible=[p for p,g in groups.items() if {v[0] for v in g}=={k} and (k!='held' or p not in exposure)]
   chosen=sorted(eligible,key=lambda p:sha(f'V176|{pair}|{k}|{p.hex()}'.encode()))[:128]
   assert len(chosen)==128,(pair,k,len(eligible))
   out[k]=[dict(id=f'{pair}:{k}:{i}',cohort=k,prompt=p.hex(),answer=bytes([groups[p][0][3]]).hex(),source_record=groups[p][0][1],position=groups[p][0][2],occurrences=len(groups[p]),alternate_labels=sum(v[3]!=groups[p][0][3] for v in groups[p])) for i,p in enumerate(chosen)]
   diagnostics[k]=dict(eligible=len(eligible),selected=128,alternate_label_occurrences=sum(r['alternate_labels'] for r in out[k]))
  assert not set(r['prompt'] for r in out['train'])&set(r['prompt'] for r in out['held'])
  manifest[pair]=dict(A=an,B=bn,rows=out,diagnostics=diagnostics)
 save('local/DATA_MANIFEST.json',manifest)
 counts=dict(source_records=source_records,source_targets=source_targets,native_records=12288,native_targets=24576,probe_fit_targets=73728,probe_touch_cap=2048,technical_record_cap=32)
 counts['record_touch_ceiling']=source_records+12288+2048+32;counts['training_target_ceiling_excluding_technical']=source_targets+24576+73728
 assert counts['record_touch_ceiling']<=28000 and counts['training_target_ceiling_excluding_technical']<1200000
 save('DATA_AUDIT.json',dict(status='PASS',corpora=meta,contexts={p:m['diagnostics'] for p,m in manifest.items()},counts=counts,manifest_sha256=sha((H/'local/DATA_MANIFEST.json').read_bytes()),no_training=True))
 save('STATUS.json',dict(identity=spec['identity'],status='C0_DATA_PASSED_IMPLEMENTATION_IN_PROGRESS',training_records=0,three_cycles_complete=False))
 print(json.dumps(counts))
if __name__=='__main__':run()
