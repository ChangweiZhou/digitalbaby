"""Reconstruct payload and complete exactness directly from saved response tensors."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=0
for f in (P/'traces').glob('*_outputs.json'):
 d=json.loads(f.read_text());tag=f.stem[:-8]
 with np.load(P/'traces'/f'{tag}.npz') as z:
  for row in d['results']:
   if row['lesion']!='NONE':continue
   i=d['pools'].index(row['pool']);vals=d['labels'][i];D=z[f'D{i}'].mean(-1)
   decoded=np.sum((D[:,:,:8]>0)*(1<<np.arange(8)),axis=-1)
   ix=[j for j,v in enumerate(vals) if v]
   payload=float(np.mean([decoded[j,:len(vals[j])].tolist()==vals[j] for j in ix])) if ix else None
   assert payload==row['metrics']['payload_exact']
   full=float(np.mean([row['metrics']['outputs'][j]['EOS'] and row['metrics']['outputs'][j]['bytes']==vals[j] for j in ix])) if ix else None
   assert full==row['metrics']['nonempty_exact'];checks+=1
for path,h in json.loads((P/'RUNTIME_MODULE_CLOSURE.json').read_text()).items():assert sha(Path(path))==h,path
for path,h in json.loads((P/'PRE_VALUE_ASSET_LOCK.json').read_text()).items():assert sha(P/path)==h,path
arithmetic=R/'PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json'
assert sha(arithmetic)=='79c74951d98b33c38e5edae23617e5a36a8573c39857d2526eeb72bf8b8144d3'
(P/'FINAL_AUDIT.json').write_text(json.dumps({'status':'ACCEPTED_ARTIFACT_METRICS','checked_metric_rows':checks,'payload_reconstructed_from_native_tensors':True,'all_pre_value_runtime_and_asset_hashes_unchanged':True,'arithmetic_pointer_unchanged':True,'native_reruns':0,'head_fits':0,'review_scope':'separate reducer process, not independent human review'},indent=2)+'\n')
