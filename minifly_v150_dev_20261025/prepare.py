"""Restore only pinned runtime source and public development corpus; no scientific trajectories."""
from pathlib import Path
import subprocess,json,hashlib
H=Path(__file__).resolve().parent;R=H.parent;p=json.loads((R/'minifly_output_learning_design_v1_20261025/PROVENANCE.json').read_text());commit=p['source_commit'];source={}
(H/'vendor'/'data').mkdir(parents=True,exist_ok=True)
for name in ['corpus.py','ladder.py','data/bryant-stories.txt','data/burgess-busterbrown.txt']:
 path='minifly_v148_pretrain_probe_dev_20261025/'+name;raw=subprocess.check_output(['git','show',commit+':'+path],cwd=R);assert hashlib.sha256(raw).hexdigest()==p['sources'][path]['sha256'];(H/'vendor'/name).write_bytes(raw);source[path]=p['sources'][path]
for name in ['fastmem.py','OUTPUT_CODEBOOK.json']:
 path='minifly_cls_dev_20261017/'+name;assert hashlib.sha256((R/path).read_bytes()).hexdigest()==p['sources'][path]['sha256']
(H/'RESTORATION.json').write_text(json.dumps({'source_commit':commit,'restored':source,'scientific_data_or_trajectories_restored':False,'Alice_restored':False},indent=2)+'\n')
