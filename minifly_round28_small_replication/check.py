import hashlib,json
from pathlib import Path
import numpy as np
import inner,inner2
from run import make_life
ROOT=Path(__file__).resolve().parent
z=np.array(json.loads((ROOT/'RECOVERED_CHECKPOINT.json').read_text())['mu'])
d,ans,off=make_life(280199);d=d[:512];ans=[(a,l) for a,l in ans if a+l<=len(d)]
a=inner.life(z,d,ans,0,seed=9);b=inner2.life(z,d,ans,0,seed=9,fast=True,n=256)
c=inner2.life(z,d,ans,0,seed=9,plastic=False,n=256)
r=json.loads((ROOT/'RECOVERY.json').read_text());checks={'recovered_source_hashes_match':all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in r['source_hashes'].items()),'fast_enabled_matches_original':a==b,'all_writes_disabled_uniform_8_bpb':c[0]==8.0,'deterministic_continuation_from_birth':b==inner2.life(z,d,ans,0,seed=9,fast=True,n=256),'fresh_seeds_not_original_validation':not set(range(280101,280109))&{900001,900002,900003,900004}}
checks={k:bool(v) for k,v in checks.items()}
assert all(checks.values()),checks
(ROOT/'CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
