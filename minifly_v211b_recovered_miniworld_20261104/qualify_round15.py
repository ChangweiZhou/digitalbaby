"""Run one archived Round 15 validation row using the identical pinned training file."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

data=Path(sys.argv[1]).resolve()
assert hashlib.sha256(data.read_bytes()).hexdigest()=="c64ba9b1a49bbb111ffc4cd2ae4f129d25c012399dd7ade9679eea464b9afb05"
code=Path("round15_learner.py").read_text()
original="'/tmp/babi/qa1_single-supporting-fact_train.txt'"
assert code.count(original)==1
code=code.replace(original,repr(str(data)))
env=dict(os.environ,CTX="1",COMB="nb",FB="true",FAM="base",TIE="random",PYTHONHASHSEED="7",
         OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
row=json.loads(subprocess.check_output([sys.executable,"-B","-c",code],env=env,text=True))
reference={"fam":"base","fb":"true","ctx":True,"tie":"random","comb":"nb","seed":"7","acc":1.0,"tie_rate":.001}
result=dict(status="PASS" if row==reference else "MISMATCH",result=row,archived=reference,
            result_event="sevt_017uf6178tA4oJjeuAAH9KWc",modification="Dataset path relocated in memory only")
Path("ROUND15_BRIDGE.json").write_text(json.dumps(result,indent=2)+"\n")
print(result)
