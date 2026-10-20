from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v4_common import *
from native_engine import Engine
from core_v4 import target_bits
x=np.zeros(5177);x[100:203]=1;e=Engine(5177,405,False);c=Engine(5177,405,False)
z=e.values(x);want=target_bits(128);coeff=(np.tanh(z/2)-want)/32
e.event(x,coeff,record_id=1);c.event(x)
after=e.values(x);passive=c.values(x)
r=dict(birth_values=z.tolist(),after_values=after.tolist(),matched_no_write_values=passive.tolist(),isolated_teaching_delta=(after-passive).tolist(),target=want.tolist(),coefficient=coeff.tolist())
save(Path('ops/PORT_BASELINE_DIAGNOSIS.json'),r);print(json.dumps(r))
