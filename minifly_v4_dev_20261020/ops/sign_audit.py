from pathlib import Path
import sys,json,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v4_common import *
from native_engine import Engine

def run():
 e=Engine(5177,6401,False);x=np.zeros(5177);x[100:203]=1
 z=e.values(x);before=z.copy()
 e.event(x,np.ones(9)*.5,record_id=1);after=e.values(x)
 # Native positive signed coefficients decrease the readout; port residual must match.
 delta=after-before
 assert np.all(delta<0),delta
 r=dict(accepted=True,native_positive_coefficient_value_delta=delta.tolist(),required_teaching_coefficient='positive (prediction_minus_target)/2, as in qualified native parent',full_record_applications=0,partial_native_events=1,scientific_records=0)
 save(Path('ops/SIGN_AUDIT.json'),r);print(json.dumps(r))
if __name__=='__main__':run()
