"""Fixed non-science shared-only plumbing test; exactly 1024 records per arm."""
import json,time
from core import InteractionCore,CANDIDATES
FIXTURE=((b'context A',b'A'),(b'context B',b'BC'),(b'context C',b'-7'),(b'context D',b'12'),(b'context E',b'49'))
def run():
 out={}
 for name in CANDIDATES:
  c=InteractionCore(name,202610129601);tic=time.process_time()
  for i in range(1024):
   prompt,answer=FIXTURE[i%5];c.begin_sequence();c.observe_bytes(prompt,False);c.start_reply()
   c.observe_bytes(answer,True);c.observe_end(True)
  outputs=[]
  for prompt,answer in FIXTURE:
   c.begin_sequence();c.observe_bytes(prompt,False);c.start_reply();prior=c.learned_digest()
   # Override prediction only for shared-only diagnostic autoregression.
   c.prediction=c.shared_prediction;g=c.generate(16)
   assert c.learned_digest()==prior
   outputs.append({'prompt_hex':prompt.hex(),'target_hex':answer.hex(),'output_hex':g.payload.hex(),
    'status':g.status,'correct':g.payload==answer and g.status=='ENDED'})
  out[name]={'records':1024,'training_and_test_cpu':time.process_time()-tic,'outputs':outputs,'pass':all(x['correct'] for x in outputs)}
 return {'technical':True,'science_jobs':0,'fixture_not_E3':True,'arms':out,'passed':all(x['pass'] for x in out.values())}
if __name__=='__main__':
 import sys
 result=run();print(json.dumps(result,indent=2))
 if len(sys.argv)>1:
  from pathlib import Path
  Path(sys.argv[1]).write_text(json.dumps(result,indent=2))
 raise SystemExit(0 if result['passed'] else 1)
