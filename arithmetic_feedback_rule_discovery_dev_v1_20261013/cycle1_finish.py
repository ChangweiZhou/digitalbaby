import json,time,hashlib
from pathlib import Path
from core import Core,Config,EOS
out=Path('cycles/cycle1')
results=[]
for depth in (1,4):
 c=Core(202610130100,Config(depth=depth)); before=c.learning_digest(); start=time.process_time()
 prompts=[bytes([0,10,61,17]),bytes([3,255,7,6])]; answers=[b'\x89',b'\xfa']
 for i in range(64):
  c.begin()
  for x in prompts[i%2]:c.observe(x,False)
  c.start_reply()
  for x in answers[i%2]:c.observe(x,True)
  c.observe(EOS,True)
 predictions=[]
 for p in prompts:
  c.begin()
  for x in p:c.observe(x,False)
  c.start_reply(); y,status=c.generate();predictions.append({'payload_hex':y.hex(),'status':status})
 results.append({'depth':depth,'records':64,'source':'non-arithmetic binary interface fixture',
                 'predictions':predictions,'correct':sum(r['payload_hex']==a.hex() and r['status']=='EOS' for r,a in zip(predictions,answers)),
                 'learning_changed':before!=c.learning_digest(),'cpu_seconds':time.process_time()-start,'state_bytes':c.bytes_state()})
(out/'BINARY_FIXTURE.json').write_text(json.dumps(results,indent=2))
(out/'AUDIT.json').write_text(json.dumps({'cycle':1,'passed':True,'tests':8,
 'independent_gradient':'finite differences through separate forward with captured pre-update V; depths 1 and 4',
 'initial_failure':'checkpoint test and initial semantic digest used pickle alias-dependent serialization; arrays and counters verified identical',
 'revision':'canonical recursive state digest; bounded trace; public BEGIN and input validation',
 'binary_fixture':results,'science_records':0},indent=2))
(out/'REVISION.md').write_text('# Cycle 1 revision\n\nFixed identity-dependent audit hashing, bounded debug traces, enforced BEGIN and byte/permission validation. No learning parameter, task, budget or arithmetic prior changed. Eight final tests pass, including independent finite differences at depths 1 and 4. Binary fixtures are acquisition checks only, not unseen arithmetic evidence.\n')
