"""Zero-native-event post-pilot boundary tests, no science replay."""
from core_boundary_revision import Fly
import json
from pathlib import Path
f=Fly.__new__(Fly) # Invalid inputs must fail before any core access or mutation.
checks=[]
for name,call in [
 ('query_input_257',lambda:f.query(bytes(257),1)),
 ('generation_cap_17',lambda:f.generation(1,17)),
 ('negative_generation_cap',lambda:f.generation(1,-1)),
 ('record_input_257',lambda:f.record(bytes(257),b'')),
 ('record_answer_17',lambda:f.record(b'',bytes(17))),
 ('invalid_payload_before_learning',lambda:f.record(b'A',[-1]))]:
 try:call()
 except ValueError:checks.append(name)
 else:raise AssertionError(name)
assert Fly._payload(bytes(range(256)),256)==bytes(range(256));assert Fly._payload(b'',16)==b''
# Strengthen the score test on an explicitly OPEN gate (the first suite used a fixed neuron).
from core import Controller,np,SIGMA
c=Controller(6101);mu,prob,g,v,u=c.step(19);i=int(np.flatnonzero(g)[0]);j=19;eps=1e-6
w=c.Wv[i,j]
def lp():return float(-.5*np.sum(g*(v-c.Wv@u)**2)/SIGMA**2)
c.Wv[i,j]=w+eps;a=lp();c.Wv[i,j]=w-eps;b=lp();c.Wv[i,j]=w
error=abs((a-b)/(2*eps)-c.Ev[i,j]);assert error<1e-7
p=Path(__file__).resolve().parent
(p/'cycles/03_BOUNDARY_REVISION_TEST.json').write_text(json.dumps(dict(status='PASS',checks=checks,all_byte_payload_roundtrip=True,open_gate_score_FD_error=error,native_events=0,records=0,fits=0,scope='API-only revision outside frozen main source. No addition run is repeated.'),indent=2)+'\n')
