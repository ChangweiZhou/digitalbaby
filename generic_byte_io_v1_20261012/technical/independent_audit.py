import json,sys,hashlib,tempfile,subprocess,os
from pathlib import Path
import numpy as np
SOURCE=Path('/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/generic_byte_io_v1_20261012')
sys.path.insert(0,str(SOURCE))
from generic_byte_io import ByteCore,CoreConfig,EOS
checks={}
def frame(c,prompt,answer,learn):
    c.begin_sequence(); c.observe_bytes(prompt,learn=learn); c.start_reply(); c.observe_bytes(answer,learn=learn); c.observe_end(learn=learn)
def query(c,prompt,cap):
    c.begin_sequence(); c.observe_bytes(prompt,learn=False); c.start_reply(); return c.generate(cap)
# Independently chosen technical fixture: different early bytes, identical four-byte suffix.
prompts=[b'first-A'+b'\x00=\n\xff',b'first-B'+b'\x00=\n\xff',b'AB'+b'\x00=\n\xff',b'BA'+b'\x00=\n\xff',b'\x00\xff\n=',b'']
answers=[b'\xff\x00\n=ANSWER-A',b'\x00\xff\n=ANSWER-B','独立审计🪰'.encode(),bytes(range(255,-1,-1)),b'',b'x']
c=ByteCore(); off=ByteCore()
for epoch in range(3):
    for p,a in zip(prompts,answers):
        frame(c,p,a,True); frame(off,p,a,False)
learned=c.learned_digest()
rows=[]
for p,a in zip(prompts,answers):
    generated=query(c,p,400)
    assert generated.payload==a and generated.status=='ENDED',(p,generated)
    assert c.learned_digest()==learned
    rows.append({'prompt_hex':p.hex(),'answer_len':len(a),'selected_events':generated.selected_events})
checks['independent_binary_utf8_empty_order_same_suffix_own_output']=True
checks['all_256_payload_emitted_in_one_learned_answer']=sorted(answers[3])==list(range(256))
checks['generation_freezes_entire_learned_bank_and_counter']=True
assert off.learning_events==0
assert query(off,prompts[0],400).payload!=answers[0]
checks['matched_no_write_does_not_recite']=True
# Correct EOS at zero or exact cap; no hidden answer-length argument.
assert query(c,prompts[4],0).status=='ENDED'
assert query(c,prompts[0],len(answers[0])).status=='ENDED'
r=query(c,prompts[0],3); assert r.status=='TRUNCATED' and r.payload==answers[0][:3]
rest=c.generate(400);assert rest.payload==answers[0][3:] and rest.status=='ENDED'
assert c.learned_digest()==learned
checks['zero_cap_exact_cap_and_truncated_continuation']=True
# Chunking and ring overflow with prefix markers included in event count.
x=bytes(range(256))*3
one=ByteCore(CoreConfig(max_context_events=17));chunks=ByteCore(CoreConfig(max_context_events=17))
one.begin_sequence();chunks.begin_sequence();one.observe_bytes(x)
for j in range(0,len(x),13):chunks.observe_bytes(x[j:j+13])
assert one.learned_digest()==chunks.learned_digest()
assert one.transient_digest()==chunks.transient_digest()
assert one.context_overflow_events==1+len(x)-17
checks['learned_chunk_parity_after_context_overflow']=True
# A truly cold process must reproduce state and next-output scores, not Python hash-seed dependent.
with tempfile.TemporaryDirectory() as tmp:
    checkpoint=Path(tmp)/'state.npz';one.save_checkpoint(checkpoint)
    child='import sys,json; from generic_byte_io import ByteCore; c=ByteCore.load_checkpoint(sys.argv[1]); print(json.dumps({"learned":c.learned_digest(),"transient":c.transient_digest(),"event":c.prediction().event,"p":c.prediction().probabilities.tolist()}))'
    env=os.environ.copy();env['PYTHONPATH']=str(SOURCE);env['PYTHONHASHSEED']='987654321'
    obj=json.loads(subprocess.check_output([sys.executable,'-c',child,str(checkpoint)],env=env))
    assert obj['learned']==one.learned_digest() and obj['transient']==one.transient_digest()
    assert obj['event']==one.prediction().event
    assert np.array_equal(np.array(obj['p']),one.prediction().probabilities)
checks['cold_process_checkpoint_and_hash_seed_parity']=True
# Read-only prediction arrays cannot be used to mutate bank.
a=c.learned_digest();c.begin_sequence();pred=c.prediction();pred.probabilities[:]=0
assert c.learned_digest()==a
assert c.prediction().probabilities.sum()>0
checks['prediction_array_not_aliasing_learning_state']=True
# A strided memoryview is a legal bytes-like payload; inspect declared surface robustness.
t=ByteCore();t.begin_sequence()
try:
    t.observe_bytes(memoryview(b'abcdef')[::2],learn=False)
    checks['strided_memoryview_admission']=t._ordered_context().tolist()==[257,97,99,101]
except Exception as error:
    checks['strided_memoryview_admission']={'error':type(error).__name__,'message':str(error)}
# Rest is explicit stable-engineering clock, with no hidden weight change.
prior=c.learned_digest();c.rest(86400);assert c.elapsed_seconds==86400 and c.learned_digest()==prior
for invalid in (-1,True,float('nan'),float('inf')):
    before=c.transient_digest()
    try:c.rest(invalid)
    except (TypeError,ValueError):pass
    else:raise AssertionError('invalid rest admitted')
    assert c.transient_digest()==before
with tempfile.TemporaryDirectory() as tmp:
    checkpoint=Path(tmp)/'rested.npz';c.save_checkpoint(checkpoint);restored=ByteCore.load_checkpoint(checkpoint)
    assert restored.elapsed_seconds==86400 and restored.learned_digest()==prior
    assert restored.transient_digest()==c.transient_digest()
checks['rest_clock_persisted_and_no_spontaneous_learning']=True
report={'classification':'INDEPENDENT_TECHNICAL_AUDIT_NOT_SCIENCE','checks':checks,'source_sha256':hashlib.sha256((SOURCE/'generic_byte_io/core.py').read_bytes()).hexdigest(),'rows':rows,'storage':c.storage_accounting()}
Path('/private/tmp/generic_io_independent_audit_20261012.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
