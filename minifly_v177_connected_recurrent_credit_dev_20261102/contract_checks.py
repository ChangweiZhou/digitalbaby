"""No-fit interface, ownership, RNG and optimizer roundtrip checks."""
import json,hashlib
import numpy as np
import core as C
p,o,e,b=C.init(202611027701,0);before=hashlib.sha256(p.tobytes()+o.tobytes()).hexdigest();h,l,_,_=C.step(p,o,e,np.zeros((257,128)),np.arange(257));assert np.isfinite(h).all() and np.isfinite(l).all() and np.all(np.exp(l)>0) and np.allclose(np.exp(l).sum(1),1)
np.testing.assert_array_equal(C.step(p,o,e,np.ones((1,128)),np.array([2]),False)[0],C.step(p,o,e,np.zeros((1,128)),np.array([2]),False)[0])
g=C.rng(202611027701,'technical','roundtrip');state=g.bit_generator.state;expected=g.permutation(64);restored=C.rng(202611027701,'technical','roundtrip');restored.bit_generator.state=state;assert np.array_equal(expected,restored.permutation(64))
opt=C.Adam(p.copy(),o.copy());gr=[np.full_like(p,.0001),np.full_like(o,-.0002)];opt.update(gr,0,97,.001)
pp,oo=[v.copy() for v in opt.params];other=C.Adam(pp,oo);other.m=[v.copy() for v in opt.m];other.v=[v.copy() for v in opt.v];other.t=opt.t
opt.update(gr,97,97,.001);other.update(gr,97,97,.001);assert all(np.array_equal(a,b) for a,b in zip(opt.params,other.params))
assert before==hashlib.sha256(p.tobytes()+o.tobytes()).hexdigest()
C.save('CONTRACT_CHECKS.json',dict(status='PASS',all256_input_bytes_and_EOS=True,unmasked257_softmax=True,RNG_restoration=True,optimizer_continuation=True,recurrence_off_removes_all_state_carry=True,base_parameters_readonly=True,synthetic_optimizer_fixtures_not_science=True,extra_input_forward_steps=257,training_records=0))
