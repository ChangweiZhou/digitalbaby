"""Independent finite differences, interface tests and causal write controls."""
import copy
import hashlib
import pickle
import tempfile
import unittest
import numpy as np
from core import Core,Config,EOS,BEGIN,REPLY

# Deliberately separate forward implementation used by numerical gradient reference.
def reference_step(theta,h,token=None,zero=False):
    e = np.zeros(theta['E'].shape[1]) if zero else theta['E'][token]
    z = 1/(1+np.exp(-(theta['Wz']@e+theta['Uz']@h+theta['bz'])))
    r = 1/(1+np.exp(-(theta['Wr']@e+theta['Ur']@h+theta['br'])))
    g = np.tanh(theta['Wg']@e+theta['Ug']@(r*h)+theta['bg'])
    return (1-z)*h+z*g

def reference_loss(theta,events,readouts,depth,width):
    h = np.zeros(width)
    total=0.
    for token,V,target in events:
        if V is not None:
            hh=h.copy()
            for _ in range(depth-1): hh=reference_step(theta,hh,zero=True)
            u=np.r_[hh,1.]; u/=np.linalg.norm(u)
            z=u@readouts[V]
            y=np.zeros(257); y[target]=1
            total += .5*np.sum((y-z)**2)
        h=reference_step(theta,h,token)
    return total/len(readouts)

class Tests(unittest.TestCase):
    def test_gradient_preupdate_online_readouts(self):
        self.gradient_errors=[]
        for depth in (1,4):
            c=Core(11,Config(depth=depth,width=5,embedding=3))
            c.V[:]=np.random.default_rng(14).normal(0,.02,c.V.shape)
            events=[(BEGIN,None,None),(7,None,None),(3,None,None),(REPLY,None,None)]
            c.begin(); c.observe(7); c.observe(3); c.start_reply()
            readouts=[]
            for target in (119,0,10):
                readouts.append(c.V.copy()); events.append((target,len(readouts)-1,target))
                c.observe(target,True)
            grad=c.pending_gradient()
            rng=np.random.default_rng(13)
            maximum=0.
            for key in c.theta:
                indices=list(np.ndindex(c.theta[key].shape))
                if key=='E': indices=[(t,j) for t in (BEGIN,REPLY,7,3,119,0) for j in range(3)]
                for index in [indices[i] for i in rng.choice(len(indices),min(4,len(indices)),replace=False)]:
                    before=c.theta[key][index]; eps=1e-6
                    c.theta[key][index]=before+eps
                    a=reference_loss(c.theta,events,readouts,depth,5)
                    c.theta[key][index]=before-eps
                    b=reference_loss(c.theta,events,readouts,depth,5)
                    c.theta[key][index]=before
                    numerical=(a-b)/(2*eps)
                    maximum=max(maximum,abs(numerical-grad[key][index]))
            self.assertLess(maximum,2e-8)
            self.gradient_errors.append(maximum)
            # Incorrect final-V convention must not accidentally match this test.
            wrong=reference_loss(c.theta,events,[c.V]*len(readouts),depth,5)
            right=reference_loss(c.theta,events,readouts,depth,5)
            self.assertGreater(abs(wrong-right),1e-4)

    def test_all_bytes_and_pure_prediction_generation(self):
        c=Core(10)
        c.begin()
        for b in range(256): c.observe(b,False)
        c.start_reply(); c.observe(200,True)
        before=pickle.dumps(c,protocol=5)
        c.predict(); c.predict(); c.generate()
        self.assertEqual(before,pickle.dumps(c,protocol=5))
        self.assertEqual(len(c.tape),32)
        self.assertGreater(c.evictions,0)

    def test_no_write_and_frozen(self):
        for learned in (False,True):
            c=Core(7,Config(learned=learned)); initial=c.learning_digest()
            c.begin()
            for x in (0,61,10,255): c.observe(x,False)
            c.start_reply(); c.observe(90,False); c.observe(EOS,False)
            self.assertEqual(initial,c.learning_digest())
            theta={k:v.copy() for k,v in c.theta.items()}
            c.begin(); c.observe(90,True)
            self.assertEqual(c.steps,0)
            for k in theta: np.testing.assert_array_equal(theta[k],c.theta[k])
            c.observe(EOS,True)
            self.assertEqual(c.steps,int(learned))
            if not learned:
                for k in theta: np.testing.assert_array_equal(theta[k],c.theta[k])

    def test_checkpoint_mid_sequence_and_chunking(self):
        a=Core(4,Config(depth=4)); a.begin(); a.observe(200,True)
        with tempfile.TemporaryDirectory() as td:
            a.save(td+'/c.pkl'); b=Core.load(td+'/c.pkl')
            for c in (a,b):
                for byte in bytes([10,0,61,255]): c.observe(byte,True)
                c.observe(EOS,True)
            self.assertEqual(pickle.dumps(a,protocol=5),pickle.dumps(b,protocol=5))

    def test_confidence_stability(self):
        rng=np.random.default_rng(66)
        for full in (True,False):
            c=Core(12,Config(full=full)); c.begin()
            for _ in range(200):
                c.observe(int(rng.integers(0,256)),True)
            c.observe(EOS,True)
            c.assert_finite()
            if full: self.assertGreater(np.linalg.eigvalsh(c.P).min(),-1e-10)
            else: self.assertGreater(c.P.min(),0)

    def test_payload_permutation_equivariance(self):
        rng=np.random.default_rng(13)
        perm=np.r_[rng.permutation(256),256,257,258]
        a=Core(32,Config(depth=4)); a.V[:]=rng.normal(0,.02,a.V.shape)
        b=a.clone(); b.theta['E'][perm]=a.theta['E']; b.V[:,perm[:257]]=a.V
        for c in (a,b): c.begin()
        for x in (0,61,10,42): a.observe(x,True); b.observe(int(perm[x]),True)
        a.start_reply(); b.start_reply()
        choice,pa,_=a.predict(); other,pb,_=b.predict()
        np.testing.assert_allclose(pa,pb[perm[:257]],atol=1e-14)
        self.assertEqual(int(perm[choice]),other)
        a.observe(EOS,True); b.observe(EOS,True)
        np.testing.assert_allclose(a.theta['E'],b.theta['E'][perm],atol=1e-13)
        for k in a.theta:
            if k!='E': np.testing.assert_allclose(a.theta[k],b.theta[k],atol=1e-13)

    def test_independent_write_protection(self):
        c=Core(22); c.begin(); c.start_reply()
        arrays=list(c.theta.values())+list(c.mom.values())+[c.P,c.V]
        for a in arrays: a.flags.writeable=False
        c.observe(5,False); c.observe(EOS,False); c.generate()
        with self.assertRaises(ValueError): c.observe(0,True)

    def test_abort_and_generated_eos_do_not_flush(self):
        c=Core(2); c.begin(); c.observe(8,True)
        c.V[:,EOS]=20.
        c.generate(); self.assertEqual(c.steps,0)
        c.observe(EOS,False); self.assertEqual(c.steps,0)
        c.begin(); self.assertEqual(c.aborted,1); self.assertEqual(c.permitted,0)

if __name__=='__main__': unittest.main(verbosity=2)
