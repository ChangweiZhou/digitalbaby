"""Independent numerical derivatives and generic causal/interface fixtures."""
import copy, pickle, tempfile, unittest
import numpy as np
from core import Core,Config,EOS,BEGIN,REPLY

def step(t,h,token):
    e=t['E'][token]
    z=1/(1+np.exp(-(t['Wz']@e+t['Uz']@h+t['bz'])))
    r=1/(1+np.exp(-(t['Wr']@e+t['Ur']@h+t['br'])))
    g=np.tanh(t['Wg']@e+t['Ug']@(r*h)+t['bg'])
    return h*(1-z)+z*g

def objective(c,tokens,permissions,anchors=None):
    total=0.;ids=[];h=np.zeros(c.config.width);permitted=sum(permissions)
    for i,(token,allowed) in enumerate(zip(tokens,permissions)):
        if allowed:
            if c.config.reader=='LAG32':
                recent=ids[-c.config.window:];x=np.zeros(c.config.window*c.config.embedding)
                for j,t in enumerate(recent):
                    k=(c.config.window-len(recent)+j)*c.config.embedding
                    x[k:k+c.config.embedding]=c.theta['E'][t]
                hh=np.tanh(c.theta['A']@x+c.theta['b'])
            elif anchors is not None:
                start=max(0,i-c.config.window);hh=anchors[start].copy()
                for t in tokens[start:i]:hh=step(c.theta,hh,t)
            else:hh=h
            u=np.concatenate((hh,[1.]));u/=np.linalg.norm(u);z=u@c.V
            if c.config.objective=='CE':total+=np.logaddexp.reduce(z)-z[token]
            else:y=np.eye(257)[token];total+=.5*np.sum((z-y)**2)
        if token!=EOS:
            ids.append(token)
            if c.config.reader=='GRU':h=step(c.theta,h,token)
    return total/max(1,permitted)

class Tests(unittest.TestCase):
    def numeric(self,c,tokens,mask,anchors=None,keys=None):
        c.begin()
        for token,allowed in zip(tokens[1:],mask[1:]):
            if token==REPLY:c.start_reply()
            elif token!=EOS:c.observe(token,allowed)
            else:
                # Accumulate actual EOS target without committing, for gradient inspection.
                original=c._flush;c._flush=lambda:None
                c.observe(EOS,allowed);c._flush=original
        grad=c.pending_gradient();rng=np.random.default_rng(18);errors=[]
        for key in keys or grad:
            array=c.parameters()[key]
            if key=='E':indices=[(t,j) for t in set(tokens)-{EOS} for j in range(c.config.embedding)]
            else:indices=list(np.ndindex(array.shape))
            for j in rng.choice(len(indices),min(3,len(indices)),replace=False):
                idx=indices[j];old=array[idx];eps=1e-6
                array[idx]=old+eps;a=objective(c,tokens,mask,anchors)
                array[idx]=old-eps;b=objective(c,tokens,mask,anchors)
                array[idx]=old
                errors.append(abs((a-b)/(2*eps)-grad[key][idx]))
        self.assertLess(max(errors),2e-8)
        return max(errors)

    def test_joint_short_gradients_all_reader_objectives(self):
        tokens=[BEGIN,0,255,REPLY,61,10,EOS];mask=[False,False,False,False,True,True,True]
        for reader in ('GRU','LAG32'):
            for loss in ('CE','SQ'):
                c=Core(12,Config(reader=reader,objective=loss,width=4,embedding=2))
                c.V[:]=np.random.default_rng(13).normal(0,.02,c.V.shape)
                self.numeric(c,tokens,mask)

    def test_B_all_byte_gradient_and_long_per_target_horizon(self):
        tokens=[BEGIN]+list(range(9))+[REPLY,51,52,EOS]
        mask=[t not in (BEGIN,REPLY) for t in tokens]
        c=Core(4,Config(width=3,embedding=2,window=4,gradient_mode='per_target'))
        c.V[:]=np.random.default_rng(9).normal(0,.1,c.V.shape)
        h=np.zeros(3);anchors=[h.copy()]
        for token in tokens:
            if token!=EOS:h=step(c.theta,h,token)
            anchors.append(h.copy())
        self.numeric(c,tokens,mask,anchors)
        self.assertEqual(c.permitted,12)
        self.assertTrue(c.streaming)
        self.assertLessEqual(len(c.tape),4)

    def test_no_update_including_decay_and_actual_EOS(self):
        for reader in ('GRU','LAG32'):
            c=Core(7,Config(reader=reader,decay=.01,gradient_mode='per_target'))
            initial=c.learning_digest();c.begin()
            for token in range(256):c.observe(token,False)
            c.start_reply();c.observe(17,False);c.observe(EOS,False)
            self.assertEqual(initial,c.learning_digest());self.assertEqual(c.steps,0)

    def test_actual_EOS_joint_step_and_independent_RMS_equation(self):
        c=Core(21,Config(width=4,embedding=2,decay=.01));c.begin();c.observe(14,False);c.start_reply()
        initial=c.learning_digest();c.observe(19,True)
        self.assertEqual(initial,c.learning_digest());self.assertEqual(c.steps,0)
        old=c._flush;c._flush=lambda:None;c.observe(EOS,True);c._flush=old
        grad=c.pending_gradient();before={k:a.copy() for k,a in c.parameters().items()}
        norm=np.sqrt(sum((g*g).sum() for g in grad.values()));clip=min(1,1/max(norm,1e-30))
        c._flush()
        for k,g in grad.items():
            clipped=g*clip
            expected=before[k]*(1-.001*.01)-.001*clipped/(np.sqrt(clipped*clipped)+1e-8)
            np.testing.assert_allclose(c.parameters()[k],expected,atol=1e-13)
        self.assertEqual((c.steps,c.learning_events,c.encoder_events),(1,1,1))

    def test_frozen_readers_encoder_unchanged_head_learns(self):
        for reader in ('GRU','LAG32'):
            c=Core(5,Config(reader=reader,learned=False));theta=copy.deepcopy(c.theta)
            c.begin();c.observe(200,False);c.start_reply();c.observe(6,True);c.observe(EOS,True)
            for k in theta:np.testing.assert_array_equal(theta[k],c.theta[k])
            self.assertGreater(np.linalg.norm(c.V),0);self.assertEqual(c.encoder_events,0)
            self.assertEqual(set(c.mom),{'V'})

    def test_generated_guesses_pure_with_pending_targets(self):
        for reader in ('GRU','LAG32'):
            c=Core(8,Config(reader=reader));c.begin();c.observe(10,False);c.start_reply();c.observe(0,True)
            before=pickle.dumps(c,protocol=5);c.predict();c.generate(16)
            self.assertEqual(before,pickle.dumps(c,protocol=5));self.assertEqual(c.steps,0)

    def test_empty_and_truncation_and_all_payload_values(self):
        c=Core(2);c.begin()
        for token in range(256):c.observe(token,False)
        c.start_reply();c.V[-1,EOS]=100
        self.assertEqual(c.generate(0),(b'','EOS'))
        c.V.fill(0);self.assertEqual(c.generate(0),(b'','TRUNCATED'))
        self.assertEqual(c.generate(2),(b'\x00\x00','TRUNCATED'))

    def test_lag_padding_distinct_from_NUL_and_current_embeddings(self):
        c=Core(3,Config(reader='LAG32',window=4));c.begin();u,_=c.features()
        c.observe(0,False);v,_=c.features();self.assertGreater(np.linalg.norm(u-v),0)
        before=v.copy();c.theta['E'][0]*=2.;after,_=c.features()
        self.assertGreater(np.linalg.norm(before-after),0)

    def test_clone_checkpoint_and_chunk_parity(self):
        c=Core(4);c.begin();c.observe(255,True)
        with tempfile.TemporaryDirectory() as td:
            c.save(td+'/c.pkl');d=Core.load(td+'/c.pkl')
            for x in (c,d):x.observe(0,True);x.start_reply();x.observe(8,True);x.observe(EOS,True)
            self.assertEqual(c.learning_digest(),d.learning_digest())
            e=c.clone();e.V[0,0]+=1.;self.assertNotEqual(c.learning_digest(),e.learning_digest())

    def test_prohibited_EOS_and_abort_do_not_flush(self):
        c=Core(3);c.begin();c.observe(7,True);c.observe(EOS,False)
        self.assertEqual(c.steps,0);c.begin();self.assertEqual(c.aborted,1);self.assertEqual(c.permitted,0)

    def test_readonly_arrays_prevent_any_partial_commit(self):
        c=Core(3);c.begin();c.start_reply();c.observe(17,True)
        before=c.learning_digest();c.V.flags.writeable=False
        with self.assertRaises(ValueError):c.observe(EOS,True)
        self.assertEqual(before,c.learning_digest())

    def test_array_parameter_ledger(self):
        for reader,total,learned_bytes,frozen_bytes in [('GRU',36401,582416,424848),('LAG32',53681,858896,563088)]:
            for learned,nbytes in [(True,learned_bytes),(False,frozen_bytes)]:
                c=Core(12,Config(reader=reader,learned=learned))
                self.assertEqual(sum(a.size for a in c.parameters().values()),total)
                self.assertEqual(sum(a.nbytes for a in c.parameters().values())+sum(a.nbytes for a in c.mom.values()),nbytes)

if __name__=='__main__':unittest.main(verbosity=2)
