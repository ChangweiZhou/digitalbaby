"""Native causal and identity qualification, not an efficacy pilot."""
import copy,dataclasses,json,time,unittest
from pathlib import Path
import numpy as np
import bridge_runtime as rt
from run_job import record,score_policies,EarnedChoiceOrgan,DEVELOPMENT_WORLDS,run
from authority_policy import Observation,emit,authority
from independent_audit import audit_record,audit_readouts,FLAGS

class Cycle2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world=rt.make_world(DEVELOPMENT_WORLDS[0],'lifetime')
        cls.base=rt.Core('ERROR')

    def test_rejected_original_gate_is_algebraically_inert(self):
        for shared in (.1,1.,3.):
            for private in (.01,.05,.09):
                if shared>private:
                    self.assertGreater(shared-private,0)
                    self.assertGreater(shared-.5*private,0)
        obs=[Observation((0.,1.5,0.,0.),(2.,0.,0.,0.)),Observation((0.,0.,0.,0.),(0.,0.,0.,0.))]
        self.assertNotEqual(emit(obs,'ERROR')['emitted'],emit(obs,'Q_EARNED')['emitted'])
        self.assertEqual(emit(obs,'Q_EARNED')['emitted'],emit(obs,'Q_HALF')['emitted'])

    def test_independent_births_and_mutable_arrays(self):
        self.assertEqual(len({id(m.fly) for m in self.base.models}),8)
        self.assertEqual(len({id(m.fly.m.fast) for m in self.base.models}),8)
        for a,m in enumerate(self.base.models):
            for n in self.base.models[a+1:]:
                for k in ('fast','slow','adapt'):
                    self.assertFalse(np.shares_memory(getattr(m.fly.m,k),getattr(n.fly.m,k)))

    def test_all_twelve_stage_branch_decisions_and_tampers(self):
        for branch,flags in FLAGS.items():
            for stage,allowed in flags.items():
                core=self.base.clone();event=copy.deepcopy(self.world['events'][0]);event['stage']=stage
                row=record(core,event,branch);audit_record(row,event,branch)
                self.assertIs(row['learn'],allowed)
                bad=copy.deepcopy(row);bad['actual_calls'][0]['write']=not allowed
                with self.assertRaises(AssertionError):audit_record(bad,event,branch)
                if not allowed:
                    bad=copy.deepcopy(row);bad['actual_calls'][0]['delta_l1']=.01
                    with self.assertRaises(AssertionError):audit_record(bad,event,branch)

    def test_output_is_independent_of_current_teacher_and_labels(self):
        e1=copy.deepcopy(self.world['events'][0]);e2=copy.deepcopy(e1);e2['outcome']=49 if e1['outcome']!=49 else 50
        a=record(self.base.clone(),e1,'W');b=record(self.base.clone(),e2,'W')
        self.assertEqual(a['readouts'],b['readouts'])
        self.assertNotEqual(a['write']['s'],b['write']['s'])
        wrong=copy.deepcopy(a);wrong['operation_order'][2:4]=reversed(wrong['operation_order'][2:4])
        with self.assertRaises(AssertionError):audit_record(wrong,e1,'W')

    def test_policies_do_not_change_learning_or_pending_state(self):
        c=self.base.clone();e=self.world['events'][0]
        for i,b in enumerate(bytes.fromhex(e['cue_hex'])):c.feed(b,e['at']+i*rt.DT)
        p=c.predict(e['at']+12*rt.DT);before=c.state_digest()
        policies,_=score_policies([p]);self.assertEqual(before,c.state_digest())
        row=record(self.base.clone(),e,'W');d=self.base.clone()
        from worker import record as native_record
        old=native_record(d,e,'W')
        nc=self.base.clone();record(nc,e,'W')
        self.assertEqual(d.state_digest(),nc.state_digest())
        self.assertEqual(row['prediction'],old['prediction'])
        for field in ('alpha','shared_margin','shared_winner','emitted'):
            bad=copy.deepcopy(policies);bad['Q_EARNED'][field]+=1
            with self.assertRaises(AssertionError):audit_readouts([dataclasses.asdict(p)],bad)

    def test_serial_choice_parent_parity_and_query_isolation(self):
        w=rt.make_world(DEVELOPMENT_WORLDS[0],'reuse');a,b=w['sets']['old'][:2]
        ca,cb=bytes.fromhex(a['cue_hex']),bytes.fromhex(b['cue_hex'])
        from core import ChoiceOrgan
        source=ChoiceOrgan(self.base);native,values,ps,half=source.choose(ca,cb,0.,rt.DT)
        before=self.base.state_digest();preds,pol,_=EarnedChoiceOrgan(self.base).choose((ca,cb),0.)
        self.assertEqual(self.base.state_digest(),before)
        self.assertEqual(pol['ERROR']['emitted'],native);self.assertEqual(pol['Q_HALF']['emitted'],half)
        self.assertEqual([dataclasses.asdict(p) for p in ps],[dataclasses.asdict(p) for p in preds])

    def test_reject_science_world(self):
        with self.assertRaises(ValueError):run(62011001,'reuse','/private/tmp/must_not_run_earned')

if __name__=='__main__':
    begin=time.monotonic();res=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cycle2))
    out=Path(__file__).parent/'cycles/cycle2';out.mkdir(parents=True,exist_ok=True)
    (out/'TEST_RESULT.json').write_text(json.dumps({'cycle':2,'tests_run':res.testsRun,'failures':len(res.failures),'errors':len(res.errors),
        'verdict':'PASS' if res.wasSuccessful() else 'FAIL','wall_s':time.monotonic()-begin,
        'scientific_revision':'Replace inert shared-dominance gate by mechanically derived .5*private_gap < shared_gap < private_gap on disagreement.',
        'formal_science_jobs':0},indent=2)+'\n')
    raise SystemExit(not res.wasSuccessful())
