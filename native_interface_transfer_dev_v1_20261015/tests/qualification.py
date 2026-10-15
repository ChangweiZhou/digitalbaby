"""Bounded technical checks, explicitly separate from developmental evidence."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import ast
import copy
import inspect
import time
import unittest
import numpy as np
import config
import models
import tasks
import native_track as nt
import addition_track as bt
from artifacts import write,now
from audits import checked_scores

LEDGER=[]

def charged(label,call):
    result=call();LEDGER.append(label);return result

class Qualification(unittest.TestCase):
    def test_01_data_and_scope(self):
        f=tasks.native_fixture()
        self.assertEqual(len(f),12)
        self.assertTrue(all(x['cue'][:8]==b' '*8 for x in f[:8]))
        self.assertEqual(len({x['cue'][-4:] for x in f}),12)
        # Two second-byte collisions and an EOS-versus-digit collision are explicit.
        self.assertEqual((f[8]['cue']+f[8]['answer'][:1])[-4:],(f[9]['cue']+f[9]['answer'][:1])[-4:])
        self.assertNotEqual(f[8]['answer'][1],f[9]['answer'][1])
        for world in config.TRACK_B_WORLDS:
            s=tasks.addition_streams(world);g=s['groups']
            self.assertEqual([len(g[k]) for k in ('old','new','held')],[60,20,20])
            self.assertFalse(set(g['old'])&set(g['new']) or set(g['old'])&set(g['held']))
            self.assertTrue(all((b,a) in g[k] for k in g for a,b in g[k]))
            self.assertFalse({tasks.render(p)[0] for p in g['held']}&{p for p,_ in s['old']+s['new']})
        tree=ast.parse((bootstrap.HERE/'models.py').read_text())
        names={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        names|={a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        self.assertFalse({'tasks','native_track','addition_track'}&names)
        self.assertFalse(config.LANGUAGE_TASK_ALLOWED or config.FORMAL_SCIENCE_ALLOWED)

    def test_02_original_event_parity(self):
        parent=bootstrap.source_module('archived_scheduled_reference','r_center_core_v1/tests/reference/learner.py')
        modern=models.CenteredCore();reference=parent.Learner('R_center',models.SCALES)
        item=tasks.native_fixture()[0];dt=nt.DT
        for i,b in enumerate(item['cue']):modern.feed(b,i*dt);reference.feed(b,i*dt)
        got=modern.predict(12*dt);expected=reference.predict(12*dt)
        np.testing.assert_array_equal(got.combined,expected['combined'])
        self.assertEqual(got.emitted,expected['emitted'])
        charged('legacy-current-W',lambda:modern.observe_outcome(item['answer'][0],12*dt))
        charged('legacy-reference-W',lambda:reference.observe_outcome(item['answer'][0],12*dt))
        modern.feed(10,13*dt);reference.feed(10,13*dt)
        modern.flush(165);reference.flush(165)
        self.assertEqual(modern.bank_digests(),reference.digests())
        # NO_WRITE after a nonzero learned state, not only at zero birth.
        charged('legacy-current-NONZERO_NO_WRITE',lambda:nt.scheduled_record(modern,item,config.TRACK_A_STAGES[0],False))

    def test_03_c16_and_no_write(self):
        model=models.ScheduledC16();item=tasks.native_fixture()[1]
        self.assertEqual(len({id(m.fly) for m in model.models}),32)
        first=charged('C16-W',lambda:nt.scheduled_record(model,item,config.TRACK_A_STAGES[1],True))
        self.assertEqual(len(first['logits']),257)
        self.assertGreater(first['raw_alpha_l1'],0)
        second=charged('C16-NONZERO_NO_WRITE',lambda:nt.scheduled_record(model,item,config.TRACK_A_STAGES[1],False))
        self.assertEqual(second['raw_alpha_l1'],0)
        self.assertLessEqual(second['forbidden_delta'],1e-8)
        before=model.state_digest();nt.probe(model,config.TRACK_A_STAGES[1],tasks.native_fixture()[:1])
        self.assertEqual(before,model.state_digest())
        for byte in range(257):self.assertEqual(int(models.PARENT.logits(models.CODE[byte]).argmax()),byte)
        # Ideal decode algebra is recorded, but it is NOT behavior qualification.

    def test_04_sequence_order_no_write_and_bytes(self):
        model=models.SequenceC16();calls=[]
        forecast=model.forecast;event=model._event
        def watch_forecast():calls.append('PREDICT');return forecast()
        def watch_event(token=None,target=None,allowed=False):
            if target is not None:calls.append(('OBSERVE',target,allowed))
            return event(token,target,allowed)
        model.forecast=watch_forecast;model._event=watch_event
        prompt=b'\x00\n =\xff';answer=b'\x00\xff'
        result=charged('sequence-W-two-payload-bytes',lambda:model.train(prompt,answer,True))
        self.assertEqual(calls,['PREDICT',('OBSERVE',0,True),'PREDICT',('OBSERVE',255,True),'PREDICT',('OBSERVE',256,True)])
        # Restore methods before clones; observer closures must never be cloned.
        del model.forecast;del model._event
        self.assertGreater(result['raw_alpha_l1'],0)
        r=charged('sequence-NONZERO_NO_WRITE',lambda:model.train(prompt,answer,False))
        self.assertEqual(r['raw_alpha_l1'],0);self.assertLessEqual(r['forbidden_delta'],1e-9)
        before=model.digest();out,status,trace=model.generate_trace(prompt,cap=2)
        self.assertIn(status,('EOS','TRUNCATED'));self.assertEqual(before,model.digest())
        self.assertEqual(len(inspect.signature(model.forecast).parameters),0)
        self.assertTrue(all(len(x['logits'])==257 for x in trace))

    def test_05_target_gate_cannot_pass_wrong_argmax(self):
        def receipt(correct):
            rows=[dict(stratum=x['stratum'],correct=correct,target_margin=1 if correct else -1) for x in tasks.native_fixture()]
            return {'histories':{'W':{'probes':[{'scores':{'rows':rows}}]*2},'NO_WRITE':{'probes':[{'scores':{'rows':[dict(x,correct=False) for x in rows]}}]*2}}}
        self.assertFalse(nt.gate(config.TRACK_A_STAGES[0],receipt(False))['qualified'])
        self.assertTrue(nt.gate(config.TRACK_A_STAGES[0],receipt(True))['qualified'])
        with self.assertRaises(AssertionError):checked_scores([1.,0.],[48,49],49)
        with self.assertRaises(AssertionError):checked_scores([1.,0.],[48,49],48,49,1.)

    def test_06_recurrent_source_parity_and_probe_purity(self):
        for version in config.TRACK_B_MODELS:
            adapted=models.AdditionReference(version,907);parent=models.PARENT.Recurrent(version,907)
            self.assertEqual(adapted.digest(),parent.digest())
            a=charged(version+'-adapted-W',lambda:adapted.train(b'2+3=',b'5',True))
            b=charged(version+'-parent-W',lambda:parent.train(b'2+3=',b'5',True))
            self.assertEqual(a['loss'],b['loss']);self.assertEqual(adapted.digest(),parent.digest())
            before=adapted.digest()
            charged(version+'-NONZERO_NO_WRITE',lambda:adapted.train(b'2+3=',b'5',False))
            self.assertEqual(before,adapted.digest())
            snapshot=copy.deepcopy(adapted.resource)
            bt.evaluate(adapted,{'tiny':[(2,3),(3,2)]})
            self.assertEqual(snapshot,adapted.resource);self.assertEqual(before,adapted.digest())
            adapted.rest(86400.);self.assertEqual(before,adapted.digest())
            self.assertEqual(sum(p.numel() for p in adapted.p.values()),28440 if version.startswith('KEEP') else 28800)
            self.assertEqual(adapted.resource['vocabulary_rows'],4)

    def test_07_independent_auditor_tamper_rejection(self):
        from audits import audit_native,audit_addition
        # Synthetic output fixtures test the reducer, with zero native records.
        fixture=tasks.native_fixture();order=tasks.ordered_stream(fixture,768,config.TRACK_A_SEED)
        histories={}
        for history in ('W','NO_WRITE'):
            probe=[]
            for x in fixture:
                target=x['answer'][0];correct=target==48
                probe.append(dict(id=x['id'],stratum=x['stratum'],expected_target=target,
                    emitted_target=48,output_universe=list(b'0123'),logits=[1.,0.,0.,0.],
                    target_margin=1. if correct else -1.,correct=correct,autonomous_generation=False))
            histories[history]=dict(rows=[dict(index=i,id=x['id'],allowed=history=='W',
                raw_alpha_l1=0.,forbidden_delta=0.,target=x['answer'][0],predicted=48,
                logits=[1.,0.,0.,0.],output_universe=list(b'0123')) for i,x in enumerate(order,1)],
                births=[dict(fly_id=i,canonical_B_sha256='32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964') for i in range(8)],
                probes=[dict(dose=n,phase=phase,scores={'rows':copy.deepcopy(probe)}) for n,phase in
                         [(n,'IMMEDIATE') for n in config.TRACK_A_DOSES]+[(768,'AFTER_24H')]])
        synthetic=dict(stage=config.TRACK_A_STAGES[0],science=False,histories=histories,gate={'qualified':False})
        self.assertTrue(audit_native(synthetic)['accepted'])
        bad=copy.deepcopy(synthetic);bad['gate']['qualified']=True
        with self.assertRaises(AssertionError):audit_native(bad)
        bad=copy.deepcopy(synthetic);bad['histories']['NO_WRITE']['rows'][0]['allowed']=True
        with self.assertRaises(AssertionError):audit_native(bad)
        bad=copy.deepcopy(synthetic);bad['histories']['W']['probes'][0]['scores']['rows'][0]['target_margin']+=1
        with self.assertRaises(AssertionError):audit_native(bad)
        # Full-size synthetic B validates phase/count accounting before execution.
        world=config.TRACK_B_WORLDS[0];source=tasks.addition_streams(world);hs={}
        for history in config.TRACK_B_HISTORY:
            rows=[]
            for phase,seq in [('OLD',source['shuffled'] if history=='SH_OLD' else source['old']),('NEW',source['new'])]:
                for i,(p,y) in enumerate(seq,1):
                    allowed=phase!='OLD' or history!='N_OLD'
                    rows.append(dict(phase=phase,index=i,prompt=p.hex(),answer=y.hex(),allowed=allowed,
                        actor_steps=int(allowed),forbidden_delta=0,loss=1.))
            scores={k:[dict(prompt=tasks.render(p)[0].hex(),expected=tasks.render(p)[1].hex(),emitted='',status='EOS',correct=False) for p in pairs] for k,pairs in source['groups'].items()}
            curves=[dict(phase=phase,dose=n,scores=scores) for phase,n in
                [('OLD',n) for n in config.TRACK_B_OLD_DOSES]+[('OLD_REST',16384)]+[('NEW',n) for n in config.TRACK_B_NEW_DOSES]+[('NEW_REST',1024)]]
            answers=sum(len(bytes.fromhex(r['answer'])) for r in rows)
            hs[history]=dict(rows=rows,curve=curves,initial_digest='birth',
                OLD_learning_end_digest='birth' if history=='N_OLD' else 'updated',parameter_count=28440,
                resource=dict(records=len(rows),observed_answer_bytes=answers,observed_EOS_targets=len(rows),
                              vocabulary_rows=answers+len(rows),actor_steps=sum(r['allowed'] for r in rows)))
        b=dict(science=False,world=world,seed=tasks.model_seed(world),version='KEEP-C16-CE',histories=hs,record_applications=52224)
        self.assertTrue(audit_addition(b)['accepted'])
        b['histories']['N_OLD']['OLD_learning_end_digest']='updated'
        with self.assertRaises(AssertionError):audit_addition(b)

if __name__=='__main__':
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Qualification)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(bootstrap.HERE/'tests/QUALIFICATION.json',dict(utc=now(),accepted=result.wasSuccessful(),
          tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
          charged_development_record_applications=len(LEDGER),record_ledger=LEDGER,
          elapsed=time.monotonic()-start,science=False,behavior_not_yet_qualified=True))
    sys.exit(0 if result.wasSuccessful() else 1)
