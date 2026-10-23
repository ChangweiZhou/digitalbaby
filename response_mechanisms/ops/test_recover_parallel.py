"""Isolated stdlib mocks only. Never imports locked science or touches real results."""
import contextlib
import copy
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('parallel_candidate', HERE / 'recover_parallel.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

STUB = r'''
import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]); w=int(sys.argv[2]); a=sys.argv[3]
config=json.loads((root/'stub_config.json').read_text()); key=f'{w}/{a}'
conf=config.get(key,{})
print('attempt started',w,a,flush=True)
print('failure evidence preserved',file=sys.stderr,flush=True)
assert all(os.getenv(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'))
(root/f'start-{w}-{a}').write_text(str(time.time()))
time.sleep(conf.get('sleep',.04))
if conf.get('exit'): sys.exit(conf['exit'])
doc=dict(world=w,arm=a,data='deterministic',valid=not conf.get('invalid',False),
         resource=dict(process_id=os.getpid(),wall_seconds=conf.get('reported_seconds',.04),
                       peak_rss_bytes=conf.get('reported_rss',100)))
target=root/'results/final'/a/f'{w}.json.gz';target.parent.mkdir(parents=True,exist_ok=True)
with target.open('x') as f: json.dump(doc,f)
(root/f'end-{w}-{a}').write_text(str(time.time()))
'''


class MockScience:
    def __init__(self, root, *, worlds=(1,2), arms=('A','B'), life=3., replays=()):
        self.root=root
        self.lock=dict(worlds=list(worlds),arms=list(arms),bouts=6,
            replay_jobs=[list(x) for x in replays], resource_caps=dict(workers=1,worker_hours=8.,
            per_life_seconds=life,peak_rss_bytes=100000000,results_bytes=2000000,active_session_hours=12.))
        self.verify_count=0
        self.commands=[]
    def verify(self):
        self.verify_count+=1
    def command(self,w,a):
        self.commands.append((w,a))
        return [p.PYTHON,str(self.root/'stub.py'),str(self.root),str(w),a]
    def checked(self,path,w,a):
        d=json.loads(path.read_text())
        assert d['world']==w and d['arm']==a and d['valid']
        assert d['resource']['wall_seconds']<=self.lock['resource_caps']['per_life_seconds']
        assert d['resource']['peak_rss_bytes']<=self.lock['resource_caps']['peak_rss_bytes']
        return d


def base_history(start):
    return dict(lock_sha256=p.LOCK_SHA,started_unix=start,worker_seconds=10.,
        reset_recovery_accounting_applied=True,active_job=None,
        jobs=[dict(seconds=10.,reason='preserved prior charge',accounting_only=True)])


class Fixture:
    def __init__(self, *, config=None, **science_args):
        self.tmp=tempfile.TemporaryDirectory(prefix='response-parallel-mock-')
        self.root=Path(self.tmp.name)
        assert p.ROOT not in self.root.parents and self.root != p.ROOT
        self.dest=self.root/'results/final';self.dest.mkdir(parents=True)
        self.start=time.time()-30
        p.atomic(self.dest/'RUN_LEDGER.json',base_history(self.start))
        (self.root/'stub.py').write_text(STUB)
        p.atomic(self.root/'stub_config.json',config or {})
        self.science=MockScience(self.root,**science_args)
        p.atomic(self.dest/'REPLAY_AUDIT.json',dict(pass_all=True,
            jobs=self.science.lock['replay_jobs'],excluded_fields=['resource']))
        self.guard=p.acquire_guard(self.root,create=True)
    def coordinator(self, **kw):
        return p.Coordinator(self.root,self.science,self.guard,evidence={'mock':True},start=self.start,
            minimum=0.,deadline=time.time()+20.,poll_seconds=.005,stop_margin=.025,**kw)
    def close(self):
        self.guard.close();self.tmp.cleanup()
    def receipt(self,w,a,*,replay=False,pid=100):
        path=self.dest/('replays' if replay else '')/a/f'{w}.json.gz'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(dict(world=w,arm=a,data='deterministic',valid=True,
             resource=dict(process_id=pid,wall_seconds=.01,peak_rss_bytes=100))))
        return path
    def history(self):return json.loads((self.dest/'RUN_LEDGER.json').read_text())
    def status(self):return json.loads((self.dest/'RUN_STATUS.json').read_text())


class AccountingTests(unittest.TestCase):
    def test_legacy_import_preserves_start_and_all_prior_charges(self):
        h=base_history(100)
        h['active_job']=dict(world=1,arm='A',replay=False,started_unix=150,stdout='saved-log')
        original=copy.deepcopy(h)
        r=p.reconcile(h,200,start=100,minimum=0)
        self.assertEqual(h,original)
        self.assertEqual(r['jobs'][0],h['jobs'][0])
        self.assertEqual(r['jobs'][1]['started_unix'],150)
        self.assertEqual(r['jobs'][1]['stdout'],'saved-log')
        self.assertEqual(r['worker_seconds'],60)
        self.assertEqual(r['started_unix'],100)
        self.assertIsNone(r['active_job']);self.assertEqual(r['active_jobs'],[])
    def test_both_interrupted_children_charged_without_clamping(self):
        h=base_history(100);h['active_jobs']=[dict(world=1,arm=a,replay=False,started_unix=t,
            reserved_seconds=300) for a,t in [('A',150),('B',175)]]
        r=p.reconcile(h,600,start=100,minimum=0)
        self.assertEqual(r['worker_seconds'],885)
        self.assertEqual([x['seconds'] for x in r['jobs']],[10,450,425])
        r2=p.reconcile(r,800,start=100,minimum=0)
        self.assertEqual(r2['worker_seconds'],885)
    def test_persisted_elapsed_lower_bound_survives_clock_change(self):
        h=base_history(100);h['active_jobs']=[dict(world=1,arm='A',started_unix=150,
            elapsed_observed_seconds=75)]
        self.assertEqual(p.reconcile(h,160,start=100,minimum=0)['worker_seconds'],85)
    def test_ambiguous_dual_schema_rejected(self):
        h=base_history(100);h['active_job']={'world':1};h['active_jobs']=[{'world':2}]
        with self.assertRaises(RuntimeError):p.reconcile(h,200,start=100,minimum=0)
    def test_duplicate_reservations_rejected(self):
        h=base_history(100);h['active_jobs']=[dict(world=1,arm='A',started_unix=150)]*2
        with self.assertRaises(AssertionError):p.reconcile(h,200,start=100,minimum=0)
    def test_ledger_total_and_original_start_cannot_reset(self):
        h=base_history(100);h['worker_seconds']=0
        with self.assertRaises(AssertionError):p.reconcile(h,200,start=100,minimum=0)
        with self.assertRaises(AssertionError):p.reconcile(base_history(101),200,start=100,minimum=0)


class OperationalTests(unittest.TestCase):
    def run_fixture(self,f,**kw):
        c=f.coordinator(**kw)
        with contextlib.redirect_stdout(io.StringIO()):
            result=c.run()
        return c,result
    def test_two_workers_fixed_missing_jobs_and_exact_prefix_queues(self):
        f=Fixture(config={'1/A':{'sleep':.13},'1/B':{'sleep':.04},'2/A':{'sleep':.12}})
        try:
            c,ok=self.run_fixture(f)
            self.assertTrue(ok);self.assertEqual(f.science.commands,[(1,'A'),(1,'B'),(2,'A'),(2,'B')])
            events=[]
            for w,a in f.science.commands:
                events += [(float((f.root/f'start-{w}-{a}').read_text()),1),
                           (float((f.root/f'end-{w}-{a}').read_text()),-1)]
            n=peak=0
            for _,change in sorted(events):n+=change;peak=max(peak,n)
            self.assertEqual(peak,2)
            self.assertEqual(f.status()['state'],'complete_pending_final_audit')
            self.assertEqual(f.history()['active_jobs'],[])
            self.assertIsNone(f.history()['active_job'])
            self.assertEqual(len(f.history()['jobs']),5)
            for w in (1,2):self.assertEqual(json.loads((f.dest/'publication_queue'/f'{w}.json').read_text())['completed'],w*2)
            self.assertEqual(c.science.verify_count,5)
        finally:f.close()
    def test_existing_receipts_and_replays_preserved_byte_for_byte(self):
        f=Fixture(replays=((1,'A'),))
        try:
            first=f.receipt(1,'A');replay=f.receipt(1,'A',replay=True,pid=101)
            before={str(x):x.read_bytes() for x in (first,replay)}
            c,ok=self.run_fixture(f)
            self.assertTrue(ok);self.assertNotIn((1,'A'),f.science.commands)
            self.assertEqual(before,{str(x):x.read_bytes() for x in (first,replay)})
        finally:f.close()
    def test_failure_stops_and_accounts_for_other_worker_preserves_logs(self):
        f=Fixture(config={'1/A':{'sleep':.05,'exit':7},'1/B':{'sleep':1}})
        try:
            with self.assertRaises(RuntimeError):self.run_fixture(f)
            h=f.history();self.assertEqual(len(h['jobs']),3);self.assertEqual(h['active_jobs'],[])
            self.assertIn('terminal_failure',h);self.assertEqual(f.status()['state'],'technical_stop')
            self.assertEqual(f.science.commands,[(1,'A'),(1,'B')])
            for job in h['jobs'][1:]:
                self.assertGreater(job['seconds'],0)
                self.assertTrue((f.dest/job['stdout']).exists());self.assertTrue((f.dest/job['stderr']).exists())
            self.assertFalse((f.dest/'publication_queue'/'1.json').exists())
            self.assertEqual(len(list((f.dest/'operations').glob('PRE_PARALLEL_LEDGER-*'))),1)
        finally:f.close()
    def test_nonzero_exit_during_other_receipt_validation_blocks_replacement(self):
        f=Fixture(config={'1/A':{'sleep':.02},'1/B':{'sleep':.05,'exit':7},'2/A':{'sleep':1}})
        try:
            original=f.science.checked
            def checked(path,world,arm):
                if (world,arm)==(1,'A'):time.sleep(.2)
                return original(path,world,arm)
            f.science.checked=checked
            with self.assertRaises(RuntimeError):self.run_fixture(f)
            self.assertEqual(f.science.commands,[(1,'A'),(1,'B')])
            self.assertFalse((f.root/'start-2-A').exists())
        finally:f.close()
    def test_invalid_receipt_completed_during_validation_is_drained_before_admission(self):
        f=Fixture(config={'1/A':{'sleep':.02},'1/B':{'sleep':.05,'invalid':True}})
        try:
            original=f.science.checked
            def checked(path,world,arm):
                if (world,arm)==(1,'A'):time.sleep(.2)
                return original(path,world,arm)
            f.science.checked=checked
            with self.assertRaises(RuntimeError):self.run_fixture(f)
            self.assertEqual(f.science.commands,[(1,'A'),(1,'B')])
        finally:f.close()
    def test_failure_during_slow_source_verification_blocks_spawn(self):
        f=Fixture(config={'1/A':{'sleep':.02},'1/B':{'sleep':.14,'exit':7}})
        try:
            original=f.science.verify
            def verify():
                original()
                if f.science.verify_count==3:time.sleep(.3)
            f.science.verify=verify
            with self.assertRaises(RuntimeError):self.run_fixture(f)
            self.assertEqual(f.science.commands,[(1,'A'),(1,'B')])
        finally:f.close()
    def test_failure_during_reservation_fsync_gap_blocks_spawn(self):
        f=Fixture(config={'1/A':{'sleep':.02},'1/B':{'sleep':.14,'exit':7}})
        try:
            c=f.coordinator();original=c.save
            def save():
                original()
                if any(x.record['world']==2 and x.process is None for x in c.live.values()):time.sleep(.3)
            c.save=save
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):c.run()
            self.assertEqual(f.science.commands,[(1,'A'),(1,'B')])
            self.assertFalse((f.root/'start-2-A').exists())
            self.assertEqual(f.history()['active_jobs'],[])
        finally:f.close()
    def test_nonzero_exit_direct_poll_blocks_even_without_monitor_flag(self):
        f=Fixture()
        try:
            c=f.coordinator();record=dict(world=1,arm='A',started_unix=time.time(),reserved_seconds=3.)
            live=p.Live(record,time.monotonic())
            live.process=subprocess.Popen([p.PYTHON,'-c','raise SystemExit(7)'])
            live.process.wait();self.assertIsNone(live.reason)
            c.live['stub-failed']=live
            with self.assertRaises(RuntimeError):c.check_child_failures()
            self.assertIn('nonzero child exit',live.reason)
            self.assertEqual(f.science.commands,[])
        finally:f.close()
    def test_invalid_receipt_is_never_counted_successful(self):
        f=Fixture(config={'1/A':{'invalid':True}})
        try:
            with self.assertRaises(RuntimeError):self.run_fixture(f)
            self.assertLess(f.status()['completed'],4)
            self.assertTrue(any('validation failure' in (j.get('reason') or '') for j in f.history()['jobs']))
        finally:f.close()
    def test_receipt_reported_resource_cap_is_post_exit_failure(self):
        f=Fixture(config={'1/A':{'reported_rss':200000000}})
        try:
            with self.assertRaises(RuntimeError):self.run_fixture(f)
            self.assertEqual(f.status()['state'],'technical_stop')
        finally:f.close()
    def test_per_life_watchdog_stops_child(self):
        f=Fixture(life=.18,config={'1/A':{'sleep':1},'1/B':{'sleep':1}})
        try:
            before=time.monotonic()
            with self.assertRaises(RuntimeError):self.run_fixture(f)
            self.assertLess(time.monotonic()-before,.7)
            self.assertTrue(any('time cap' in (j.get('reason') or '') for j in f.history()['jobs']))
        finally:f.close()
    def test_reservation_prevents_budget_overcommit(self):
        f=Fixture(life=1.,config={'1/A':{'sleep':.4}})
        try:
            c,ok=self.run_fixture(f,worker_cap=11.2)
            self.assertFalse(ok);self.assertEqual(f.science.commands,[(1,'A')])
            self.assertLess(f.history()['worker_seconds'],11.2)
            self.assertEqual(f.status()['state'],'budget_stop')
        finally:f.close()
    def test_deadline_checks_before_admission(self):
        f=Fixture()
        try:
            c=f.coordinator();c.deadline=time.time()-1
            with self.assertRaises(RuntimeError):c.run()
            self.assertEqual(f.science.commands,[])
        finally:f.close()
    def test_extra_receipt_stops_before_simulation(self):
        f=Fixture()
        try:
            f.receipt(999,'A')
            with self.assertRaises(AssertionError):self.run_fixture(f)
            self.assertEqual(f.science.commands,[])
        finally:f.close()
    def test_lock_inherited_by_each_child_after_supervisor_fd_closes(self):
        f=Fixture()
        children=[]
        try:
            for delay in (.1,.35):
                children.append(subprocess.Popen([p.PYTHON,'-c',f'import time;time.sleep({delay})'],
                    pass_fds=(f.guard.fileno(),)))
            f.guard.close()
            with self.assertRaises(RuntimeError):p.acquire_guard(f.root)
            children[0].wait()
            self.assertIsNone(children[1].poll())
            with self.assertRaises(RuntimeError):p.acquire_guard(f.root)
            children[1].wait()
            released=p.acquire_guard(f.root);released.close()
        finally:
            for child in children:
                if child.poll() is None:child.kill();child.wait()
            f.close()
    def test_old_supervisor_blocker_and_no_ledger_change(self):
        f=Fixture()
        try:
            before=(f.dest/'RUN_LEDGER.json').read_bytes()
            with self.assertRaises(RuntimeError):p.acquire_guard(f.root)
            self.assertEqual(before,(f.dest/'RUN_LEDGER.json').read_bytes())
        finally:f.close()
    def test_publication_prefix_waits_for_earlier_world(self):
        f=Fixture()
        try:
            c=f.coordinator();c.done={(2,'A'),(2,'B')};c.queue_worlds()
            self.assertFalse((f.dest/'publication_queue').exists())
            c.done|={(1,'A'),(1,'B')};c.queue_worlds()
            self.assertEqual(json.loads((f.dest/'publication_queue'/'2.json').read_text())['completed'],4)
        finally:f.close()
    def test_missing_authorization_and_wrong_source_review_rejected(self):
        f=Fixture()
        try:
            approval=f.root/'approval.json';review=f.root/'review.json'
            p.atomic(approval,dict(approved=True));p.atomic(review,dict(pass_all=True))
            with self.assertRaises(AssertionError):p.authorization(approval,review)
        finally:f.close()


if __name__=='__main__':unittest.main(verbosity=2)
