"""Development/synthetic adversarial checks, never scientific sample evidence."""
import unittest,copy,math
import numpy as np
from scipy.stats import t
from runtime import load_plan,DEVELOPMENT_IDS,ROOT,require_sources,require_environment
from protocol import validate_plan,jobs,screen_worlds,confirmation_worlds,check_selection
from selection import choose,validate_metrics
from confirmation import confirm
from statistics_v2 import lower,risk_upper
from locks import manifest,validate_manifest,require_authorized
from dispatcher import decision
from job_runner import valid_job

def synthetic(worlds,arms=None):
    p=load_plan();arms=p['report_configurations'] if arms is None else arms;out={}
    for arm in arms:
        rows=[]
        for i,w in enumerate(worlds):
            v=(i%4-1.5)*.002
            rows.append({'world':w,'old':.94+v,'new':.94+v,'revision':(.65 if arm=='CENTER' else .94)+v,'reuse_W':.7+v,'reuse_N':.5,'taught_old_end':.9,'taught_final':.85,'new_W':.85,'new_N':.85,'N_old':.25,'N_new':.25,'N_revision':.25,'tauL':30+i*.01,'tauR':10+i*.01})
        out[arm]=rows
    if 'P005' in out:
        for i,r in enumerate(out['P005']):r['old']+=.05+(i%3-1)*.001
    return out

class Cycle1(unittest.TestCase):
    def setUp(self):self.p=load_plan();self.ws=screen_worlds(self.p);self.rows=synthetic(self.ws)
    def test_pinned_source(self):require_sources();require_environment();validate_plan(self.p)
    def test_screen_count(self):self.assertEqual(len(jobs(self.p,'screen')),176)
    def test_worst_confirmation(self):
        s=choose(self.rows,self.p,self.ws);s['winner'].update(configuration='REL10',parent='ERROR',target='old',delta=.02);s['confirmation_roster']=self.p['confirmation']['physical_roster_by_winner']['REL10']
        self.assertEqual(len(jobs(self.p,'confirm',s)),480)
    def test_one_target_not_task_grid(self):
        s=choose(self.rows,self.p,self.ws);self.assertEqual((s['winner']['configuration'],s['winner']['target']),('P005','old'));self.assertEqual(len(s['confirmation_roster']),3)
    def test_no_candidate_terminal(self):
        for rs in self.rows.values():
            for r in rs:r['revision']=.94;r['old']=.94
        self.assertIsNone(choose(self.rows,self.p,self.ws)['winner'])
    def test_control_ineligible(self):
        for r in self.rows['S3_RAND']:r['old']=1;r['revision']=1
        self.assertNotEqual(choose(self.rows,self.p,self.ws)['winner']['configuration'],'S3_RAND')
    def test_exact_boundary_harm(self):
        for a in self.rows:
            if a=='CENTER':continue
            for r in self.rows[a]:r['revision']=.94
        self.rows['P005'][0]['old']=self.rows['ERROR'][0]['old']-.05001
        s=choose(self.rows,self.p,self.ws);self.assertEqual(sum(s['guards']['P005']['harms']),1);self.assertNotEqual(s['winner']['configuration'],'P005')
    def test_unpaired_duplicates_rejected(self):
        self.rows['P005'][1]['world']=self.ws[0]
        with self.assertRaises(ValueError):choose(self.rows,self.p,self.ws)
    def test_nonfinite_rejected(self):
        self.rows['P005'][0]['old']=float('nan')
        with self.assertRaises(ValueError):choose(self.rows,self.p,self.ws)
    def test_cpu_not_free(self):
        self.rows['P005'][0]['tauR']=0
        with self.assertRaises(ValueError):choose(self.rows,self.p,self.ws)
    def test_source_manifest_tamper(self):
        m=manifest('qualification',qualification_jobs=[{'world':61005001,'arm':'CENTER','assay':'reuse'}],short=2);m['execution_source_identity']='bad'
        with self.assertRaises(ValueError):validate_manifest(m)
    def test_science_not_authorized(self):
        self.assertFalse((ROOT/'LAUNCH_AUTHORIZATION.json.gz').exists())
        with self.assertRaises((FileNotFoundError,ValueError)):require_authorized()
    def test_no_science_world_in_development(self):
        with self.assertRaises(ValueError):valid_job(self.ws[0],'CENTER','reuse','qualification',short=2)
    def test_short_science_illegal(self):
        with self.assertRaises(ValueError):valid_job(self.ws[0],'CENTER','reuse','screen',short=2)
    def test_alpha_no_redistribution(self):
        p=copy.deepcopy(self.p);p['statistics']['adoption_joint_alpha']=.05
        with self.assertRaises(ValueError):validate_plan(p)
    def test_sample_extension_illegal(self):
        p=copy.deepcopy(self.p);p['confirmation']['n']=64
        with self.assertRaises(ValueError):validate_plan(p)
    def test_student_bound_independent(self):
        x=np.array([.12,.03,.14,-.01,.08,.07,.10,.04]);b=lower(x,.03,(-1,1));self.assertAlmostEqual(b['lower'],x.mean()-t.ppf(.97,7)*x.std(ddof=1)/np.sqrt(8),places=14)
    def test_zero_variance_conservative(self):
        b=lower([.05]*48,.03,(-1,1));self.assertLess(b['lower'],0);self.assertEqual(b['method'],'Hoeffding_registered_range');self.assertIsNone(lower([2.]*48,.03)['lower'])
    def test_harm_zero_exact(self):self.assertAlmostEqual(risk_upper([0]*48),1-.03**(1/48),places=14)
    def test_harm_one_fails(self):self.assertGreater(risk_upper([1]+[0]*47),.10)
    def test_positive_synthetic_confirm(self):
        s=choose(self.rows,self.p,self.ws);cw=confirmation_worlds(self.p);out=confirm(synthetic(cw,s['confirmation_roster']+['Q_HALF']),self.p,s,cw)
        self.assertTrue(out['adoption']['passed']);self.assertEqual(out['verdict'],'CONFIRMED_E1_ENGINEERING_UPGRADE');self.assertFalse(out['mechanism']['registered'])
    def test_no_screen_confirm_mix(self):
        s=choose(self.rows,self.p,self.ws);cw=self.ws+confirmation_worlds(self.p)[:40]
        with self.assertRaises(ValueError):confirm(synthetic(cw,s['confirmation_roster']+['Q_HALF']),self.p,s,cw)
    def test_confirm_harm_rejects(self):
        s=choose(self.rows,self.p,self.ws);cw=confirmation_worlds(self.p);rows=synthetic(cw,s['confirmation_roster']+['Q_HALF']);rows['P005'][0]['old']=.88
        out=confirm(rows,self.p,s,cw);self.assertFalse(out['adoption']['passed']);self.assertFalse(out['adoption']['harm_risk_passed'])
    def test_mechanism_not_inferred_from_adoption(self):
        s=choose(self.rows,self.p,self.ws);s['winner'].update(configuration='S3_CUE',parent='ERROR',target='old',delta=.02);s['confirmation_roster']=self.p['confirmation']['physical_roster_by_winner']['S3_CUE'];cw=confirmation_worlds(self.p)
        rows=synthetic(cw,s['confirmation_roster']+['Q_HALF']);rows['S3_CUE']=copy.deepcopy(rows['P005']) if 'P005' in rows else [dict(r,old=r['old']+.05+(i%3-1)*.001) for i,r in enumerate(rows['S3_CUE'])];rows['S3_RAND']=copy.deepcopy(rows['S3_CUE'])
        out=confirm(rows,self.p,s,cw);self.assertTrue(out['adoption']['passed']);self.assertFalse(out['mechanism']['passed'])
    def test_unavailable_is_not_mechanism_pass(self):
        s=choose(self.rows,self.p,self.ws);s['winner'].update(configuration='S3_CUE',parent='ERROR',target='old',delta=.02);s['confirmation_roster']=self.p['confirmation']['physical_roster_by_winner']['S3_CUE'];cw=confirmation_worlds(self.p)
        rows=synthetic(cw,['CENTER','ERROR','Q_HALF','S3_CUE']);rows['S3_CUE']=[dict(r,old=r['old']+.05+(i%3-1)*.001) for i,r in enumerate(rows['S3_CUE'])]
        out=confirm(rows,self.p,s,cw);self.assertTrue(out['adoption']['passed']);self.assertFalse(out['mechanism']['available']);self.assertFalse(out['mechanism']['passed'])
    def test_changed_target_rejected(self):
        s=choose(self.rows,self.p,self.ws);s['winner']['parent']='CENTER'
        with self.assertRaises(ValueError):check_selection(self.p,s)
    def test_deadline_exact_boundaries(self):
        lim=self.p['session'];self.assertEqual([decision(x,lim) for x in (28799,28800,32400,34200)],['DISPATCH','DRAIN','REQUEST_SAFE_PAUSE','WATCHDOG_STOP'])
    def test_q_no_duplicate_learning(self):self.assertNotIn('Q_HALF',self.p['physical_configurations'])
    def test_rel_controls_fixed(self):self.assertEqual(self.p['confirmation']['physical_roster_by_winner']['REL10'][-2:],['REL_PERM10','FIRST10'])
    def test_revised_old_not_counted_as_intact(self):
        from metrics import row_accuracy
        rows=[{'stage':'old','item':i,'correct':int(i<24),'target':48,'policies':{'Q_HALF':{'emitted':48 if i<24 else 49}}} for i in range(32)]
        p={'rows':rows};self.assertEqual(row_accuracy(p,'old',items=list(range(24))),1.);self.assertEqual(row_accuracy(p,'old',True,items=list(range(24))),1.);self.assertEqual(row_accuracy(p,'old'),.75)
    def test_missing_intact_item_rejected(self):
        from metrics import row_accuracy
        with self.assertRaises(ValueError):row_accuracy({'rows':[{'stage':'old','item':0,'correct':1}]},'old',items=[0,1])
    def test_duplicate_intact_item_rejected(self):
        from metrics import row_accuracy
        with self.assertRaises(ValueError):row_accuracy({'rows':[{'stage':'old','item':0,'correct':1}]*2},'old',items=[0])
    def test_tie_deterministic(self):
        for a in ('P0005','REL05'):
            self.rows[a]=copy.deepcopy(self.rows['P005'])
        self.assertEqual(choose(self.rows,self.p,self.ws)['winner']['configuration'],'P0005')

if __name__=='__main__':unittest.main(verbosity=2)
