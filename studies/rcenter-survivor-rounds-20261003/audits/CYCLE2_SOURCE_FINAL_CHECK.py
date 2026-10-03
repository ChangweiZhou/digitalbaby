"""Independent source/fixture/helper checks; no learner/vendor import or birth."""
import ast, importlib.util, hashlib, json
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
i=load('cycle2_audit_integrity',ROOT/'source/runtime/integrity.py')
f=load('cycle2_audit_fixture',ROOT/'source/runtime/survivor_fixture.py')
source=i.hashes()
manifest=json.loads((ROOT/'source/BASELINE_PROJECTION_MANIFEST.json').read_text())
lookup={v['path']:v for v in manifest['members']}
public=json.loads((ROOT/'source/baseline/PUBLICATION_MANIFEST.json').read_text())['files']
verified=[]
for path,actual in source.items():
 if path.startswith('source/baseline/'):
  rel=path[len('source/baseline/'):];assert actual==lookup[rel]['sha256']
  assert (ROOT/path).stat().st_size==lookup[rel]['size']
  assert public[rel]['sha256']==actual
  verified.append(path)
parsed=[]
for path in source:
 if path.endswith('.py'):ast.parse((ROOT/path).read_text(),filename=path);parsed.append(path)
worlds=[]
for seed in (310000,310101,310102,310200,*f.OFFICIAL_WORLDS):
 w=f.make_world(seed);sets=w['sets'];events=w['events'];assert len(events)==384
 assert [e['stage'] for e in events]==['old']*192+['new']*192
 held=set(sets['heldout']['cues']);assert not held & {e['cue_hex'] for e in events}
 for name,n,repeat in [('old',12,16),('new',16,12)]:
  assert len(sets[name]['cues'])==n
  assert all(sum(e['cue_hex']==ch for e in events)==repeat for ch in sets[name]['cues'])
  assert all(sum(e['outcome']==v for e in events if e['stage']==name)==48 for v in f.ALPHABET)
 assert all(len(bytes.fromhex(e['cue_hex']))==12 for e in events)
 assert w['identifiability']['heldout_unique'] and w['identifiability']['consistent_tables']==1
 assert [w[k] for k in ('old_end','new_start','new_end','final')]==[31680.,118080.,149760.,236160.]
 worlds.append({'seed':seed,'fixture_sha256':w['sha256']})
# Extract only the pure clock checker, passing synthetic namespaces, never native state.
tree=ast.parse((ROOT/'tests/cycle2_adversarial.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_clock')
ns={'clock_vector':lambda m: []};exec(compile(ast.Module(body=[node],type_ignores=[]),'isolated_check_clock','exec'),ns)
for field in ('brain_t','elapsed','fe_t'):
 store=SimpleNamespace(brain_t=0.,fly=SimpleNamespace(m=SimpleNamespace(elapsed=0.)),elapsed_base=0.,pending_t=None,pending_x=None,fe=SimpleNamespace(t=0.),last_byte_t=None,bytes_seen=0)
 if field=='elapsed':store.fly.m.elapsed=1.
 elif field=='fe_t':store.fe.t=1.
 else:store.brain_t=1.
 try:ns['check_clock'](SimpleNamespace(shared=[store],private=[]),0.)
 except AssertionError:pass
 else:raise AssertionError(field+' corruption accepted')
# Pure evaluator/helper tests using synthetic objects; no Learner or vendor import.
import copy,sys,numpy as np
sys.path.insert(0,str(ROOT/'source/runtime'))
import assay
class SyntheticSystem:
 def __init__(self):
  native=SimpleNamespace(fast=np.zeros((2,2)),slow=np.zeros(2),adapt=np.ones(2),elapsed=0.,event_count=0,presentation_count=0,B=np.eye(2),kernel={'tau':2.})
  fly=SimpleNamespace(m=native,reader=SimpleNamespace(Q=np.eye(2),head={'intercept':np.zeros(1)}),stats={'ignored':0})
  fe=SimpleNamespace(p=np.zeros(2),t=0.,tau=1.,visible=b'')
  store=SimpleNamespace(fly=fly,fe=fe,pool_of_kc=np.zeros(2),pair_hash=np.zeros((2,2)),w=np.zeros((2,4)),bias=np.zeros(4),seed=1,temporal=False,order_via_native=False,lr=.03,bias_lr=.001,brain_t=0.,elapsed_base=0.,pending_t=None,pending_x=None,last_byte_t=None,bytes_seen=0,teach_seen=0,n_native_kc=2)
  self.shared=[store];self.private=[];self.arm='R_center';self.scales=(1.4911274663291492,1.3452365735750882);self.cached=None;self.prediction_time=None;self.cue_count=0;self.audit=[]
 def digests(self):return ['synthetic-only']
 def clone(self):return copy.deepcopy(self)
 def feed(self,b,t):self.cue_count+=1
 def predict(self,t):return {'emitted':48,'shared':[0.,0.,0.,0.],'private':[0.,0.,0.,0.],'combined':[0.,0.,0.,0.]}
base=SyntheticSystem();full=assay.operative_state_identity(base);fixed=assay.fixed_identity(base)
for field in ('tau','arm','scales','reader','connectome','predictor'):
 a=copy.deepcopy(base)
 if field=='tau':a.shared[0].fe.tau+=1
 elif field=='arm':a.arm='altered'
 elif field=='scales':a.scales=(2.,2.)
 elif field=='reader':a.shared[0].fly.reader.head['intercept'][0]+=1
 elif field=='connectome':a.shared[0].fly.m.B[0,0]+=1
 else:a.shared[0].w[0,0]+=1
 assert assay.operative_state_identity(a)!=full and assay.fixed_identity(a)!=fixed
 assert assay.operative_state_identity(base)==full
pred=base.predict(0.);before=copy.deepcopy(pred);policies=assay.scores(pred,base.scales);assert pred==before
assert all(q['emitted']==48 and q['ties']==[0,1,2,3] for q in policies.values())
for target in (48,49):
 world={'sets':{'heldout':{'cues':[(b' '*8+b'0+0=').hex()],'outcomes':[target]}}}
 result=assay.probe(base,world,0.,('heldout',));assert result['state_before']==result['state_after']==full
 row=result['rows'][0];assert all(q['correct']==int(q['emitted']==target) for q in row['policies'].values())
assert not any(name in sys.modules for name in ('engine','survivor_frozen_learner','brain_byte','stores'))
# Frozen wrapper attributes are accounted for; birth records are provenance, predict binding is intentional policy treatment.
learner_tree=ast.parse((ROOT/'source/baseline/src/learner.py').read_text())
assigned={n.attr for n in ast.walk(learner_tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and isinstance(n.ctx,ast.Store)}
assert assigned=={'arm','scales','shared','private','births','cached','prediction_time','cue_count','audit'}
assert i.hashes()==source, 'source changed during final pure checks'
doc={'schema':'rcenter-cycle2-source-final-check-v1','reviewer_executed_native_learner':False,'native_outcomes':0,
'design_file':'ROUND1_DESIGN_CYCLE2.md','design_sha256':sha(ROOT/'protocol/ROUND1_DESIGN_CYCLE2.md'),
'source_hashes':source,'source_digest':i.digest_map(source),'frozen_projection_files_verified':len(verified),'python_files_parsed':len(parsed),
'pure_fixture_worlds':worlds,'latin_squares':len(f.latin_squares()),'pure_helper_checks':{'brain_elapsed_fe_corruption_rejected':True,'operative_and_fixed_mutation_sensitivity':True,'nonmutating_alpha_scoring_and_ties':True,'actual_probe_target_scoring_and_parent_guards':True,'wrapper_fields_accounted_for':sorted(assigned),'no_native_imports':True}}
p=ROOT/'audits/CYCLE2_SOURCE_FINAL_CHECK.json';p.write_text(json.dumps(doc,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in doc.items() if k not in ('source_hashes','pure_fixture_worlds')}))
