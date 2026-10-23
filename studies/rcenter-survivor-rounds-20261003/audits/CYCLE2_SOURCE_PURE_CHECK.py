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
store=SimpleNamespace(brain_t=0.,fly=SimpleNamespace(m=SimpleNamespace(elapsed=0.)),elapsed_base=0.,pending_t=None,pending_x=None,fe=SimpleNamespace(t=12345.),last_byte_t=None)
ns['check_clock'](SimpleNamespace(shared=[store],private=[]),0.)
doc={'schema':'rcenter-cycle2-source-pure-check-v1','reviewer_executed_native_learner':False,'native_outcomes':0,
'design_file':'ROUND1_DESIGN_CYCLE2.md','design_sha256':sha(ROOT/'protocol/ROUND1_DESIGN_CYCLE2.md'),
'source_hashes':source,'source_digest':i.digest_map(source),'frozen_projection_files_verified':len(verified),'python_files_parsed':len(parsed),
'pure_fixture_worlds':worlds,'latin_squares':len(f.latin_squares()),'demonstrated_defects':{'fe_only_clock_desynchronization_accepted_by_checker':True}}
p=ROOT/'audits/CYCLE2_SOURCE_PURE_CHECK.json';p.write_text(json.dumps(doc,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in doc.items() if k not in ('source_hashes','pure_fixture_worlds')}))
