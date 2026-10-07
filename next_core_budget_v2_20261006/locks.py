"""Immutable stage manifests, independent selection audit, explicit launch gate."""
import json,os,re,ast,subprocess
from pathlib import Path
from runtime import ROOT,DOCUMENTS,DEVELOPMENT_IDS,execution_identity,require_sources,source_files,require_environment,read_receipt,write_receipt,atomic_json,file_sha,digest,load_plan
from protocol import validate_plan,jobs,screen_worlds,confirmation_worlds,check_selection,roster
def source_lock(qualification):
    if qualification['verdict']!='V2_FUNCTIONAL_AND_RESOURCE_QUALIFIED' or qualification['execution_source_identity']!=execution_identity():raise ValueError('qualification not accepted/current')
    r=ROOT/'WORLD_EXCLUSION_REGISTRY.json'
    verify_registry(json.loads(r.read_text()))
    d={'execution_source_identity':execution_identity(),'scientific_source_identity':qualification['scientific_source_identity'],'files':source_files(),'runtime':require_environment(),'qualification_sha256':file_sha(ROOT/'results/QUALIFICATION.json'),'world_registry_sha256':file_sha(r),'science_authorized':False}
    atomic_json(ROOT/'SOURCE_LOCK.json',d);return d
def require_qualified():
    d=json.loads((ROOT/'SOURCE_LOCK.json').read_text());require_sources(d['execution_source_identity'])
    if d['files']!=source_files() or d['qualification_sha256']!=file_sha(ROOT/'results/QUALIFICATION.json') or d['world_registry_sha256']!=file_sha(ROOT/'WORLD_EXCLUSION_REGISTRY.json'):raise ValueError('stale source/qualification/registry lock')
    q=json.loads((ROOT/'results/QUALIFICATION.json').read_text())
    if q['verdict']!='V2_FUNCTIONAL_AND_RESOURCE_QUALIFIED':raise ValueError('not qualified')
    verify_registry(json.loads((ROOT/'WORLD_EXCLUSION_REGISTRY.json').read_text()));return d
def require_authorized():
    q=require_qualified();a=read_receipt(ROOT/'LAUNCH_AUTHORIZATION.json.gz')
    if a!={'authorized':True,'execution_source_identity':q['execution_source_identity'],'plan_sha256':file_sha(ROOT/'PLAN.json'),'purpose':'authorized screen then at most one confirmation'}:raise ValueError('launch not authorized')
def manifest(stage,*,selection=None,qualification_jobs=None,short=None,workers=8,session=None):
    p=validate_plan(load_plan());ident=execution_identity()
    if stage=='qualification':
        if not qualification_jobs:raise ValueError('empty qualification roster')
        rs=[dict(j,stage=stage) for j in qualification_jobs]
        if any(j['world'] not in DEVELOPMENT_IDS for j in rs):raise ValueError('qualification science world')
        timing=session or p['session']
    else:
        require_authorized()
        if stage not in ('screen','confirm') or short is not None or workers!=8 or session is not None:raise ValueError('science operational contract changed')
        if stage=='confirm':
            if selection is None:raise ValueError('missing selection')
            check_selection(p,selection)
        rs=jobs(p,stage,selection);timing=p['session']
    return {'schema':'BUDGET_STAGE_MANIFEST_V2','stage':stage,'execution_source_identity':ident,'plan_sha256':file_sha(ROOT/'PLAN.json'),'jobs':rs,'short':short,'workers':workers,'session':timing,'selection':selection,'selection_digest':None if selection is None else digest(selection)}
def validate_manifest(m,science=False):
    if m['schema']!='BUDGET_STAGE_MANIFEST_V2' or m['execution_source_identity']!=execution_identity() or m['plan_sha256']!=file_sha(ROOT/'PLAN.json'):raise ValueError('manifest source/spec')
    if len({(j['world'],j['arm'],j['assay']) for j in m['jobs']})!=len(m['jobs']):raise ValueError('duplicate jobs')
    if m['stage']=='qualification':
        if science or any(j['world'] not in DEVELOPMENT_IDS or j['stage']!='qualification' for j in m['jobs']):raise ValueError('qualification/science mismatch')
        if m['workers'] not in (4,8):raise ValueError('qualification worker count')
    else:
        require_authorized();p=validate_plan(load_plan())
        if m['stage'] not in ('screen','confirm') or m['short'] is not None or m['workers']!=8 or m['session']!=p['session']:raise ValueError('science stage/cap')
        if m['jobs']!=jobs(p,m['stage'],m['selection']):raise ValueError('changed science roster')
        if m['stage']=='confirm':
            s=read_receipt(ROOT/'SELECTION_LOCK.json.gz')
            if s!=m['selection'] or digest(s)!=m['selection_digest']:raise ValueError('unlocked/changed confirmation target')
            verify_selection_lock(s)
    return m
def write_manifest(path,m):validate_manifest(m);write_receipt(path,m);return path
def seal_selection(s,evidence):
    from selection import choose
    from metrics import collect
    p=load_plan();worlds=screen_worlds(p);rows,files=collect(ROOT/'science/screen',worlds,p['physical_configurations'],'screen')
    rebuilt=choose(rows,p,worlds)
    if rebuilt!=s or files!=evidence:raise ValueError('selection independent reducer mismatch')
    s=dict(s,execution_source_identity=execution_identity(),plan_sha256=file_sha(ROOT/'PLAN.json'),screen_receipts=evidence)
    path=ROOT/'SELECTION_LOCK.json.gz'
    if path.exists():
        if read_receipt(path)!=s:raise ValueError('selection already frozen differently')
    else:write_receipt(path,s)
    return s
def verify_selection_lock(s):
    from selection import choose
    from metrics import collect
    p=load_plan();rows,evidence=collect(ROOT/'science/screen',screen_worlds(p),p['physical_configurations'],'screen')
    rebuilt=choose(rows,p,screen_worlds(p))
    if any(s[k]!=v for k,v in rebuilt.items()) or s['screen_receipts']!=evidence or s['execution_source_identity']!=execution_identity() or s['plan_sha256']!=file_sha(ROOT/'PLAN.json'):raise ValueError('selection lock evidence altered')
def verify_registry(r):
    p=load_plan()
    if r['proposed_screen']!=screen_worlds(p) or r['proposed_confirmation']!=confirmation_worlds(p) or r['development']!=list(DEVELOPMENT_IDS):raise ValueError('registry phase roster')
    proposed=set(screen_worlds(p)+confirmation_worlds(p))
    if proposed & set(r['historical_exclusion_worlds']):raise ValueError('historical world collision')
    if not r['verified_accessible_local_history'] or not r['scan_roots']:raise ValueError('world census incomplete')
def exclusion_registry():
    # Inventory names and declared source/lock world rosters only; no efficacy selection.
    roots=[ROOT.parents[1],Path('/Users/pencilbard/Documents/second_try')]
    historic=set();evidence=[];files_examined=0
    def extract(obj,key=''):
        if isinstance(obj,dict):
            for k,v in obj.items():extract(v,k.lower())
        elif isinstance(obj,list):
            if any(s in key for s in ('world','seed')):
                for v in obj:
                    if type(v) is int and v>=0:historic.add(v)
            else:
                for v in obj:extract(v,key)
        elif type(obj) is int and key in ('world','world_id','seed','base_seed'):historic.add(obj)
    def literal(node):
        if isinstance(node,ast.Constant) and type(node.value) is int:return node.value
        if isinstance(node,(ast.List,ast.Tuple)):return [literal(x) for x in node.elts]
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            args=[literal(x) for x in node.args]
            if node.func.id=='range' and all(type(x) is int for x in args):
                r=range(*args)
                if len(r)<=10000:return list(r)
            if node.func.id in ('list','tuple') and len(args)==1:return args[0]
        raise ValueError('not a static world list')
    for root in roots:
        cmd=['rg','--files','--hidden','-g','!.git/**','-g','!**/__pycache__/**','-g','!**/scratch/**',str(root)]
        run=subprocess.run(cmd,capture_output=True,text=True)
        if run.returncode not in (0,1):raise RuntimeError('history inventory failed')
        for name in run.stdout.splitlines():
            path=Path(name)
            if ROOT in path.parents or DOCUMENTS in path.parents:continue
            files_examined+=1
            if 'receipt' in name.lower() or 'checkpoint' in name.lower():
                nums=re.findall(r'(?<!\d)(\d{3,10})(?=[_.])',path.name)
                if nums:historic.update(map(int,nums));evidence.append({'path':str(path),'kind':'receipt/checkpoint filename','worlds':list(map(int,nums))})
            if path.suffix=='.py' and path.stat().st_size<2*1024**2:
                text=path.read_text(errors='replace')
                if 'WORLD' not in text.upper():continue
                try:tree=ast.parse(text)
                except SyntaxError:continue
                for n in tree.body:
                    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and 'WORLD' in t.id.upper() for t in n.targets):
                        try:vals=literal(n.value)
                        except (ValueError,TypeError):continue
                        if type(vals) is int:vals=[vals]
                        if isinstance(vals,list) and all(type(x) is int for x in vals):historic.update(vals);evidence.append({'path':str(path),'kind':'static declared world roster','worlds':vals})
            if path.suffix=='.json' and any(s in path.name.upper() for s in ('LOCK','LAUNCH','STATUS','MANIFEST','REGISTRY','QUALIFICATION')) and path.stat().st_size<2*1024**2:
                try:extract(json.loads(path.read_text()))
                except (ValueError,OSError):continue
    r={'schema':'WORLD_EXCLUSION_REGISTRY_V2','scan_roots':list(map(str,roots)),'files_examined':files_examined,'verified_accessible_local_history':True,'scope':'Accessible local project history, including downloaded remote archives; unavailable remote runs not certified. Design-only documents are not execution evidence.','historical_exclusion_worlds':sorted(historic),'proposed_screen':screen_worlds(load_plan()),'proposed_confirmation':confirmation_worlds(load_plan()),'development':list(DEVELOPMENT_IDS),'evidence':evidence}
    verify_registry(r);atomic_json(ROOT/'WORLD_EXCLUSION_REGISTRY.json',r);return r
