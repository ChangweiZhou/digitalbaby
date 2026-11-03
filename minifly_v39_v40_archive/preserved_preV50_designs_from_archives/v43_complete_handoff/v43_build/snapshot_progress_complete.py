#!/usr/bin/env python3
"""Capture a V43 resume ZIP with the verified completion driver and diagnostics fix.

python3 snapshot_progress_complete.py --out-dir v43_final_fb8ebb_results --output v43_progress.zip
The helper never unpickles a checkpoint or calls a learner. It does not write into
--out-dir. Snapshot times may differ between independent seed/load trajectories.
"""
from __future__ import annotations
import argparse, ast, base64, hashlib, io, json, os, struct, sys, time, types, uuid, zipfile, zlib
from pathlib import Path

FORMAT = 'V43-live-complete-resume-snapshot-1'
BASE_HELPER_SHA256 = '0909335fa27ed9788254f8735633a55a11ca3b9be5329f609c08f3e4878f6d55'
COMPLETION_SHA256 = '456daa818b500a6d9ad128e52b43451626a691f674c7141d018cc605071d4b2d'
CORE_SHA256 = 'fb8ebb5355ea03e79d6049e5498db60772b18aa2bd51a54beed0689266a77bd8'
CORRECTED_DIAGNOSTICS_SHA256 = '98f88ef5b5f17efe0982f0876faae599a76d4e85fff22fde84e507c9bfd133f9'

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(x).hexdigest()
def json_bytes(x):return json.dumps(x,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n'
def read_once(path):
    try:return Path(path).read_bytes()
    except FileNotFoundError:return None

def checkpoint_header(raw, expected):
    if len(raw)<8:raise ValueError('Truncated checkpoint')
    n=struct.unpack('>Q',raw[:8])[0]
    if n>len(raw)-8:raise ValueError('Truncated checkpoint header')
    header=json.loads(raw[8:8+n]);payload=raw[8+n:]
    if header.get('format')!='V43-pickle5-zlib':raise ValueError('Unknown checkpoint format')
    if header.get('identity')!=expected:raise ValueError('Checkpoint source/plan/job identity mismatch')
    if header.get('sha256')!=digest(payload):raise ValueError('Checkpoint payload checksum mismatch')
    return {'identity':expected,'payload_sha256':header['sha256'],'payload_bytes':len(payload),'unpickled':False}

def verify_zip(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError('Snapshot ZIP CRC failure')
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP member')
        hashes=json.loads(z.read('SNAPSHOT_HASHES.json'))
        if set(names)!=set(hashes)|{'SNAPSHOT_HASHES.json'}:raise ValueError('Snapshot manifest coverage mismatch')
        for name,want in hashes.items():
            if digest(z.read(name))!=want:raise ValueError('Snapshot member checksum mismatch: '+name)
    return {'members':len(names),'exact_manifest_coverage':True,'CRC_passed':True,'member_hashes_passed':True}

def source_plan(source, raw, mode):
    # Execute only trusted frozen runner definitions. main(), dependency setup,
    # module payload loading and every learner function remain uncalled.
    module=types.ModuleType('v43_snapshot_source_definitions')
    module.__file__=str(source);module.__dict__['__name__']=module.__name__
    exec(compile(raw,str(source),'exec'),module.__dict__)
    plan=module.make_plan(mode=='engineering_smoke')
    if plan['source_sha256']!=digest(raw):raise ValueError('Source changed during snapshot preparation')
    return module,plan

def completion_assets(root, runner, core_raw, explicit_driver=None):
    """Read the fixed driver once; decode its verified embedded assets only."""
    name='elm_v43_fast_slow_authority_complete.py'
    path=Path(explicit_driver).resolve() if explicit_driver else root/name
    if not explicit_driver and not path.exists():path=Path(__file__).resolve().parent.parent/name
    raw=path.read_bytes()
    if digest(raw)!=COMPLETION_SHA256:raise ValueError('Completion driver identity mismatch')
    tree=ast.parse(raw.decode())
    assignments=[node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='PAYLOADS' for target in node.targets)]
    if len(assignments)!=1:raise ValueError('Invalid completion payload declaration')
    payloads=ast.literal_eval(assignments[0].value)
    def unpack(key):
        item=payloads[key];value=zlib.decompress(base64.b85decode(item['data']))
        if digest(value)!=item['sha256']:raise ValueError('Completion asset checksum mismatch: '+key)
        return value
    if digest(core_raw)!=CORE_SHA256 or unpack('core')!=core_raw:raise ValueError('Completion driver and scientific core differ')
    diagraw=unpack('diagnostics')
    if digest(diagraw)!=CORRECTED_DIAGNOSTICS_SHA256:raise ValueError('Unexpected corrected diagnostic source')
    # Definitions and their code identity only; no fit, data load or learner call.
    diagnostic=types.ModuleType('v43_snapshot_diagnostic_definitions')
    exec(compile(diagraw,'<snapshot:corrected-diagnostic-definitions>','exec'),diagnostic.__dict__)
    expected={
        'format':'V43-analysis-implementation-1',
        'scientific_core_sha256':CORE_SHA256,
        'completion_driver_sha256':COMPLETION_SHA256,
        'original_diagnostics_sha256':runner.PAYLOADS['diagnostics']['sha256'],
        'corrected_diagnostics_sha256':CORRECTED_DIAGNOSTICS_SHA256,
        'diagnostic_implementation_sha256':diagnostic._implementation_hash(),
        'scientific_plan_and_checkpoint_identity':'unchanged frozen core',
        'change':'Decode never-revised revision_age=-1 as missing; reject contradictory markers.',
        'physical_learning_changes':False,'policy_scoring_changes':False,
        'model_or_hyperparameter_changes':False,'gate_changes':False,
        'status_scope':'Implementation provenance only; completion is recorded in PREDICTABILITY_REPORT.json.',
        'pause_exit_status':'75 when an interruption is requested, including during diagnostics',
    }
    assets={name:raw}
    for key,filename in [('diagnostics','diagnostics_corrected.py'),('fix_md','DIAGNOSTIC_ENCODING_FIX.md'),('fix_json','DIAGNOSTIC_ENCODING_FIX.json'),('erratum','PROTOCOL_ERRATUM.md')]:
        payload=unpack(key);present=read_once(root/filename)
        if present is not None and present!=payload:raise ValueError('Existing completion asset differs: '+filename)
        assets[filename]=present if present is not None else payload
    present=read_once(root/'ANALYSIS_IMPLEMENTATION.json')
    if present is not None:
        if json.loads(present)!=expected:raise ValueError('Analysis implementation provenance mismatch')
        assets['ANALYSIS_IMPLEMENTATION.json']=present
    return assets,{'completion_driver_sha256':COMPLETION_SHA256,'corrected_diagnostics_sha256':CORRECTED_DIAGNOSTICS_SHA256,
        'diagnostic_implementation_sha256':expected['diagnostic_implementation_sha256'],
        'analysis_provenance_present_at_capture':present is not None,
        'analysis_provenance_note':'Existing provenance is verified when present. Otherwise the completion driver creates it under the writer lock on resume. Snapshot capture does not apply the analysis override or execute diagnostics.'}


def snapshot(out_dir, output, source=None, completion_driver=None):
    root=Path(out_dir).resolve();output=Path(output).resolve()
    if not root.is_dir():raise ValueError('Missing output directory: '+str(root))
    if output==root or root in output.parents:raise ValueError('Snapshot output must be outside the running output directory')
    source=Path(source).resolve() if source else root/'elm_v43_fast_slow_authority.py'
    if not source.exists():source=Path(__file__).resolve().parent.parent/'elm_v43_fast_slow_authority.py'
    rawsource=source.read_bytes();sourcesha=digest(rawsource)
    existing=read_once(root/'PLAN.json');oldplan=json.loads(existing) if existing else None
    mode=oldplan.get('mode') if oldplan else 'science'
    if mode not in ('science','engineering_smoke'):raise ValueError('Unknown source plan mode')
    runner,plan=source_plan(source,rawsource,mode)
    if oldplan is not None and oldplan!=plan:raise ValueError('Existing plan/source/configuration mismatch')
    planraw=existing if existing is not None else json_bytes(plan)
    plansha=digest(canonical(plan));protocol=runner.PROTOCOL.encode();protocolsha=digest(protocol)
    assets,completion_info=completion_assets(root,runner,rawsource,completion_driver)
    engineering_identity={'source_sha256':sourcesha,'protocol_sha256':protocolsha}
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.tmp-'+uuid.uuid4().hex)
    hashes={};details=[];begun=time.time();completed=[];current_cps=[];engineering_done=[];engineering_cps=[];diagnostic=[]
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            def put(name,raw,kind,validation=None):
                if name in hashes:raise ValueError('Duplicate snapshot member: '+name)
                if not raw and kind=='checkpoint':raise ValueError('Empty checkpoint')
                compression=zipfile.ZIP_STORED if '.checkpoint' in name or name.endswith('.npz') else zipfile.ZIP_DEFLATED
                z.writestr(name,raw,compress_type=compression)
                hashes[name]=digest(raw)
                details.append({'path':name,'bytes':len(raw),'kind':kind,'captured_unix_time':time.time(),**({'validation':validation} if validation else {})})
            def file(name,kind,required=False):
                raw=read_once(root/name)
                if raw is None:
                    if required:raise ValueError('Required artifact disappeared: '+name)
                    return None
                put(name,raw,kind);return raw
            put('elm_v43_fast_slow_authority.py',rawsource,'frozen_runner')
            put('PLAN.json',planraw,'locked_plan')
            put('PROTOCOL.md',protocol,'locked_protocol')
            put('ORIGINAL_DESIGN.md',runner.ORIGINAL_PROTOCOL.encode(),'original_design')
            manifest={'source_sha256':sourcesha,'protocol_sha256':protocolsha,'plan_sha256':plansha,'seeds':plan['seeds'],'loads':plan['loads'],'expected_trajectories':plan['expected_trajectories']}
            manifestraw=read_once(root/'MANIFEST.json')
            if manifestraw is not None and json.loads(manifestraw)!=manifest:raise ValueError('Manifest/source/plan mismatch')
            put('MANIFEST.json',manifestraw if manifestraw is not None else json_bytes(manifest),'run_identity')
            for name,raw in assets.items():put(name,raw,'verified_completion_asset')
            file('BEFORE_SCIENCE_REVIEW.json','immutable_before_science_addendum')

            for job in plan['jobs']:
                name=runner.job_name(job);identity={'plan_sha256':plansha,'job':job}
                raw=read_once(root/'completed'/(name+'.json'))
                if raw is not None:
                    env=json.loads(raw)
                    if env.get('identity')!=identity or env.get('sha256')!=digest(canonical(env['data'])):raise ValueError('Completed envelope identity/checksum mismatch: '+name)
                    result=env['data']
                    if result.get('seed')!=job['seed'] or result.get('load')!=job['load'] or result.get('physical_trajectories')!=1:raise ValueError('Completed cell mismatch: '+name)
                    eventname='authority_events/'+name+'.npz'
                    if result.get('authority_events_file')!=eventname:raise ValueError('Unexpected authority table path')
                    eraw=read_once(root/eventname)
                    if eraw is None or digest(eraw)!=result['authority_events_sha256']:raise ValueError('Completed authority table checksum mismatch: '+name)
                    with zipfile.ZipFile(io.BytesIO(eraw)) as nz:
                        if nz.testzip() is not None:raise ValueError('Authority NPZ CRC failure')
                    put(eventname,eraw,'immutable_authority_events')
                    put('completed/'+name+'.json',raw,'verified_completed_result')
                    completed.append(name)
                    continue
                # Read previous first so it cannot be newer than captured current.
                prev=read_once(root/'checkpoints'/(name+'.checkpoint.previous'))
                current=read_once(root/'checkpoints'/(name+'.checkpoint'))
                if prev is not None and current is None:raise ValueError('Previous checkpoint without current: '+name)
                if current is not None:
                    if prev is not None:put('checkpoints/'+name+'.checkpoint.previous',prev,'checkpoint',checkpoint_header(prev,identity))
                    put('checkpoints/'+name+'.checkpoint',current,'checkpoint',checkpoint_header(current,identity));current_cps.append(name)
                    file('progress/'+name+'.json','advisory_progress')

            engraw=read_once(root/'ENGINEERING.json')
            if engraw is not None:
                eng=json.loads(engraw)
                if mode=='engineering_smoke':
                    if eng.get('source_sha256')!=sourcesha or eng.get('mode')!='smoke_only':raise ValueError('Smoke engineering identity mismatch')
                else:
                    if eng.get('sha256')!=digest(canonical({k:v for k,v in eng.items() if k!='sha256'})) or eng.get('identity')!=engineering_identity:raise ValueError('Engineering summary checksum/identity mismatch')
                    expected={(s,l) for s in runner.ENGINEERING_SEEDS for l in (64,256)}
                    runs=eng.get('runs',[])
                    if len(runs)!=4 or {(r['seed'],r['load']) for r in runs}!=expected or not all(r.get('passed') and r.get('full_length') for r in runs):raise ValueError('Engineering summary roster/completion mismatch')
                    if not eng.get('passed') or not all(eng.get('checks',{}).get('E'+str(i)) is True for i in range(9)):raise ValueError('Engineering summary has failed/missing checks')
                put('ENGINEERING.json',engraw,'verified_engineering_summary')
            for seed in runner.ENGINEERING_SEEDS:
                for load in (64,256):
                    name=f'{seed}_{load}';raw=read_once(root/'engineering'/(name+'.json'))
                    if raw is not None:
                        env=json.loads(raw)
                        if env.get('identity')!=engineering_identity or env.get('sha256')!=digest(canonical(env['data'])):raise ValueError('Engineering cell checksum/identity mismatch: '+name)
                        result=env['data']
                        if result.get('seed')!=seed or result.get('load')!=load or not result.get('passed') or not result.get('full_length'):raise ValueError('Engineering cell mismatch/incomplete: '+name)
                        put('engineering/'+name+'.json',raw,'verified_engineering_cell');engineering_done.append(name)
                        continue
                    ident={'engineering':engineering_identity,'seed':seed,'load':load}
                    prev=read_once(root/'engineering'/(name+'.checkpoint.previous'))
                    current=read_once(root/'engineering'/(name+'.checkpoint'))
                    if prev is not None and current is None:raise ValueError('Engineering previous without current: '+name)
                    if current is not None:
                        if prev is not None:put('engineering/'+name+'.checkpoint.previous',prev,'checkpoint',checkpoint_header(prev,ident))
                        put('engineering/'+name+'.checkpoint',current,'checkpoint',checkpoint_header(current,ident));engineering_cps.append(name)
                        file('engineering/'+name+'.progress.json','advisory_progress')
            identities=set()
            for path in sorted((root/'diagnostic_checkpoints').glob('*.checkpoint')):
                raw=read_once(path)
                if raw is None:continue
                if not raw.startswith(b'V43DIAG1\n'):raise ValueError('Invalid diagnostic checkpoint magic')
                env=json.loads(zlib.decompress(raw[9:]));value=env['data']
                if env.get('sha256')!=digest(canonical(value)):raise ValueError('Diagnostic cache checksum mismatch')
                if value['held_seed'] not in plan['seeds'] or value['model'] not in ('LOGISTIC','SPLINE-LOGISTIC','GRADIENT-BOOSTING'):raise ValueError('Unexpected diagnostic seed/model')
                if value['held_seed'] in value['training_seeds'] or len(value['probabilities'])!=len(value['test_indices']):raise ValueError('Invalid diagnostic fold layout')
                identities.add(value['identity']);name='diagnostic_checkpoints/'+path.name
                put(name,raw,'checksummed_diagnostic_fold');diagnostic.append(name)
            if len(identities)>1:raise ValueError('Mixed diagnostic input identities')
            for path in sorted((root/'invocations').glob('*.json')):
                raw=read_once(path)
                if raw is not None:
                    value=json.loads(raw)
                    if value.get('source_sha256')!=sourcesha:raise ValueError('Invocation source mismatch')
                    put('invocations/'+path.name,raw,'invocation_metadata')
            if source.read_bytes()!=rawsource:raise ValueError('Source changed while snapshot was captured')
            if existing is not None and read_once(root/'PLAN.json')!=existing:raise ValueError('Plan changed while snapshot was captured')
            metadata={'format':FORMAT,'base_snapshot_helper_sha256':BASE_HELPER_SHA256,**completion_info,'source_sha256':sourcesha,'protocol_sha256':protocolsha,'plan_sha256':plansha,'mode':mode,
                'plan_origin':'captured_existing' if existing is not None else 'generated_from_frozen_source_for_engineering_only_snapshot',
                'capture_started_unix':begun,'capture_finished_unix':time.time(),'completed_physical_cells':completed,
                'resumable_physical_cells':current_cps,'completed_scientific_cells':completed if mode=='science' else [],
                'resumable_scientific_cells':current_cps if mode=='science' else [],'completed_engineering_cells':engineering_done,'resumable_engineering_cells':engineering_cps,
                'diagnostic_fold_count':len(diagnostic),'diagnostic_input_identities':sorted(identities),
                'checkpoint_unpickling_performed':False,'learner_updates_performed':0,
                'consistency':'Each atomic file captured once and these exact validated bytes archived. Independent jobs may represent different times. Previous checkpoints read before current checkpoints. Progress JSON is advisory and may be ahead of the captured checkpoint.',
                'diagnostic_validation':'Fold checksum, seed/model roster, and common input identity checked. Frozen runner revalidates full implementation/data/fold identity before reuse.',
                'finished_large_checkpoints_omitted':True,'merged_events_and_dynamic_reports_omitted':'Regenerate from immutable per-cell results with the frozen runner.',
                'files':details}
            put('SNAPSHOT.json',json_bytes(metadata),'snapshot_metadata')
            smoke=' --smoke' if mode=='engineering_smoke' else ''
            instructions=f'''# V43 resume snapshot\n\nExtract this archive into a directory, then run:\n\n```bash\npython3 /path/to/extracted/elm_v43_fast_slow_authority_complete.py --out-dir /path/to/extracted{smoke} --install-deps\n```\n\nThe completion driver preserves the unchanged scientific core identities, resumes unfinished checkpoints and reuses completed results. It also applies the separately verified diagnostic missing-value decoder. Invoke the completion driver above, rather than the original core directly. This snapshot contains independently captured job states; it is not a simultaneous global barrier. That is sufficient because seed/load learners do not share mutable state. Progress JSON is advisory. Checkpoint payloads were checksummed without unpickling. Diagnostic caches receive full input/implementation validation again at resume. Finished large physical checkpoints and regenerable merged reports are omitted. No learning was performed during capture.\n'''
            put('SNAPSHOT_README.md',instructions.encode(),'resume_instructions')
            z.writestr('SNAPSHOT_HASHES.json',json_bytes(hashes))
        audit=verify_zip(tmp)
        with tmp.open('rb') as handle:os.fsync(handle.fileno())
        os.replace(tmp,output)
        fd=os.open(output.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return {'output':str(output),**completion_info,'source_sha256':sourcesha,'zip_sha256':digest(output.read_bytes()),'bytes':output.stat().st_size,
                'completed_physical_cells':len(completed),'resumable_physical_cells':len(current_cps),'completed_scientific_cells':len(completed) if mode=='science' else 0,'resumable_scientific_cells':len(current_cps) if mode=='science' else 0,'completed_engineering_cells':len(engineering_done),'resumable_engineering_cells':len(engineering_cps),'diagnostic_folds':len(diagnostic),**audit}
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--source');p.add_argument('--completion-driver');args=p.parse_args()
    print(json.dumps(snapshot(args.out_dir,args.output,args.source,args.completion_driver),indent=2,sort_keys=True))

if __name__=='__main__':main()
