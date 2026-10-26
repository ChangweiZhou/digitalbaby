"""Package compact V154 source, aggregate results and audits; never raw records/states."""
from pathlib import Path
import json, hashlib, zipfile, time
H=Path(__file__).resolve().parent;R=H.parent;D=R/'minifly_v154_design_20261025'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    start=time.process_time()
    assert load(H/'FINAL_AUDIT.json')['status']=='ACCEPTED'
    assert load(H/'CLOSURE_AUDIT.json')['status']=='ACCEPTED'
    original=load(H/'SOURCE_LOCK.json')
    assert all(sha(R/p)==v for p,v in original.items())
    preservation=load(H/'PRESERVATION_BASELINE.json')
    assert sha(R/preservation['arithmetic_pointer'])==preservation['arithmetic_pointer_sha256']
    assert sha(R/preservation['arithmetic_pointer'])==load(D/'PROVENANCE.json')['local_sources'][preservation['arithmetic_pointer']]
    assert (H/'REPORT.md').exists() and (H/'REVISED_DESIGN.md').exists()
    assert (H/'PROTOCOL_REVIEW.md').exists() and (H/'POSTRUN_COSTS.json').exists()
    post={str(p.relative_to(R)):sha(p) for p in sorted(H.glob('*.py')) if str(p.relative_to(R)) not in original and p.name!='continue154.py'}
    save(H/'POSTRUN_SOURCE_LOCK.json',dict(scope='saved-evidence reduction, plotting, closure and packaging only; no learner/inference calls',files=post))
    status=load(H/'STATUS.json');status.update(status='THREE_CYCLES_COMPLETE',package_complete=True,new_core_adopted=False,github_pushed=False)
    save(H/'STATUS.json',status)
    sources=list(D.glob('*.md'))+list(D.glob('*.json'))+list(H.glob('*.py'))
    names=['AUTHORIZATION.json','SOURCE_LOCK.json','POSTRUN_SOURCE_LOCK.json','STATUS.json',
           'SUMMARY.json','COUNTS.json','DATA_LOCK.json','ACQUISITION_LOCK.json','ANALYSIS.json',
           'FINAL_AUDIT.json','CLOSURE_AUDIT.json','PRESERVATION_BASELINE.json','POSTRUN_COSTS.json',
           'REPORT.md','REVISED_DESIGN.md','PROTOCOL_REVIEW.md','OPERATIONS_SOURCE_LOCK.json','OPERATIONS_CONTINUATION.json','FAILURE.json','FAILURE_RESOLUTION.json','DECISION.json','LEDGER.jsonl']
    sources += [H/n for n in names]
    sources += list((H/'cycles').glob('*.json'))+list((H/'cycles/logging_failure').glob('*.json'))+list(H.glob('PROBE_*.json'))
    sources += list((H/'groups').glob('*/RESULT.json'))+list((H/'groups').glob('*/RECEIPT.json'))
    sources += list(H.glob('ACQUISITION_*.png'))+list(H.glob('ACQUISITION_*.svg'))
    sources=sorted(set(sources))
    for p in sources:
        assert p.is_file()
        assert 'local' not in p.relative_to(H).parts if p.is_relative_to(H) else True
        assert p.name not in ['DATA.json','ACQUISITION.json','HELD.json','COUNTERMODELS.json','LANE_C.json']
    manifest={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sources}
    save(H/'BUNDLE_MANIFEST.json',dict(version='V154',entries=manifest,raw_records_states_and_per_query_outputs_included=False))
    bundle=H/'RESULT_BUNDLE.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sources+[H/'BUNDLE_MANIFEST.json']:z.write(p,str(p.relative_to(R)))
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        assert set(z.namelist())==set(manifest)|{str((H/'BUNDLE_MANIFEST.json').relative_to(R))}
        assert all(hashlib.sha256(z.read(p)).hexdigest()==v['sha256'] for p,v in manifest.items())
    package=dict(version='V154',status='COMPLETE',bundle='RESULT_BUNDLE.zip',sha256=sha(bundle),bytes=bundle.stat().st_size,
                 entries=len(sources)+1,manifest='BUNDLE_MANIFEST.json',archive_entries_verified=True,
                 scientific_audit='ACCEPTED',closure_audit='ACCEPTED',frozen_sources_unchanged=True,
                 arithmetic_pointer_unchanged=True,raw_data_included=False,core_adopted=False,github_pushed=False,
                 package_CPU=time.process_time()-start)
    save(H/'PACKAGE.json',package);print(json.dumps(package))

if __name__=='__main__':main()
