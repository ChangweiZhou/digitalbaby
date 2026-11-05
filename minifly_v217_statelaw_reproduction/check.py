"""Protocol and instrumentation checks; no parameter search or seed selection."""
import contextlib, hashlib, io, json, os, runpy, sys, tempfile
from pathlib import Path
from worker import evaluate
ROOT=Path(__file__).resolve().parent

def main():
    plan=json.loads((ROOT/'PLAN.json').read_text());manifest=json.loads((ROOT/'DATA_MANIFEST.json').read_text());checks={}
    checks['fresh_seed_rosters']=not(set(plan['round23_seeds'])&{0,1,2,3,4,5,6,7,8,9}) and not(set(plan['round24_seeds'])&{0,1,2,3,4,5,6,7,8,9})
    checks['150_jobs']=len(plan['jobs'])==150
    checks['round23_1500_questions']=all(m['questions']==1500 for m in manifest if 'r23_' in m['stream'])
    checks['round24_80_scored_questions']=True
    for job in plan['jobs']:
        if job['round']!=24: continue
        lines=(ROOT/job['stream']).read_text().splitlines();off=job['testfrom']
        # Historical TESTFROM is a one-based >= boundary; last history line is always an answer,
        # so boundary must be checked rather than silently corrected in reproduction.
        scored=sum('?' in ln for i,ln in enumerate(lines,1) if i>=off)
        if scored!=80+(job['arm']=='k24'): checks['round24_80_scored_questions']=False
    # Native recovered code includes the final historical question in the test metrics (81 vs 80).
    checks['confirmed_native_boundary_overlap']=checks.pop('round24_80_scored_questions')
    checks['disjoint_C']=all(m['cross_world_reused_content_types']==0 for m in manifest if '_C_' in m['stream'])
    checks['paired_test_text_identical']=True
    checks['completed_with_receipts']=True
    for job in plan['jobs']:
        result=ROOT/'results'/(job['id']+'.json')
        if not result.exists(): checks['completed_with_receipts']=False; continue
        row=json.loads(result.read_text());receipt=row.get('receipts',{})
        if receipt.get('stream_sha256')!=hashlib.sha256((ROOT/job['stream']).read_bytes()).hexdigest() or receipt.get('source_sha256')!=hashlib.sha256((ROOT/job['source']).read_bytes()).hexdigest() or receipt.get('python_hash_seed')!=plan['python_hash_seed']:checks['completed_with_receipts']=False
        if job['round']==24 and job['arm']=='k24':
            historical=(ROOT/job['stream']).read_text().splitlines()
            cold=ROOT/job['stream'].replace('_k24.txt','_k0.txt')
            if historical[job['testfrom']:]!=cold.read_text().splitlines():checks['paired_test_text_identical']=False
            trace=[json.loads(l) for l in result.with_suffix('.trace.jsonl').read_text().splitlines()]
            if sum(r['line']>job['testfrom'] for r in trace)!=80:checks['completed_with_receipts']=False
    recovered=json.loads((ROOT/'RECOVERY.json').read_text())['recovered_sha256']
    checks['recovered_sources_unchanged']=all(hashlib.sha256((ROOT/'recovered'/name).read_bytes()).hexdigest()==digest for name,digest in recovered.items())
    first=plan['jobs'][0];source=ROOT/first['source'];stream=ROOT/first['stream']
    res,trace=evaluate(source,stream,'full')
    old=sys.argv[:];sys.argv=[str(source),str(stream),'full'];stdout=io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout): ns=runpy.run_path(str(source),run_name='__native__')
    finally:sys.argv=old
    checks['instrumentation_native_output_identical']=json.loads(stdout.getvalue().splitlines()[0])==res['native']
    checks['round23_trace_matches_native_accuracy']=all(round(res['metrics'][k]['top1'],2)==res['native']['avail_top1'][k][1] for k in ['per','obj'])
    # A changed answer cannot alter that answer's top-1 prediction. It may affect later predictions.
    lines=stream.read_text().splitlines();target=next(i for i,ln in enumerate(lines) if '?' in ln)
    lines[target]=lines[target].split('? ')[0]+'? UNSEENPLACE'
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'stream.txt';p.write_text('\n'.join(lines)+'\n');_,other=evaluate(source,p,'full')
    checks['prediction_precedes_answer_feedback']=trace[0]['prediction']==other[0]['prediction'] and trace[0]['op_evidence']==other[0]['op_evidence']
    checks['gold_fallback_detected']=res['diagnostics']['empty_fallback_count']>0
    assert all(checks.values()),checks
    (ROOT/'CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
