"""Cycle 3 targeted revision check: new metadata and raw-byte edge cases only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy,time,traceback,ast
import bootstrap,config,data,models,engine,audits,checkpoint
from artifacts import read,write,now
from ledger import apply


def rejects(call):
    try:call()
    except (ValueError,AssertionError):return
    raise AssertionError('tamper accepted')


def run():
    q=b'\x00\n =\xff';answer=b'\x00\xff';checks=[]
    for variant in config.NATIVE_VARIANTS:
        m=models.Native(variant);birth=engine.native_birth_evidence(m);audits.audit_birth_evidence(birth)
        bad=copy.deepcopy(birth);bad['births'][0]['canonical_B_sha256']='BAD'
        rejects(lambda:audits.audit_birth_evidence(bad))
        for allowed in (True,False):
            r=apply(3,'final-revision/'+variant+'/'+str(allowed),lambda:m.train(q,answer,allowed))
            assert [x['target'] for x in r['trace']]==[0,255,256]
            assert m.fronts[1].q==q and m.fronts[1].prefix==answer and m.fronts[1].frozen
        before=m.digest();m.generate_trace(q,cap=2);assert before==m.digest()
    k=models.Keep(81)
    apply(3,'final-revision/KEEP/raw-bytes',lambda:k.train(q,answer,True))
    checks+=['actual_canonical_birth_rows_and_tamper_rejection','both_native_raw_NUL_newline_space_0xFF_and_nonzero_forbidden_update','recurrent_raw_payload_contract']
    receipt=copy.deepcopy(read(config.HERE/'cycles/cycle3/RESTORED/RECEIPT.json.gz'))
    receipt['manifest']['model_seed']=data.seed(receipt['manifest']['world'],'model')
    for h in receipt['histories'].values():
        h['initial_digest']=next(q['operative_digest'] for q in h['curve'] if q['T']==0)
        h['parameter_count']=sum(p.numel() for p in k.p.values())
    assert audits.audit_addition_job(receipt)['accepted']
    checks.append('metadata_only_compatibility_fixture_passes_revised_independent_auditor')
    for p in config.HERE.glob('*.py'):ast.parse(p.read_text())
    return dict(accepted=True,checks=checks,record_applications=5,cycle=3,
                metadata_fixture_not_a_new_science_receipt=True,old_receipts_preserved=True)

if __name__=='__main__':
    start=time.monotonic()
    try:r=run()
    except BaseException as e:
        write(config.HERE/'cycles/cycle3/FINAL_REVISION_RESULT.json',dict(accepted=False,error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
    r.update(utc=now(),elapsed=time.monotonic()-start)
    write(config.HERE/'cycles/cycle3/FINAL_REVISION_RESULT.json',r);print(r)
