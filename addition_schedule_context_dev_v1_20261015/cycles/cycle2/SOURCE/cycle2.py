import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import copy,math,time,traceback
import bootstrap,config,data,models,engine,checkpoint,diagnostics,audits,probes
import numpy as np
from artifacts import write,read,now
from ledger import apply


def rejects(call):
    try:call()
    except (ValueError,AssertionError,FileNotFoundError):return
    raise AssertionError('tamper not rejected')


def run():
    folder=config.HERE/'cycles/cycle2';world=config.TECH_WORLDS[1];s=data.make_addition(world,True);checks=[]
    model=models.Keep(data.seed(world,'model'));records=data.ordered(s,'BLOCKED','W');rows=[]
    for r in records[:2]:rows.append(apply(2,'KEEP/warm/'+r['id'],lambda r=r:engine.apply_addition(model,r)))
    path=folder/'keep.json.gz';checkpoint.save(model,path,cursor=2,next_id=records[2]['id'],manifest_id='C2')
    restored,cursor=checkpoint.load(path,expected_kind='KEEP',manifest_id='C2')
    assert restored.digest()==model.digest() and cursor['next_id']==records[2]['id']
    r=records[2]
    for label,m in [('original',model),('restored',restored)]:
        apply(2,'KEEP/continue/'+label,lambda m=m:engine.apply_addition(m,r))
    assert model.digest()==restored.digest()
    checks.append('full_recurrent_parameter_RMS_cursor_and_next_update_roundtrip')
    r=records[3];apply(2,'KEEP/captured_proposal',lambda:engine.apply_addition(model,r,capture=True))
    capture=copy.deepcopy(model.last_capture);before=model.digest();resources=copy.deepcopy(model.resource)
    result=diagnostics.views(capture,[data.example(p) for p in s['groups']['old'][:3]])
    assert len(result['views'])==7 and result['training_applications']==0 and model.digest()==before
    assert resources==model.resource
    write(folder/'DIAGNOSTIC_VIEWS.json.gz',result)
    checks.append('seven_offline_views_exact_ALL_NONE_no_gradient_reexecution')
    for r in records[:2]:
        off=dict(r,allowed=False)
        row=apply(2,'KEEP/nonzero_forbidden/'+r['id'],lambda:engine.apply_addition(restored,off))
        assert row['versions_before']==row['versions_after']
    for variant in config.NATIVE_VARIANTS:
        native=models.Native(variant);item=data.native_fixture(world)[0]
        row=apply(2,variant+'/warm',lambda:engine.apply_native(native,item,True));audits.audit_native_record(row,item,True)
        p=folder/(variant+'.json.gz');checkpoint.save(native,p,cursor=1,next_id='NEXT',manifest_id='C2')
        loaded,_=checkpoint.load(p,expected_kind=variant,manifest_id='C2')
        assert loaded.digest()==native.digest()
        for label,m in [('original',native),('restored',loaded)]:
            apply(2,variant+'/continue/'+label,lambda m=m:engine.apply_native(m,item,True))
        assert native.digest()==loaded.digest()
        apply(2,variant+'/nonzero_forbidden',lambda:engine.apply_native(loaded,item,False))
        view=native.clone();view._prompt(item['cue']);view._event(item['answer'][0])
        p=folder/(variant+'_MID_RESPONSE.json.gz');checkpoint.save(view,p,cursor=0,next_id='CONTINUATION',manifest_id='C2')
        continuation,_=checkpoint.load(p,expected_kind=variant,manifest_id='C2')
        assert continuation.digest()==view.digest() and continuation.fronts[1].q==item['cue'] and continuation.fronts[1].prefix==item['answer'][:1]
        np.testing.assert_array_equal(view.forecast()[1],continuation.forecast()[1])
        for a,b in zip(view.arrays(),continuation.arrays()):assert not np.shares_memory(a,b)
        rejects(lambda:checkpoint.load(p,expected_kind='BOUND-C16' if variant=='ROLLING-C16' else 'ROLLING-C16',manifest_id='C2'))
        record=read(p);record['next_id']='BAD';bad=folder/(variant+'_BAD_CURSOR.json.gz');write(bad,record)
        rejects(lambda:checkpoint.load(bad,expected_kind=variant,manifest_id='C2'))
        record=read(p);record['payload_base64']='AA==';bad=folder/(variant+'_BAD_PAYLOAD.json.gz');write(bad,record)
        rejects(lambda:checkpoint.load(bad,expected_kind=variant,manifest_id='C2'))
    checks.append('both_native_roundtrip_complete_mid_answer_context_and_nonaliasing')
    checks.append('wrong_variant_cursor_and_payload_tamper_rejected')
    audits.audit_addition_rows(s,'BLOCKED','W',rows)
    bad=copy.deepcopy(rows);bad[0]['allowed']=False
    rejects(lambda:audits.audit_addition_rows(s,'BLOCKED','W',bad))
    bad=copy.deepcopy(rows);bad[0]['versions_after']['actor_steps']+=1
    rejects(lambda:audits.audit_addition_rows(s,'BLOCKED','W',bad))
    q=probes.own(model,[data.example(s['groups']['held'][0])]);audits.checked_own(q['rows'],[data.example(s['groups']['held'][0])])
    q['rows'][0]['correct']=not q['rows'][0]['correct']
    rejects(lambda:audits.checked_own(q['rows'],[data.example(s['groups']['held'][0])]))
    checks.append('independent_auditor_rejects_permission_write_count_and_own_output_tampering')
    return dict(accepted=True,checks=checks,technical_only=True,behavior_at_full_dose_not_tested=True,
                zero_feedback_native_continuation_events=4,production_updates_replayed_by_diagnostics=0)

if __name__=='__main__':
    start=time.monotonic()
    try:r=run()
    except BaseException as e:
        write(config.HERE/'cycles/cycle2/RESULT.json',dict(accepted=False,error=repr(e),traceback=traceback.format_exc(),utc=now()));raise
    r.update(utc=now(),elapsed=time.monotonic()-start)
    write(config.HERE/'cycles/cycle2/RESULT.json',r);print(r)
