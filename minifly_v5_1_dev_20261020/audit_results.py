"""Independent artifact reducer: no model import, retraining or trajectory replay."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def load(p):return json.loads((HERE/p).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):(HERE/p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
    a1,a2,a3=[load(f'cycles/cycle{i}/AUDIT.json') for i in (1,2,3)]
    assert a1['accepted'] and a2['accepted'] and a3['accepted']
    assert a1['checks']['max_native_state_error']==0 and a1['checks']['max_neutral_reference_error']==0
    assert a2['Q0_passed'] and a2['selected']=={'a_max':.0625,'W':1024}
    expected_records=0;target_events=0;port_rows=[]
    for setting in a2['settings']:
        for b in setting['births']:
            expected_records+=b['external_training_records']+b['matched_passive_records']
            target_events+=b['target_events_per_branch']*2
            assert b['external_training_records']==b['matched_passive_records']==8*256
            for row in b['snapshots']:
                for mode in ['W','N']:
                    rr=row[mode]['rows']
                    exact=sum(bytes.fromhex(x['expected'])==bytes.fromhex(x['payload']) and x['ended'] for x in rr)/len(rr)
                    assert exact==row[mode]['exact']
                assert row['W']['exact']==1 and row['N']['exact']==0
            rest=b['one_hour_taught_exact'];assert rest['exact']==1
            assert all(x['ended'] and x['payload']==x['expected'] for x in rest['rows'])
            port_rows.append({'seed':b['seed'],'immediate_exact':b['snapshots'][-1]['W']['exact'],'passive_exact':b['snapshots'][-1]['N']['exact'],'hour_exact':rest['exact']})
    assert expected_records==a2['full_record_applications']==8192
    listening=answer_bytes=complete=partials=0;identity_rows=[]
    for birth in a3['births']:
        assert not birth['M2_qualified'] and not birth['M3_run'] and not birth['integration_run']
        assert len(birth['M2_candidates'])==4
        for c in birth['M2_candidates']:
            t=c['trace'][-1];m=list(t['winner_map'].values())
            assert len(m)==12 and len(set(m))==8 and t['alias_count']==4 and not c['qualified']
            assert t['bytes']==200000 and t['stable']
            ex=t['exposure'];listening+=ex['payload_bytes'];answer_bytes+=ex['answer_payload_bytes']
            complete+=ex['complete_record_presentations'];partials+=ex['partial_record_presentations']
            identity_rows.append(dict(seed=birth['seed'],eta=c['eta'],gain=c['gain'],types=12,winners=8,bytes=200000))
    assert listening==a3['listening_payload_bytes']==1600000
    assert answer_bytes==511012 and complete==155664 and partials==8
    assert a3['full_record_applications']==0 and not a3['arithmetic_efficacy_run'] and not a3['oracle_run']
    # Failed implementation attempts remain outside qualified model evidence.
    failed=[]
    for f in sorted((HERE/'cycles').glob('cycle*/*/ACCOUNTING.json')):
        d=json.loads(f.read_text());failed.append({'path':str(f.relative_to(HERE)),**d})
    invalid=sum(x.get('training_records',0) for x in failed)
    assert invalid==9
    total=a1['full_record_applications']+a2['full_record_applications']+invalid
    assert total==8206
    ledger={'native_full_record_applications_total':total,'accepted_technical_full_record_applications':8197,
        'preserved_failed_attempt_full_record_applications':invalid,
        'isolated_port_W_presentations':4096,'isolated_port_no_write_presentations':4096,
        'isolated_port_native_target_event_calls':target_events,
        'accepted_cycle1_structural_full_records':5,
        'partial_fixture_events':{'accepted_primary':21,'accepted_native_reference':12,'accepted_passive_frames':4,
            'failed_primary':63,'failed_native_reference':36,'failed_passive_frames':8,
            'accounting_method':'Function/control-flow counts from preserved test sources; query events reported in phase CPU, not counted as training samples'},
        'listening':{'payload_byte_exposures':listening,'answer_payload_byte_exposures':answer_bytes,
            'complete_record_presentations':complete,'partial_record_presentations':partials,
            'bytes_per_setting':200000,'settings_times_births':8,'parameter_settings_reuse_world_stream':True,
            'native_value_writes_during_listening':0},
        'failed_attempts':failed,'science_jobs':0,'scientific_histories':0,
        'cost_limits':'Failed-attempt CPU not independently itemized; phase CPU includes evaluation; no overall efficiency claim'}
    write('TRAINING_LEDGER.json',ledger)
    # Causal learner source stays fixed through successful cycles. New operation
    # modules/phase locks can be added between cycles; old snapshots are retained.
    core=['v51_common.py','engine.py','representation.py','v51_core.py']
    unchanged={}
    for name in core:
        key=str((HERE/name).relative_to(ROOT));current=sha(HERE/name)
        for audit in (a1,a2,a3):assert audit['source_after'][key]==current,(name,key)
        unchanged[name]=current
    parent=json.loads((ROOT/'minifly_v4_dev_20261020/PARENT_ASSET_AUDIT.json').read_text())['files']
    for name,digest in parent.items():assert sha(ROOT/name)==digest,name
    write('PARENT_ASSET_AUDIT.json',{'accepted':True,'scope':'Pinned historical native assets remain unchanged','files':parent})
    summary={'status':'THREE_CYCLES_COMPLETE_NO_ADVANCE','verdict':'NO_ADVANCE_Q1_BYTE_ALIASING','cycles_executed':3,
        'port_qualified':True,'identity_qualified':False,'context_learning_run':False,'integrated_encoder_recall_run':False,
        'unseen_addition_efficacy_run':False,'evidence':'E0 numerical/causal checks and isolated-port E1; full-model E3 untested',
        'port_results':port_rows,'identity_results':identity_rows,
        'native_full_record_applications_total':total,'listening_payload_byte_exposures':listening,
        'scientific_jobs':0,'scientific_histories':0,'formal_campaign_ready':False,'adopted':False,
        'accepted_phase_CPU_seconds':sum(x['CPU_seconds'] for x in (a1,a2,a3)),
        'phase_peak_RSS_bytes':max(x['peak_RSS_bytes'] for x in (a1,a2,a3)),
        'actual_full_CPU_including_failed_work_available':False,
        'revised_design':str(ROOT/'minifly_v5_1_design_20261020/REVISED_DESIGN.md'),
        'revision_implemented':False,'revision_tested':False}
    write('SUMMARY.json',summary)
    write('QUALIFICATION.json',{'status':'NO_ADVANCE','numerical_and_causal_fixture_checks_passed':True,
        'Q0_isolated_port_passed':True,'Q1_identity_passed':False,'complete_encoder_qualified':False,
        'formal_campaign_ready':False,'scientific_jobs':0,'next_revision_unimplemented':True})
    write('FINAL_AUDIT.json',{'accepted':True,'audit_type':'Independent artifact reduction, no learner execution',
        'exact_scores_recomputed_from_payloads':True,'exposure_and_native_counts_reduced':True,
        'model_sources_unchanged_during_accepted_cycles':unchanged,'historical_parent_files_verified':len(parent),
        'prerequisite_stop_respected':True,'evidence_level_not_promoted':True,
        'formal_campaign_ready':False,'verdict':summary['verdict'],'source_files':{
            str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.glob('*.py'))}})
    print(json.dumps({'accepted':True,'verdict':summary['verdict'],'native_records':total,'listening_bytes':listening,'port_births_verified':len(port_rows),'sensory_settings_verified':len(identity_rows)},indent=2))
if __name__=='__main__':main()
