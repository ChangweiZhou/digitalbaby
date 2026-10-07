"""Metadata/dispatcher tests only; never execute native learning or science."""
import copy
import json
from pathlib import Path
import statistics
import tempfile
import time
from unittest.mock import patch
from ops_common import OPS, ROOT, RESULTS, WORLDS, EVIDENCE, atomic, authorized_worlds, read, sha


def plan(world):
    import spec
    import streams
    inputs = streams.world_inputs(world)
    result = dict(inputs={n:{k.hex():v for k,v in m.items()} for n,m in inputs.items()}, training={}, probes={})
    for si, stage in enumerate(('old','day1','new','day2','revised')):
        if stage in ('old','new','revised'):
            result['training'][stage] = streams.stream(inputs[stage], spec.REPEATS, world+si).hex()
        mappings = {'old':inputs['old']}
        if si >= 2:
            mappings['new'] = inputs['new']
        if stage == 'revised':
            mappings.pop('old')
            mappings.update(current_old={**inputs['old'],**inputs['revised']}, revised=inputs['revised'],
                            unchanged={k:v for k,v in inputs['old'].items() if k not in inputs['revised']})
        result['probes'][stage] = {n:streams.stream(m, spec.PROBE_REPEATS, world+100+si).hex() for n,m in mappings.items()}
    return result


def test_dispatcher():
    import ops_common as common
    import supervisor
    import analysis
    cases = []
    print('SYNTHETIC DISPATCHER TESTS ONLY: no native worlds or science receipts', flush=True)
    # Isolated synthetic process/receipt doubles: no native model constructed.
    for fail in (False, True):
        with tempfile.TemporaryDirectory(prefix='observation_ops_') as tmp:
            tmp = Path(tmp)
            resultdir = tmp/'results'
            operations = tmp/'ops'
            operations.mkdir()
            atomic(operations/'QUALIFICATION.json', {'verdict':'PASS'})
            calls = []

            class Child:
                def __init__(self, command, **kwargs):
                    self.pid = 123456
                    self.returncode = 0
                    self.command = command
                    if '--world' in command:
                        world = int(command[-1])
                        calls.append(world)
                        if fail:
                            self.returncode = 9
                            atomic(resultdir/'failures'/f'{world}.json', {'error':'SYNTHETIC_ACTIONABLE_FAILURE'})
                        else:
                            atomic(resultdir/'receipts'/f'WORLD_{world}.json.gz',
                                   {'world':world,'independent_audit':{'verdict':'PASS'}}, exclusive=True)
                def poll(self):
                    return self.returncode
                def terminate(self):
                    self.returncode = 0
                def wait(self, **kwargs):
                    return self.returncode

            with patch.object(common,'RESULTS',resultdir), patch.object(supervisor,'RESULTS',resultdir), \
                 patch.object(supervisor,'OPS',operations), patch.object(supervisor,'verify_sources',lambda:'SYNTHETIC'), \
                 patch.object(supervisor.subprocess,'Popen',Child), \
                 patch.object(analysis,'analyze',lambda:{'verdict':'SYNTHETIC_NEGATIVE_VALID_COMPLETION'}), \
                 patch.object(supervisor,'package',lambda summary:None):
                if fail:
                    try:
                        supervisor.main()
                    except RuntimeError as exc:
                        assert 'SYNTHETIC_ACTIONABLE_FAILURE' in str(exc)
                    else:
                        raise AssertionError('failed child did not stop dispatcher')
                    assert calls == [WORLDS[0]]
                    assert read(resultdir/'STATUS.json')['stage'] == 'FAILED'
                    assert read(resultdir/'FAILURE.json')['committed_worlds'] == 0
                    cases.append('failure_stops_without_retry_or_next_world')
                else:
                    supervisor.main()
                    assert calls == list(WORLDS)
                    final = read(resultdir/'STATUS.json')
                    assert final['stage']=='COMPLETE' and final['committed_training_lives']==96
                    assert final['verdict']=='SYNTHETIC_NEGATIVE_VALID_COMPLETION'
                    cases.append('exact_roster_and_negative_completion')
                    try:
                        supervisor.main()
                    except (ValueError, BlockingIOError):
                        pass
                    else:
                        raise AssertionError('dispatcher accepted repeat launch')
                    cases.append('repeat_launch_rejected')
    return cases


def main():
    import spec
    import audit
    from science_audit import audit_full
    from analysis import gates, interval, measure
    started = time.time()
    if (OPS/'SOURCE_LOCK.json').exists() or (RESULTS/'LAUNCH.json').exists():
        raise ValueError('qualification cannot mutate an already frozen or launched program')
    q = read(ROOT/'QUALIFICATION.json')
    assert q['status']=='DEV_QUALIFIED_SCIENCE_NOT_STARTED' and q['technical_verdict']=='PASS' and q['behavior_verdict']=='PASS'
    import checkpoint
    assert checkpoint.source_identity()==read(ROOT/'SOURCE_LOCK.json')['checkpoint_scientific_identity']
    old_guard = spec.require_dev
    with authorized_worlds():
        for world in WORLDS:
            spec.require_dev(world)
        for world in [True,810201,811000,811033,-1,'811001']:
            try:
                spec.require_dev(world)
            except ValueError:
                pass
            else:
                raise AssertionError('authorization failed closed')
        manifest = {'worlds':{str(w):plan(w) for w in WORLDS}, 'worlds_frozen':list(WORLDS),
                    'generated_without_native_learning':True,'phase_or_label_metadata_enters_learner':False}
    assert spec.require_dev is old_guard
    if (OPS/'INPUT_MANIFEST.json').exists():
        assert read(OPS/'INPUT_MANIFEST.json')==manifest
    else:
        atomic(OPS/'INPUT_MANIFEST.json',manifest,exclusive=True)
    original = read(ROOT/'cycles/cycle3/WORLD_810201.json')
    original_audit = audit.audit_receipt(original)
    adapted = {**original,'world':WORLDS[0],'evidence':EVIDENCE,'confirmation_source_lock_sha256':'TECHNICAL_FIXTURE'}
    devplan = plan(810201)
    accepted = audit_full(adapted,devplan,'TECHNICAL_FIXTURE')
    assert accepted['independently_checked_rows']==original_audit['independently_checked_rows']==8240
    assert measure(adapted)['retained']==q['worlds'][0]['day2_retained']
    tamper=[]
    def reject(name, change):
        r=copy.deepcopy(adapted)
        change(r)
        try:
            audit_full(r,devplan,'TECHNICAL_FIXTURE')
        except (ValueError,AssertionError,KeyError):
            tamper.append(name)
        else:
            raise AssertionError('accepted tamper: '+name)
    reject('wrong_world',lambda r:r.update(world=810201))
    reject('wrong_evidence',lambda r:r.update(evidence='DEV_ONLY_E0_E1'))
    reject('wrong_source',lambda r:r.update(confirmation_source_lock_sha256='wrong'))
    reject('missing_rest_stage',lambda r:r['probes'].pop('day1'))
    reject('missing_training_branch',lambda r:r['training']['old']['branches'].pop('N_OLD'))
    reject('missing_probe',lambda r:r['probes']['day2']['W'].pop('new'))
    reject('wrong_input',lambda r:r['inputs']['old'].update({next(iter(r['inputs']['old'])):99}))
    reject('wrong_raw_hash',lambda r:r['training']['old'].update(sha256='wrong'))
    reject('drop_event',lambda r:r['training']['old']['branches']['W'].pop())
    reject('changed_event',lambda r:r['training']['old']['branches']['W'][0].update(byte=99))
    reject('future_context',lambda r:r['training']['old']['branches']['W'][0].update(context='30'))
    reject('wrong_clock',lambda r:r['training']['old']['branches']['W'][0].update(t=1.))
    reject('wrong_emit',lambda r:r['training']['old']['branches']['W'][0].update(emitted=99))
    reject('wrong_internal_error',lambda r:r['training']['old']['branches']['W'][0].update(signs=[0.,0.,0.,0.]))
    reject('no_old_write_violation',lambda r:r['training']['old']['branches']['N_OLD'][0].update(plastic=True))
    reject('disabled_actual_write',lambda r:r['training']['old']['branches']['N_ALL'][0].update(native_write_l1=[1.,0.,0.,0.]))
    reject('reused_birth',lambda r:r['birth']['W'][1].update(fly_id=r['birth']['W'][0]['fly_id']))
    reject('wrong_birth',lambda r:r['birth']['W'][0].update(canonical_B_sha256='wrong'))
    reject('partial_byte_counter',lambda r:r['states']['revised/W'].update(bytes_seen=1279))
    reject('wrong_metric',lambda r:r['probes']['day2']['W']['old']['metrics'].update(focal_accuracy=0.1234))
    reject('wrong_lives',lambda r:r['resources'].update(actual_training_lives=1))
    reject('missing_erase',lambda r:r['probes']['old'].pop('NATIVE_ERASE'))
    assert interval([.25]*32)['low']==.25
    ci=interval([0.]*16+[1.]*16)
    assert ci['low'] < .5 < ci['high'] and ci['n_worlds']==32
    passing = {k:dict(mean=v,low=v) for k,v in dict(causal_gain=.1,all_old_probe_bpb_gain=.1,retained=.9,retention_delta=-.05,new=.8,revised=.8,unchanged=.85).items()}
    assert all(gates(passing).values())
    passing['causal_gain']['low']=0
    assert not gates(passing)['causal_gain']
    for length in (8,31,33):
        try:
            interval([1.]*length)
        except ValueError:
            pass
        else:
            raise AssertionError('partial or expanded n accepted')
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'receipt.json.gz'
        atomic(f,{'complete':True},exclusive=True)
        assert read(f)=={'complete':True}
        try:
            atomic(f,{'complete':False},exclusive=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError('committed data overwritten')
        assert read(f)=={'complete':True}
    pipeline=test_dispatcher()
    atomic(OPS/'QUALIFICATION.json',dict(verdict='PASS',time_unix=started,
        admission_guard_scope_only=True, qualified_native_source_edited=False,
        row_auditor_parity=True, rejected_tamper_cases=tamper,
        synthetic_dispatcher_tests=pipeline, world_level_inference_tests=True,
        no_overwrite_atomic_commit=True, no_new_native_trajectories_executed=True,
        real_native_import_and_parent_source_check=True,
        synthetic_adapted_DEV_fixture_is_not_science=True,
        formal_worlds_qualified_for_dispatch=32, workers=1,
        estimate_wall_minutes=q['conservative_32_world_single_worker_wall_minutes']), exclusive=True)
    print(json.dumps(read(OPS/'QUALIFICATION.json'),indent=2))


if __name__=='__main__':
    main()
