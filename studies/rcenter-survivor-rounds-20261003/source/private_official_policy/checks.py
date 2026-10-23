# SPDX-License-Identifier: GPL-3.0-or-later
"""Focused P2 policy tests on tiny synthetic files; no learner or real state."""
import copy
import json
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import adapter as a
from durable import create, replace
from integrity import digest_map


def run_checks(directory):
    root = Path(directory) / 'official-policy-tests';root.mkdir(parents=True, exist_ok=False)
    checks = []
    def reject(fn):
        try:fn()
        except (RuntimeError, OSError, KeyError, ValueError, AssertionError):return
        raise AssertionError('official policy guard accepted forbidden input')
    def plan():
        return {'completed': list(a.CYCLE_KEYS), 'missing': list(a.KEYS), 'blocked': ['unresolved_transport'],
                'pending_barriers': [], 'next_key': None}
    public = root / 'operations/TRANSPORT_PENDING.json'
    for phase in ('public_tree', 'public_commit'):
        replace(public, {'phase': phase, 'status': 'uncertain'})
        before = public.read_bytes();result = a.filter_plan(root, plan())
        assert result['next_key'] == 'science/310001' and result['publication_deferred'] and public.read_bytes() == before
    replace(public, {'phase': 'private_upload'})
    assert a.filter_plan(root, plan())['next_key'] is None
    replace(public, {'phase': 'public_commit'})
    replace(root / a.STATE / 'TRANSPORT_PENDING.json', {'phase': 'private_readback'})
    assert 'unresolved_private_transport' in a.filter_plan(root, plan())['blocked']
    replace(root / a.STATE / 'TRANSPORT_PENDING.json', None)
    p = plan();p['blocked'].append('total worker cap');assert a.filter_plan(root, p)['next_key'] is None
    p = plan();p['completed'].remove('cycle3/replay');reject(lambda:a.filter_plan(root,p))
    assert a.KEYS == tuple(f'science/{w}' for w in range(310001,310065)) + ('analysis/final',)
    checks.append('official_fixed64_then_analysis_public_only_deferral_and_private_budget_blocks')

    create(root / 'source/runtime/fake.py', {'frozen': True})
    source = {'source/runtime/fake.py': a.sha(root / 'source/runtime/fake.py')}
    create(root / 'source/private_policy/fake.py', {'historical': True})
    create(root / a.p1.POLICY_DIR / 'REPORT.md', {'historical_review': True})
    historical = {'schema':'RC-CYCLE3-PRIVATE-POLICY-v1','accepted':True,'independent_review':True,
                  'base_source_digest':digest_map(source),'policy_hashes':a.p1.policy_hashes(root),
                  'policy_digest':digest_map(a.p1.policy_hashes(root)), 'allowed_keys':list(a.p1.KEYS), 'mode':'cycle3',
                  'report_file':a.p1.POLICY_DIR+'/REPORT.md','report_sha256':a.sha(root/a.p1.POLICY_DIR/'REPORT.md'),
                  'original_transport_sha256':'a'*64}
    create(root / a.p1.POLICY_DIR / 'POLICY_ACCEPTED.json', historical)
    create(root / 'source/private_official_policy/fake.py', {'official_adapter':True})
    create(root / a.STATE / 'REPORT.md', {'official_review':True})
    policy = {'schema':'RC-OFFICIAL-PRIVATE-POLICY-v1','accepted':True,'independent_review':True,
              'base_source_digest':digest_map(source),'policy_hashes':a.policy_hashes(root),
              'policy_digest':digest_map(a.policy_hashes(root)), 'allowed_keys':list(a.KEYS),'mode':'science',
              'private_backup_required':True,'github_requirement':'deferred_not_acknowledged',
              'cycle3_policy_sha256':a.sha(root/a.p1.POLICY_DIR/'POLICY_ACCEPTED.json'),
              'report_file':a.STATE+'/REPORT.md','report_sha256':a.sha(root/a.STATE/'REPORT.md')}
    create(root / a.STATE / 'POLICY_ACCEPTED.json', policy)
    with patch.object(a,'BASE_DIGEST',digest_map(source)), patch.object(a,'hashes',return_value=source):
        reject(lambda:a.verify_policy(root))
        accepted={}
        for c in (1,2,3):
            rel=f'audits/CYCLE{c}_ACCEPTED.json';create(root/rel,{'accepted':True});accepted[rel]=a.sha(root/rel)
        create(root/'protocol/OFFICIAL_LOCK.json',{'hashes':source,'acceptances':accepted})
        reject(lambda:a.verify_policy(root))
        create(root/'audits/LAUNCH_ACCEPTED.json',{'accepted':False,'lock_sha256':a.sha(root/'protocol/OFFICIAL_LOCK.json')})
        reject(lambda:a.verify_policy(root))
        replace(root/'audits/LAUNCH_ACCEPTED.json',{'accepted':True,'lock_sha256':a.sha(root/'protocol/OFFICIAL_LOCK.json')})
        assert a.verify_policy(root)==policy
        # Today's moving public queue never relabels or invalidates historical P1.
        assert a.historical_policy(root)==historical
        hpath=root/'source/private_policy/fake.py';saved=hpath.read_bytes();hpath.write_bytes(b'changed');reject(lambda:a.verify_policy(root));hpath.write_bytes(saved)
        reject(lambda:a.prelaunch_guard(root,policy))
        create(root/'operations/PERSISTENCE_STATE.json',{'prelaunch_verified':True,'prelaunch_source_digest':digest_map(source),
                  'prelaunch_persistence_policy':'verified-private-only-v1','private_official_policy_digest':policy['policy_digest']})
        a.prelaunch_guard(root,policy)
        checks += ['official_missing_lock_or_parent_launch_acceptance_fails_closed',
                   'historical_p1_retains_own_identity_despite_current_public_phase_changes',
                   'truthful_private_prelaunch_metadata_required_without_github_flags']

        # Old per-job ZIP is intentionally absent; current genuine archive must
        # cover the exact immutable payload. No old path is opened or asserted.
        create(root/'receipts/science/310001/part.json',{'synthetic':True})
        part=root/'receipts/science/310001/part.json'
        create(root/'receipts/science/310001/manifest.json',{'parts':{'part.json':{'sha256':a.sha(part)}}})
        receipt='receipts/science/310001/manifest.json';required={receipt:a.sha(root/receipt),'receipts/science/310001/part.json':a.sha(part)}
        archive=root/'backups/current.zip';archive.parent.mkdir()
        manifest={'content_identity':'b'*64,'all_payload_sha256':required}
        with zipfile.ZipFile(archive,'w') as z:
            for rel in required:z.write(root/rel,rel)
            z.writestr('CHECKPOINT_MANIFEST.json',json.dumps(manifest))
        state={'status':'private_backup_readback_verified','library_file_id':'mock-owned','version':20,'local_path':'backups/current.zip',
               'sha256':a.sha(archive),'content_identity':'b'*64}
        create(root/'operations/CHECKPOINT_STATE.json',state)
        assert a.verify_coverage(root,required)['version']==20 and not (root/'backups/retired-old.zip').exists()
        reject(lambda:a.verify_coverage(root,{receipt:'f'*64}))
        checks.append('current_verified_archive_covers_immutable_receipts_without_old_zip_dependency')

        # Exact original inputs() structure/source/runtime/64-world rules remain;
        # only its one operational publication predicate is swapped in memory.
        import validate_analysis
        original=validate_analysis.inputs
        science=root/'receipts/science'
        for w in a.WORLDS:
            p=science/str(w)/'manifest.json'
            if not p.exists():create(p,{'world':w})
        runtime={'python':'synthetic'};gate_calls=[]
        def loader(path,world,expected_source):
            assert expected_source==source
            return {'manifest':{'kind':'science'},'manifest_sha256':a.sha(Path(path)/'manifest.json'),
                    'header':{'runtime':runtime,'acceptance_sha256':a.sha(root/'protocol/OFFICIAL_LOCK.json')},'probes':[]}
        def private_gate(root_arg,world,h):
            assert Path(root_arg)==root and h==a.sha(science/str(world)/'manifest.json');gate_calls.append(world)
        with patch.object(a,'private_input_gate',side_effect=private_gate):
            with a.analysis_gate(root):
                result=validate_analysis.inputs(root,source,runtime,receipt_loader=loader)
                assert list(result['input_manifest'])==[str(w) for w in a.WORLDS] and gate_calls==list(a.WORLDS)
                last=science/'310064';last.rename(science/'bad-name')
                reject(lambda:validate_analysis.inputs(root,source,runtime,receipt_loader=loader))
                (science/'bad-name').rename(last)
            assert validate_analysis.inputs is original
        with patch.object(a,'private_input_gate',side_effect=RuntimeError('private receipt not covered')):
            with a.analysis_gate(root):reject(lambda:validate_analysis.inputs(root,source,runtime,receipt_loader=loader))
        checks.append('analysis_ast_replaces_only_one_operational_gate_preserves_exact64_and_private_rejection')
    return checks


if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='rcenter-official-policy-',dir='/tmp') as d:
        print(json.dumps({'passed':True,'checks':run_checks(d),'native_events':0}))
