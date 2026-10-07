"""Archived DEV-data doubles test final reducer/ZIP, with zero native executions."""
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile
import ops_common as common
from ops_common import OPS, ROOT, WORLDS, EVIDENCE, atomic, read, sha
import analysis
import supervisor


def main():
    fixture = read(ROOT/'cycles/cycle3/WORLD_810201.json')
    with tempfile.TemporaryDirectory(prefix='observation_finish_test_') as tmp:
        result = Path(tmp)/'results'
        index = {}
        for world in WORLDS:
            r = copy.deepcopy(fixture)
            r.update(world=world,evidence=EVIDENCE,confirmation_source_lock_sha256='TECHNICAL_FIXTURE')
            r['independent_audit'] = dict(verdict='PASS',independently_checked_rows=8240,
                                        confirmation_source_lock_sha256='TECHNICAL_FIXTURE')
            path = result/'receipts'/f'WORLD_{world}.json.gz'
            atomic(path,r,exclusive=True)
            index[str(world)] = {'sha256':sha(path)}
        atomic(result/'COMMIT_INDEX.json',index)
        with patch.object(common,'RESULTS',result),patch.object(analysis,'RESULTS',result), \
             patch.object(supervisor,'RESULTS',result),patch.object(analysis,'verify_sources',lambda:'TECHNICAL_FIXTURE'):
            s=analysis.analyze()
            assert s['verdict']=='ADOPT_AUTONOMOUS_OBSERVATION_BASELINE'
            assert s['committed_training_lives']==96 and s['total_learning_and_probe_rows']==263680
            assert read(result/'FINAL_AUDIT.json')['status']=='ACCEPTED_COMPLETE_EXPERIMENT'
            supervisor.package(s)
            p=read(result/'PACKAGE.json')
            assert p['status']=='COMPLETE' and sha(result/'RESULT_BUNDLE.zip')==p['sha256']
            with zipfile.ZipFile(result/'RESULT_BUNDLE.zip') as z:
                assert z.testzip() is None
                names=z.namelist()
                assert 'results/confirm/REPORT.md' in names and 'results/confirm/FINAL_AUDIT.json' in names
                assert sum(n.startswith('results/confirm/receipts/WORLD_') for n in names)==32
            # A changed committed receipt must block final reduction, rather than publish it.
            (result/'receipts'/f'WORLD_{WORLDS[0]}.json.gz').write_bytes(b'tampered')
            try:
                analysis.analyze()
            except ValueError as exc:
                assert 'changed' in str(exc)
            else:
                raise AssertionError('final reducer accepted changed commit')
    atomic(OPS/'FINISH_TEST.json',dict(verdict='PASS',archived_DEV_doubles_only=True,
        native_training_executions=0, synthetic_results_not_science=True,
        full_roster_reduction_and_final_audit=True, real_packaging_and_zip_read=True,
        changed_committed_receipt_rejected=True),exclusive=True)
    print('PASS: final analysis/audit/package on isolated synthetic receipt doubles; native executions=0')


if __name__=='__main__':
    main()
