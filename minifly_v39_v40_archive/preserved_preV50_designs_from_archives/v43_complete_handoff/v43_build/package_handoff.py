"""Package the completed V43 deliverables; refuses missing or incomplete work."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

CORE_SHA = 'fb8ebb5355ea03e79d6049e5498db60772b18aa2bd51a54beed0689266a77bd8'
COMPLETION_SHA = '456daa818b500a6d9ad128e52b43451626a691f674c7141d018cc605071d4b2d'
DIAGNOSTICS_SHA = '98f88ef5b5f17efe0982f0876faae599a76d4e85fff22fde84e507c9bfd133f9'
SUPPLEMENTAL = ['ALL_CELLS_SUMMARY.csv', 'ALL_CELLS_SUMMARY.json', 'V43_all_cells.png',
                'EXPOSURE_SCOPE.json', 'EXPOSURE_SCOPE.md']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_file(path, expected):
    if not path.is_file() or digest(path.read_bytes()) != expected:
        raise RuntimeError('Missing or stale source: ' + str(path))


def verify_raw_archive(results, sources):
    """Refuse a raw archive that predates supplements or analysis corrections."""
    archive = results.with_name(results.name + '_results.zip')
    required = set(SUPPLEMENTAL) | set(sources) | {
        'REPORT.json', 'PREDICTABILITY_REPORT.json', 'ANALYSIS_IMPLEMENTATION.json',
        'DIAGNOSTIC_ENCODING_FIX.md', 'DIAGNOSTIC_ENCODING_FIX.json', 'PROTOCOL_ERRATUM.md',
    }
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(set(names)) != len(names) or z.testzip() is not None:
            raise RuntimeError('Raw archive has duplicate names or failed CRC')
        manifest = json.loads(z.read('ARTIFACT_HASHES.json'))
        if set(names) != set(manifest) | {'ARTIFACT_HASHES.json'} or not required <= set(manifest):
            raise RuntimeError('Raw archive manifest lacks final reports or has incomplete coverage')
        current_files = set()
        for path in results.rglob('*'):
            if not path.is_file() or 'locks' in path.parts or '.tmp-' in path.name or path.name == 'ARTIFACT_HASHES.json':
                continue
            if 'engineering' in path.parts and '.checkpoint' in path.name:
                engineering_record = path.parent / (path.name.split('.checkpoint')[0] + '.json')
                if engineering_record.exists():
                    continue
            elif 'checkpoints' in path.parts:
                cell_name = path.name.split('.checkpoint')[0]
                if (results / 'completed' / (cell_name + '.json')).exists():
                    continue
            current_files.add(str(path.relative_to(results)))
        if current_files != set(manifest):
            raise RuntimeError('Raw archive predates files in the final result directory; regenerate it after supplemental reports')
        if json.loads((results / 'ARTIFACT_HASHES.json').read_bytes()) != manifest:
            raise RuntimeError('Raw archive and on-disk manifests disagree')
        for name, expected in manifest.items():
            raw = z.read(name)
            path = (results / name).resolve()
            if not path.is_relative_to(results):
                raise RuntimeError('Raw archive path escapes result directory')
            if digest(raw) != expected or not path.is_file() or digest(path.read_bytes()) != expected:
                raise RuntimeError('Raw archive is corrupt or stale: ' + name)
        for name, expected in sources.items():
            if manifest.get(name) != expected:
                raise RuntimeError('Unexpected archived implementation identity: ' + name)
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--output', default='v43_complete_handoff.zip')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent.parent
    results = Path(args.out_dir).resolve()
    plan = json.loads((results / 'PLAN.json').read_bytes())
    expected_cells = {(s, l) for s in range(43101, 43117) for l in (64, 128, 256, 512)}
    if plan.get('mode') != 'science' or plan.get('source_sha256') != CORE_SHA or len(plan['jobs']) != 64 or {(j['seed'], j['load']) for j in plan['jobs']} != expected_cells:
        raise RuntimeError('Handoff requires the original frozen64-cell science plan')
    report = json.loads((results / 'REPORT.json').read_bytes())
    if report['status'] != 'complete' or report['completed_physical_trajectories'] != 64:
        raise RuntimeError('A complete handoff requires all 64 physical trajectories')
    diagnostics = json.loads((results / 'PREDICTABILITY_REPORT.json').read_bytes())
    if diagnostics.get('status') != 'complete' or diagnostics.get('fold_count') != 48 or set(diagnostics.get('seeds', [])) != set(range(43101, 43117)) or set(diagnostics.get('loads', [])) != {64, 128, 256, 512}:
        raise RuntimeError('A complete handoff requires all48 diagnostic folds')
    if diagnostics.get('new_learner_updates') != 0 or diagnostics.get('qualification_use') is not False:
        raise RuntimeError('Diagnostic scope unexpectedly changes scientific work')
    source = base / 'elm_v43_fast_slow_authority.py'
    completion = base / 'elm_v43_fast_slow_authority_complete.py'
    corrected = base / 'v43_build/diagnostics_corrected.py'
    for path, expected in [(source, CORE_SHA), (completion, COMPLETION_SHA), (corrected, DIAGNOSTICS_SHA)]:
        verify_file(path, expected)
    analysis = json.loads((results / 'ANALYSIS_IMPLEMENTATION.json').read_bytes())
    expected_identities = {'scientific_core_sha256': CORE_SHA, 'completion_driver_sha256': COMPLETION_SHA,
                           'corrected_diagnostics_sha256': DIAGNOSTICS_SHA}
    if any(analysis.get(k) != v for k, v in expected_identities.items()):
        raise RuntimeError('Unexpected final analysis implementation identity')
    if any(analysis.get(k) is not False for k in ['physical_learning_changes', 'policy_scoring_changes', 'model_or_hyperparameter_changes', 'gate_changes']):
        raise RuntimeError('Completion provenance does not preserve frozen science')
    verification = json.loads((base / 'v43_build/completion_verification.json').read_bytes())
    if verification.get('passed') is not True or any(verification.get(k) != v for k, v in expected_identities.items()):
        raise RuntimeError('Missing successful verification of the final completion implementation')
    if not verification.get('checks') or any(c.get('passed') is not True for c in verification['checks']):
        raise RuntimeError('Incomplete completion-driver verification evidence')
    sources = {'elm_v43_fast_slow_authority.py': CORE_SHA,
               'elm_v43_fast_slow_authority_complete.py': COMPLETION_SHA,
               'diagnostics_corrected.py': DIAGNOSTICS_SHA}
    for name, expected in sources.items():
        verify_file(results / name, expected)
    raw_archive = verify_raw_archive(results, sources)
    paths = {
        'elm_v43_fast_slow_authority_complete.py': completion,
        'elm_v43_fast_slow_authority.py': source,
        'v43_README.md': base / 'v43_README.md',
        'v43_analysis_report.md': base / 'v43_analysis_report.md',
        'v43_protocol.md': base / 'v43_protocol.md',
        'PROTOCOL_ERRATUM.md': base / 'v43_build/PROTOCOL_ERRATUM.md',
        'ORIGINAL_DESIGN.md': base / 'v43_build/ORIGINAL_DESIGN.md',
        'v43_results.zip': raw_archive,
        'V43_results.png': results / 'V43_results.png',
        'V43_all_cells.png': results / 'V43_all_cells.png',
        'v43_final_engineering.log': base / 'v43_final_engineering.log',
        'v43_final_run.log': base / 'v43_final_run.log',
        'v43_final_diagnostics.log': base / 'v43_final_diagnostics.log',
        'v43_final_completed_rerun.log': base / 'v43_final_completed_rerun.log',
    }
    names = [
        'PROTOCOL.md', 'ORIGINAL_DESIGN.md',
        'engine.py', 'offline_controls.py', 'reporting.py', 'diagnostics.py',
        'runner_template.py', 'build_runner.py', 'verify_cli.py',
        'completion_template.py', 'build_completion.py', 'diagnostics_corrected.py',
        'diagnostic_encoding_fix.md', 'diagnostic_encoding_fix.json',
        'completion_verification.py', 'completion_verification.json', 'completion_verification.md',
        'verification.json', 'cli_verification.json', 'cli_verification_run.log',
        'engine_verification.md', 'runner_review.md', 'reporting_independent_review.md',
        'antecedent_analysis.md', 'audit_scientific_results.py', 'audit_offline_controls.py',
        'snapshot_progress.py', 'snapshot_verification.json', 'all_cells_summary.py',
        'analysis_digest.py',
        'snapshot_progress_complete.py', 'snapshot_completion_verification.json', 'snapshot_completion_verification.md',
        'exposure_scope_summary.py', 'exposure_scope_verification.json',
        'monitor_progress.py', 'package_handoff.py', 'package_handoff_verification.json',
    ]
    for name in names:
        paths['v43_build/' + name] = base / 'v43_build' / name
    for name in ['v41_legacy.py', 'v41_claude.py', 'v41_hybrid.py']:
        paths['v43_sources/' + name] = base / 'v43_sources' / name
    for path in sorted((base / 'v43_build').glob('final_*audit*')):
        if path.is_file():
            paths['verification/' + path.name] = path
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    hashes = {name: digest(raw) for name, raw in sorted(payloads.items())}
    payloads['HANDOFF_SHA256.json'] = json.dumps(hashes, indent=2, sort_keys=True).encode() + b'\n'
    output = Path(args.output).resolve()
    temporary = output.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, raw in sorted(payloads.items()):
            z.writestr(name, raw)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(hashes) | {'HANDOFF_SHA256.json'}
        assert all(digest(z.read(name)) == value for name, value in hashes.items())
    temporary.replace(output)
    print(json.dumps({'output': str(output), 'members': len(payloads), 'bytes': output.stat().st_size,
                      'sha256': digest(output.read_bytes()), 'verified': True}, indent=2))


if __name__ == '__main__':
    main()
