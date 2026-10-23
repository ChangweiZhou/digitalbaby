# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure terminal-analysis certificate validation; no learner or worker imports."""
import json
import math
from pathlib import Path

from durable import sha
from integrity import digest_map

WORLDS = tuple(range(310001, 310065))
SCOPE = 'taught-dependent held-out table completion in the locked relabeled Latin-table fixture only'
CAP = 900.0
RSS = 768 * 1024**2


def check(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def inputs(root, expected_source, runtime, *, receipt_loader=None):
    """Return independently revalidated complete cohort plus bound manifest map."""
    root = Path(root)
    lock_path = root / 'protocol/OFFICIAL_LOCK.json'
    check(read(lock_path)['hashes'] == expected_source, 'analysis official source lock mismatch')
    acceptance = sha(lock_path)
    directory = root / 'receipts/science'
    check(directory.is_dir(), 'analysis requires all64 registered worlds')
    check({p.name for p in directory.iterdir()} == {str(w) for w in WORLDS}, 'analysis requires all64 canonical registered worlds only')
    for p in directory.iterdir():
        check(p.is_dir() and not p.is_symlink(), 'analysis input must be canonical local world directory')
    if receipt_loader is None:
        from validate_receipt import load as validate_world
        def receipt_loader(path, world, source):
            return validate_world(path, world, source, expected_kind='science', expected_acceptance=acceptance)
    persistence = read(root / 'operations/PERSISTENCE_STATE.json')
    values = []
    manifests = {}
    for world in WORLDS:
        value = receipt_loader(directory / str(world), world, expected_source)
        check(value['manifest']['kind'] == 'science', 'analysis input must be science receipt')
        check(value['header']['runtime'] == runtime, 'analysis input runtime mismatch')
        check(value['header']['acceptance_sha256'] == acceptance, 'analysis input official lock mismatch')
        h = value['manifest_sha256']
        check(h == sha(directory / str(world) / 'manifest.json'), 'analysis input manifest identity mismatch')
        barrier = persistence.get('worlds', {}).get(str(world))
        check(barrier and barrier.get('manifest_sha256') == h and barrier.get('private_readback_verified') is True and barrier.get('github_tree_verified') is True,
              'analysis input awaiting private/public persistence barrier')
        values.append(value)
        manifests[str(world)] = h
    return {'validated': values, 'input_manifest': manifests, 'lock_sha256': acceptance}


def expected_result(validated, manifests):
    # Invoke the frozen functions; this wrapper introduces no new estimator,
    # threshold, routing rule, contrast, sample size or statistical family.
    from analyze import world_values, intervals, decide, N, M, FAMILY_ALPHA
    check(N == 64 and M == 11, 'analysis registered family changed')
    rows = [world_values(value['probes']) for value in validated]
    ci = intervals(rows)
    return {'schema': 'RC-SURVIVOR-R1-ANALYSIS-v1', 'worlds': list(WORLDS),
            'family_m': M, 'alpha': FAMILY_ALPHA, 'input_manifest': manifests,
            'world_values': rows, 'intervals': ci, 'decision': decide(ci), 'scope': SCOPE}


def load(root, directory, expected_source, runtime, *, attempt=None, receipt_loader=None):
    root = Path(root)
    p = Path(directory)
    check(p.resolve() == (root / 'results/final').resolve() and not p.is_symlink(), 'noncanonical final analysis directory')
    check({f.name for f in p.iterdir()} == {'analysis.json', 'manifest.json'}, 'unexpected/partial final analysis files')
    check(all(f.is_file() and not f.is_symlink() for f in p.iterdir()), 'analysis files must be local regular files')
    manifest = read(p / 'manifest.json')
    check(manifest.get('schema') == 'RC-SURVIVOR-ANALYSIS-RECEIPT-v1' and manifest.get('complete') is True,
          'incomplete analysis receipt')
    check(manifest.get('kind') == 'analysis' and manifest.get('key') == 'analysis/final' and manifest.get('native_teaching_events') == 0,
          'analysis receipt job/kind mismatch')
    check(manifest.get('source') == expected_source and manifest.get('source_digest') == digest_map(expected_source), 'analysis source mismatch')
    check(manifest.get('runtime') == runtime, 'analysis runtime mismatch')
    check(type(manifest.get('attempt')) is int and 1 <= manifest['attempt'] <= 3, 'analysis attempt identity missing')
    reservation = manifest.get('reservation_id')
    check(isinstance(reservation, str) and len(reservation) == 32 and all(c in '0123456789abcdef' for c in reservation), 'analysis reservation identity missing')
    if attempt is not None:
        check(manifest['attempt'] == attempt['attempt'] and reservation == attempt['reservation_id'] and attempt['key'] == 'analysis/final', 'analysis receipt attempt identity mismatch')
    check(set(manifest.get('parts', {})) == {'analysis.json'}, 'analysis part map mismatch')
    part = manifest['parts']['analysis.json']
    check((p / 'analysis.json').stat().st_size == part['bytes'] <= 512 * 1024 and sha(p / 'analysis.json') == part['sha256'], 'analysis part hash/size mismatch')
    measured = manifest['resources']
    check(set(measured)=={'wall_s','peak_rss_bytes'} and type(measured['wall_s']) in (int,float) and math.isfinite(measured['wall_s']) and 0 < measured['wall_s'] <= CAP and type(measured['peak_rss_bytes']) is int and 0 < measured['peak_rss_bytes'] <= RSS, 'analysis time/RSS cap')
    cohort = inputs(root, expected_source, runtime, receipt_loader=receipt_loader)
    check(manifest['lock_sha256'] == cohort['lock_sha256'], 'analysis receipt official lock mismatch')
    check(manifest['input_manifest'] == cohort['input_manifest'], 'analysis receipt cohort identity mismatch')
    result = read(p / 'analysis.json')
    check(result == expected_result(cohort['validated'], cohort['input_manifest']), 'analysis semantics differ from complete-cohort frozen formulas/decision')
    return {'manifest': manifest, 'manifest_sha256': sha(p / 'manifest.json'), 'result': result,
            'header': {'runtime': runtime, 'source': expected_source}, 'parts_count': 1}
