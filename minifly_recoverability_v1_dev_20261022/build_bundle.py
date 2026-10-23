"""Package existing artifacts only. Never imports native code or a learner."""
from pathlib import Path
import hashlib
import json
import time
import zipfile

P = Path(__file__).resolve().parent
R = P.parent
D = R / 'minifly_recoverability_pilot_design_v1_20261022'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = time.monotonic()
    authority = json.loads((D / 'SOURCE_LOCK.json').read_text())
    for name, expected in authority['sources'].items():
        assert sha(R / name) == expected, name
    for name, expected in authority['documents'].items():
        assert sha(D / name) == expected, name
    sources = [x for x in P.rglob('*') if x.is_file()
               and not {'scratch', '__pycache__'}.intersection(x.relative_to(P).parts)
               and x.name not in {'RESULT_BUNDLE.zip', 'PACKAGE.json', 'FULL_ARTIFACT_MANIFEST.json', 'SOURCE_LOCK.json'}]
    lock_files = [x for x in sources if x.suffix in {'.py', '.md'} or x.name in {'AUTHORIZATION.json', 'ENVIRONMENT.json'}]
    delivery_lock = dict(scope='Post-execution delivery lock; prospective cycle snapshots remain in cycles/',
        files={str(x.relative_to(P)): sha(x) for x in sorted(lock_files)},
        unchanged_parent_sources=authority['sources'])
    (P / 'SOURCE_LOCK.json').write_text(json.dumps(delivery_lock, indent=2) + '\n')
    sources.append(P / 'SOURCE_LOCK.json')
    sources += [x for x in D.iterdir() if x.is_file()]
    sources += [R / name for name in authority['sources']]
    sources = sorted(set(sources))
    entries = []
    for x in sources:
        rel = x.relative_to(R)
        compact = not (x.is_relative_to(P / 'checkpoints') or x.name.endswith('_decoded.npz'))
        entries.append(dict(path=str(rel), bytes=x.stat().st_size, sha256=sha(x), included_in_compact_bundle=compact))
    manifest = dict(schema='RECOVERABILITY_V1_FULL_ARTIFACT_MANIFEST',
        excludes=['runtime caches', 'this self-referential manifest', 'archive itself', 'PACKAGE.json (external archive receipt)'],
        omitted_from_compact_only='Native checkpoints and redundant derived logits remain individual repository files', files=entries)
    (P / 'FULL_ARTIFACT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    target = P / 'RESULT_BUNDLE.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for entry in entries:
            if entry['included_in_compact_bundle']:
                z.write(R / entry['path'], entry['path'])
        z.write(P / 'FULL_ARTIFACT_MANIFEST.json', f'{P.name}/FULL_ARTIFACT_MANIFEST.json')
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        count = len(z.namelist())
    package = dict(status='COMPLETE', bundle=target.name, bytes=target.stat().st_size, sha256=sha(target),
        zip_entries=count, full_artifact_files=len(entries), compact_archive=True,
        all_files_retained_individually_in_repository=True, CRC_verified=True,
        arithmetic_pointer_unchanged=True, native_or_head_training=False,
        packaging_active_wall_seconds=time.monotonic()-start)
    (P / 'PACKAGE.json').write_text(json.dumps(package, indent=2) + '\n')
    print(json.dumps(package, indent=2))


if __name__ == '__main__':
    main()
