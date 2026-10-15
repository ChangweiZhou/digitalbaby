"""Reassemble the published archive only; never import or run a learner."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'SPLIT_MANIFEST.json').read_text())
    if a.output.exists():
        raise SystemExit('Output already exists; choose a new output path. Existing files are preserved.')
    temporary = a.output.with_name(a.output.name + '.assembling')
    if temporary.exists():
        raise SystemExit('Partial output exists; inspect it before trying again.')
    h = hashlib.sha256()
    with temporary.open('xb') as out:
        for part in manifest['parts']:
            path = root / part['name']
            if path.stat().st_size != part['bytes']:
                raise ValueError('Part size mismatch: ' + part['name'])
            with path.open('rb') as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    out.write(block)
                    h.update(block)
    if temporary.stat().st_size != manifest['bundle_bytes'] or h.hexdigest() != manifest['bundle_sha256']:
        raise ValueError('Archive mismatch; partial file preserved for inspection')
    temporary.replace(a.output)
    print(str(a.output))


if __name__ == '__main__':
    main()
