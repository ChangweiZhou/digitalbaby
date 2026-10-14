"""Prepare a lossless publication plan; no learner imports, runs or data edits."""
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LIMIT = 100 * 1024 * 1024
CHUNK = 45 * 1024 * 1024


def names(*args):
    raw = subprocess.run(['git', *args, '-z'], cwd=ROOT, check=True, capture_output=True).stdout
    return [os.fsdecode(x) for x in raw.split(b'\0') if x]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    tracked = set(names('ls-files'))
    untracked = names('ls-files', '--others', '--exclude-standard')
    dirty = names('diff', '--name-only')
    roots = {x.split('/')[0] for x in untracked}
    archive_locations = {}
    large = []
    for name in untracked:
        p = ROOT / name
        if p.is_file() and p.stat().st_size >= LIMIT:
            if p.suffix != '.zip':
                raise RuntimeError('Need a publication representation for oversized file: ' + name)
            large.append(name)
    for name in sorted(large):
        p = ROOT / name
        top = name.split('/')[0]
        parts = sorted(x for x in tracked if x.startswith(top + '/')
                       and Path(x).name.startswith(p.name + '.part'))
        if parts and sum((ROOT / x).stat().st_size for x in parts) == p.stat().st_size:
            archive_locations[name] = dict(kind='already_tracked_parts', parts=parts,
                                           bytes=p.stat().st_size)
            continue
        folder = p.parent / (p.stem + '_parts_20261014')
        folder.mkdir(parents=True, exist_ok=True)
        parts = []
        total_hash = hashlib.sha256()
        with p.open('rb') as f:
            index = 1
            while True:
                data = f.read(CHUNK)
                if not data:
                    break
                part = folder / (p.name + '.part' + str(index).zfill(3))
                part.write_bytes(data)
                total_hash.update(data)
                parts.append(dict(path=str(part.relative_to(ROOT)), bytes=len(data)))
                index += 1
        manifest = dict(original=str(p.relative_to(ROOT)), original_bytes=p.stat().st_size,
                        original_sha256=total_hash.hexdigest(), chunk_bytes=CHUNK, parts=parts,
                        scope='Transport split only. Original frozen experiment archive unchanged.')
        write(folder / 'SPLIT_MANIFEST.json', manifest)
        (folder / 'REASSEMBLE.py').write_text(
            'from pathlib import Path\nimport json,hashlib\n'
            'p=Path(__file__).resolve().parent\n'
            'm=json.loads((p/"SPLIT_MANIFEST.json").read_text())\n'
            'out=p.parent/Path(m["original"]).name\n'
            'if out.exists(): raise SystemExit("Output exists; preserving it: "+str(out))\n'
            'h=hashlib.sha256()\n'
            'with out.open("xb") as w:\n'
            ' for x in m["parts"]:\n'
            '  data=(p/Path(x["path"]).name).read_bytes()\n'
            '  assert len(data)==x["bytes"]\n'
            '  w.write(data);h.update(data)\n'
            'assert out.stat().st_size==m["original_bytes"]\n'
            'assert h.hexdigest()==m["original_sha256"]\n'
            'print(out)\n')
        archive_locations[name] = dict(kind='new_parts', parts=[x['path'] for x in parts],
                                      manifest=str((folder / 'SPLIT_MANIFEST.json').relative_to(ROOT)),
                                      bytes=p.stat().st_size)

    # Match archived binary payload copies by path, size and ZIP CRC. Text/source
    # files are published directly so amended notes and code remain browsable.
    by_path = collections.defaultdict(list)
    archive_count = 0
    all_names = set(tracked) | set(untracked)
    for name in sorted(all_names):
        p = ROOT / name
        if p.suffix != '.zip' or not p.is_file() or name.split('/')[0] not in roots:
            continue
        if name not in tracked and name not in archive_locations and p.stat().st_size >= LIMIT:
            continue
        archive_count += 1
        with zipfile.ZipFile(p) as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                q = Path(item.filename)
                if q.is_absolute() or '..' in q.parts:
                    continue
                top = Path(name.split('/')[0])
                options = {q, top / q, p.parent.relative_to(ROOT) / q}
                if len(q.parts) > 1:
                    options |= {top / Path(*q.parts[1:]), p.parent.relative_to(ROOT) / Path(*q.parts[1:])}
                entry = dict(archive=name, member=item.filename, size=item.file_size, crc=item.CRC)
                for option in options:
                    by_path[str(option)].append(entry)
    direct = set(dirty)
    covered = []
    binary_suffixes = {'.gz', '.npz', '.npy', '.pkl', '.pickle', '.bin', '.checkpoint', '.pt', '.pth'}
    for name in untracked:
        p = ROOT / name
        if not p.is_file():
            continue
        if name in archive_locations:
            covered.append(dict(path=name, representation=archive_locations[name]))
            continue
        candidates = [x for x in by_path.get(name, []) if x['size'] == p.stat().st_size]
        match = None
        if p.suffix in binary_suffixes and candidates:
            crc = 0
            with p.open('rb') as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    crc = zlib.crc32(block, crc)
            match = next((x for x in candidates if x['crc'] == crc), None)
        if match:
            covered.append(dict(path=name, archive=match['archive'], member=match['member'],
                                bytes=match['size'], coverage='same size and ZIP CRC; archive published whole or in parts'))
        else:
            if p.stat().st_size >= LIMIT:
                raise RuntimeError('Oversized direct file: ' + name)
            direct.add(name)
    # Include new split files and this publication's inventory after preparation.
    for loc in archive_locations.values():
        if loc['kind'] == 'new_parts':
            folder = ROOT / loc['manifest']
            direct.update(str(p.relative_to(ROOT)) for p in folder.parent.iterdir() if p.is_file())
    plan = dict(scope='All current nonignored uncommitted project work. Binary duplicates are available inside exact-byte-matching archived members; original local files are retained.',
                branch='codex/addition-u042-core-adoption-20261011', starting_commit=subprocess.run(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
                original_untracked_files=len(untracked), direct_files=sorted(direct),
                covered_payloads=covered, archive_locations=archive_locations,
                archives_indexed=archive_count, science_rerun=False,
                new_design_native_qualification=False, publication_state='PREPARED_NOT_YET_PUSHED')
    write(OUT / 'PUBLICATION_PLAN.json', plan)
    direct.update(str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file())
    (OUT / 'STAGE_PATHS.nul').write_bytes(b'\0'.join(os.fsencode(x) for x in sorted(direct)) + b'\0')
    payload_bytes = sum((ROOT / x).stat().st_size for x in direct)
    print(json.dumps(dict(original_files=len(untracked), direct_files=len(direct),
                          covered_copies=len(covered), direct_bytes=payload_bytes,
                          new_split_archives=sum(x['kind'] == 'new_parts' for x in archive_locations.values()),
                          existing_split_archives=sum(x['kind'] == 'already_tracked_parts' for x in archive_locations.values())), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
