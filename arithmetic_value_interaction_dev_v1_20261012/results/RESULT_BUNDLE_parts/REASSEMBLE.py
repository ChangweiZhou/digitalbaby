from pathlib import Path
import json
p=Path(__file__).resolve().parent
m=json.loads((p/'SPLIT_MANIFEST.json').read_text())
out=p.parent/m['original_filename']
if out.exists():
    raise SystemExit(f'Refusing to overwrite existing {out}')
for part in m['parts']:
    f=p/part['filename']
    if not f.is_file() or f.stat().st_size!=part['bytes']:
        raise SystemExit(f'Missing or incomplete part: {f}')
with out.open('xb') as dst:
    for part in m['parts']:
        with (p/part['filename']).open('rb') as src:
            while block:=src.read(1024*1024):
                dst.write(block)
print(f'Reassembled {out} ({out.stat().st_size} bytes)')
