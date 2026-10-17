"""Reassemble published immutable result archive without any experiment execution."""
from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent;m=json.loads((p/'MANIFEST.json').read_text());out=p.parent/m['original']
if out.exists():raise SystemExit('Output exists;preserved unchanged')
h=hashlib.sha256()
with out.with_suffix('.zip.tmp').open('wb') as f:
 for row in m['parts']:
  data=(p/row['file']).read_bytes()
  if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Changed part:'+row['file'])
  f.write(data);h.update(data)
if h.hexdigest()!=m['sha256']:raise ValueError('Archive checksum mismatch')
out.with_suffix('.zip.tmp').replace(out);print(out)
