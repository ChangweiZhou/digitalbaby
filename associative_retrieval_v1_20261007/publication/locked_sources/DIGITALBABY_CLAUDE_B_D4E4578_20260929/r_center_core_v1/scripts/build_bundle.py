"""Deterministic distribution; include receipts/docs/lock, omit caches/checkpoints."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from centered_core.integrity import verify_sources
verify_sources()
out = ROOT / 'releases' / 'R_CENTER_CORE_V1_20261003.zip'
out.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if not p.is_file() or any(x in rel.parts for x in ('scratch', '__pycache__', 'releases', 'dist')) or p.suffix == '.pyc': continue
        if p.is_symlink(): raise ValueError('symlinks not allowed')
        i = zipfile.ZipInfo('r_center_core_v1/' + rel.as_posix(), date_time=(2026, 10, 3, 0, 0, 0))
        i.compress_type = zipfile.ZIP_DEFLATED
        i.external_attr = 0o100644 << 16
        z.writestr(i, p.read_bytes())
print(json.dumps({'zip': str(out), 'bytes': out.stat().st_size, 'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))
