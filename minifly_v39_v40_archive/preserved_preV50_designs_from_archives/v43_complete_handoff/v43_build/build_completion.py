#!/usr/bin/env python3
"""Build the self-contained analysis-corrected V43 entrypoint."""
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parent.parent
files = {
    'core': ROOT / 'elm_v43_fast_slow_authority.py',
    'diagnostics': ROOT / 'v43_build/diagnostics_corrected.py',
    'fix_md': ROOT / 'v43_build/diagnostic_encoding_fix.md',
    'fix_json': ROOT / 'v43_build/diagnostic_encoding_fix.json',
    'erratum': ROOT / 'v43_build/PROTOCOL_ERRATUM.md',
}
payloads = {}
for name, path in files.items():
    raw = path.read_bytes()
    payloads[name] = {'sha256': hashlib.sha256(raw).hexdigest(),
                      'data': base64.b85encode(zlib.compress(raw, 9)).decode()}
assert payloads['core']['sha256'] == 'fb8ebb5355ea03e79d6049e5498db60772b18aa2bd51a54beed0689266a77bd8'
assert payloads['diagnostics']['sha256'] == '98f88ef5b5f17efe0982f0876faae599a76d4e85fff22fde84e507c9bfd133f9'
template = (ROOT / 'v43_build/completion_template.py').read_text()
assert template.count('__COMPLETION_PAYLOADS__') == 1
target = ROOT / 'elm_v43_fast_slow_authority_complete.py'
target.write_text(template.replace('__COMPLETION_PAYLOADS__', repr(payloads)))
print(json.dumps({'file': str(target), 'bytes': target.stat().st_size,
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'payload_hashes': {k:v['sha256'] for k,v in payloads.items()}}, indent=2))
