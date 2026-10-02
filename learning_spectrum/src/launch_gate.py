"""Fail-closed preflight gate. No simulation-launch implementation is included."""
import hashlib,json
from pathlib import Path
REQUIRED=('static_spectrum','finite_formation','finite_final','shared_geometry','total_margin_prediction','three_audited_cycles','resource_coordination')

def verify_gate(path,expected_sha256):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha256:raise ValueError('gate manifest digest mismatch')
    g=json.loads(data)
    if set(g['gates'])!=set(REQUIRED):raise ValueError('missing or extra gates')
    if any(g['gates'][k] is not True for k in REQUIRED):raise ValueError('preflight failed: full run prohibited')
    if g.get('total_margin_prediction') in (None,'NO QUANTITATIVE THEORY PREDICTION'):raise ValueError('no quantitative total forecast')
    return True

if __name__=='__main__':
    raise SystemExit('No fresh-world launcher supplied: this preflight failed necessary theoretical gates.')
