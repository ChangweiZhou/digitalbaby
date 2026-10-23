"""Portable, exact field extraction from already completed development receipts."""
import gzip,hashlib,json
from geometry import ROOT

def load_history():
    root=ROOT/'data';data=(root/'development_history.json.gz').read_bytes()
    manifest=json.loads((root/'PROVENANCE.json').read_text())
    if hashlib.sha256(data).hexdigest()!=manifest['bundle_sha256']:raise ValueError('development input hash mismatch')
    d=json.loads(gzip.decompress(data))
    return {r['world']:r for r in d['worlds']},d['provenance']
