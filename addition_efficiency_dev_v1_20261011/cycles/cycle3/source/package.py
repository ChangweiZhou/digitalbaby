"""Packaging utility: no model, selection, launch or upload."""
import hashlib,json,zipfile
from pathlib import Path
def package_files(files,out,verdict,science_worlds=0):
    out=Path(out)
    if out.exists():raise ValueError('existing package; no overwrite')
    files={name:Path(p) for name,p in files.items()}
    if any(not p.is_file() for p in files.values()):raise ValueError('missing package input')
    if any(n.startswith('/') or '..' in Path(n).parts for n in files):raise ValueError('unsafe archive name')
    meta={'verdict':verdict,'science_worlds':science_worlds,'files':{n:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for n,p in files.items()}}
    tmp=out.with_name(out.name+'.pending')
    with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(meta,indent=2))
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise AssertionError('zip integrity')
        for n,p in files.items():
            if z.read(n)!=p.read_bytes():raise AssertionError('zip content parity')
    tmp.rename(out)
    return {'verdict':'PASS','result_verdict':verdict,'bytes':out.stat().st_size,'files':len(files),'science_worlds':science_worlds}
