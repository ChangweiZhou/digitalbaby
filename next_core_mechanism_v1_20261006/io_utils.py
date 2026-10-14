import hashlib,json,os,tempfile,gzip
from pathlib import Path
def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        d=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(d)
        finally: os.close(d)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def atomic_json(path,x): atomic_bytes(path,canonical(x))
def write_receipt(path,x):
    doc=dict(x);doc['seal']=digest(doc)
    atomic_bytes(path,gzip.compress(canonical(doc),mtime=0))
def read_receipt(path):
    d=json.loads(gzip.decompress(Path(path).read_bytes()));s=d.pop('seal')
    if digest(d)!=s:raise ValueError('receipt seal')
    return d
