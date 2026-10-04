# SPDX-License-Identifier: GPL-3.0-or-later
"""Crash-consistent JSON/part writes for a single exclusively locked writer."""
import hashlib,json,os,uuid
from pathlib import Path

def canonical(d):return json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fsync_dir(p):
 fd=os.open(str(p),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def write_bytes(path,data,replace=False):
 p=Path(path);missing=[];ancestor=p.parent
 while not ancestor.exists():missing.append(ancestor);ancestor=ancestor.parent
 for d in reversed(missing):d.mkdir();fsync_dir(d.parent)
 p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with q.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 try:
  if replace:os.replace(q,p)
  else:os.link(q,p);q.unlink()
  fsync_dir(p.parent)
 finally:
  if q.exists():q.unlink();fsync_dir(q.parent)
 return sha(p)
def create(path,d):return write_bytes(path,canonical(d))
def replace(path,d):return write_bytes(path,canonical(d),replace=True)
def journal(directory,event):
 d=Path(directory);validate_journal(d);prior=sorted(d.glob('*.json'))
 seq=len(prior);prev=sha(prior[-1]) if prior else None
 row={'sequence':seq,'previous_sha256':prev,**event}
 p=d/f'{seq:06d}.json';create(p,row);return {'path':str(p),'sha256':sha(p)}
def validate_journal(directory):
 prev=None
 for seq,p in enumerate(sorted(Path(directory).glob('*.json'))):
  d=json.loads(p.read_text());assert 'sequence' in d and 'previous_sha256' in d;assert d['sequence']==seq and d['previous_sha256']==prev;prev=sha(p)
 return prev
