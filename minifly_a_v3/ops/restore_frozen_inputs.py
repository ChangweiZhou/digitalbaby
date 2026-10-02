"""Restore already-published frozen inputs after checking the exact public archive.

Run from the repository root: python minifly_a_v3/ops/restore_frozen_inputs.py
No network access, package installation, or simulation is performed.
Existing equal files are retained; any unequal existing file stops restoration.
"""
from pathlib import Path
import hashlib,json,os,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def main():
 manifest=json.loads((ROOT/'results/PUBLIC_FINAL_MANIFEST.json').read_text())
 ref=manifest['archived_locked_inputs'];lock=json.loads((ROOT/'SOURCE_LOCK.json').read_text())
 archive=ROOT.parent/ref['archive_path'];raw=archive.read_bytes()
 assert sha(raw)==ref['archive_sha256'],'archive SHA256 mismatch'
 assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==ref['archive_git_blob_sha'],'archive Git blob mismatch'
 with zipfile.ZipFile(archive) as z:
  verified=[]
  for dest,item in sorted(ref['members'].items()):
   relative=Path(dest)
   assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0]=='package'
   member=Path(item['archive_member']);assert not member.is_absolute() and '..' not in member.parts
   b=z.read(item['archive_member']);assert len(b)==item['bytes'] and sha(b)==item['sha256']==lock['files'][dest]
   p=ROOT/relative
   if p.exists():assert p.is_file() and p.read_bytes()==b,'existing input differs: '+dest
   verified.append((p,b))
  assert len(verified)==108
  for p,b in verified:
   if p.exists():continue
   p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 for n,h in lock['files'].items():assert sha((ROOT/n).read_bytes())==h,'source lock mismatch: '+n
 print(json.dumps({'restored_or_retained_inputs':108,'verified_locked_files':len(lock['files']),'source_lock_digest':lock['lock_digest'],'simulation_run':False}))
if __name__=='__main__':main()
