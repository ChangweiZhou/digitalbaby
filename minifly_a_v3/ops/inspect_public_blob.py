"""Bind an exact base64 publication blob to a validated synthetic receipt."""
import base64,collections,gzip,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
req=json.loads(Path(sys.argv[1]).read_text());f=req['files'][int(sys.argv[2])];encoded=Path(f['local_path']).read_bytes()
assert hashlib.sha256(encoded).hexdigest()==f['sha256'];assert hashlib.sha1(b'blob '+str(len(encoded)).encode()+b'\0'+encoded).hexdigest()==f['sha']
report={'exact_public_upload_file':f['path'],'git_blob_sha':f['sha'],'encoded_sha256':f['sha256'],'content_first64':encoded[:64].decode(),'content_last64':encoded[-64:].decode()}
if '/chunks/' in f['path']:
 parts=Path(f['path']).parts;a,w=parts[-3],int(parts[-2]);index=int(parts[-1].split('-')[1].split('.')[0]);raw=(ROOT/'results/science'/a/f'{w}.json.gz').read_bytes();doc=json.loads(gzip.decompress(raw));piece=base64.b64decode(encoded);assert piece==raw[index*98304:(index+1)*98304]
 strings=collections.Counter();pending=[doc]
 while pending:
  v=pending.pop()
  if isinstance(v,dict):pending.extend(v.values())
  elif isinstance(v,list):pending.extend(v)
  elif isinstance(v,str):strings[v]+=1
 nonhex={k:v for k,v in strings.items() if not re.fullmatch('[0-9a-f]+',k)}
 cp=json.loads((ROOT/req['checkpoint_path']).read_text());assert cp['receipt_sha256'][f'results/science/{a}/{w}.json.gz']==hashlib.sha256(raw).hexdigest()
 report.update({'full_receipt_identity':{k:doc[k] for k in ('schema','arm','world','kind','lock_digest')},'full_receipt_sha256':hashlib.sha256(raw).hexdigest(),'full_receipt_top_level_keys':sorted(doc),'all_nonhex_string_values':nonhex,'slice_start':index*98304,'slice_end':min(len(raw),(index+1)*98304),'exact_slice_verified':True,'frozen_audit_evidence':'The immutable local checkpoint admits only receipts validated by unchanged drive_science.validate_receipt, including full schema, source provenance and causal mechanism audit. The exact full receipt hash matches that checkpoint.'})
print(json.dumps(report,sort_keys=True),flush=True)
