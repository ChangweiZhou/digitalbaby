# SPDX-License-Identifier: GPL-3.0-or-later
"""Owner-authorized private transport; publications are independent of execution."""
import argparse,json,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'operations'));sys.path.insert(0,str(ROOT/'source/runtime'))
import transport as legacy
import checkpoint as cp
from durable import replace,journal,validate_journal,sha
sys.path.insert(0,str(ROOT/'source/private_official_policy'))
STATE=ROOT/'operations/private_official_policy'
PENDING=STATE/'TRANSPORT_PENDING.json'
def setup_checkpoint():
 original=cp.state_files
 def selected():
  extra=[p for folder in (ROOT/'operations/private_policy',STATE,ROOT/'operations/retention_policy') for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.tmp' not in p.suffixes]
  for p in extra:
   if p.name=='POLICY_ACCEPTED.json' or 'attestations' in p.parts:cp.SEMANTIC_NAMES.add(p.name)
  return sorted(set(original()+extra))
 cp.state_files=selected

def main(action,d):
 import adapter
 import recovery
 adapter.verify_policy()
 if action not in ('pending','info'):legacy.assert_host()
 if action=='pending':return json.loads(PENDING.read_text()) if PENDING.exists() else None
 if action=='info':return {'private_pending':main('pending',{}),'checkpoint':json.loads((ROOT/'operations/CHECKPOINT_STATE.json').read_text()),'host':json.loads((ROOT/'operations/HOST_WALL.json').read_text())}
 if action=='completed':
  from integrity import hashes,require_runtime
  import recovery
  with recovery.exclusive_lock(ROOT) as lock:
   ledger=recovery.load_ledger(ROOT,lock=lock);recovery.validate_attempts(ROOT,ledger,hashes(),require_runtime())
   return [a['key'] for a in ledger['attempts'] if a['status']=='completed' and a['key'] in adapter.KEYS]
 if action=='assert_host':return legacy.assert_host()
 if action=='stop_host':return legacy.main('stop_host',{})
 if action=='phase':
  with recovery.exclusive_lock(ROOT):
   validate_journal(STATE/'transport_journal');replace(PENDING,None if d.get('status')=='complete' else d);journal(STATE/'transport_journal',d);return {'saved':True}
 if action=='checkpoint':
  with recovery.exclusive_lock(ROOT):
   setup_checkpoint();return legacy.checkpoint()
 if action=='preflight_io':return legacy.main(action,d)
 if action=='private_ack':
  with recovery.exclusive_lock(ROOT):return legacy.private_ack(d)
 if action=='reservation_ack':
  with recovery.exclusive_lock(ROOT):return legacy.main(action,d)
 if action=='barrier':return adapter.ack_private(ROOT,d['key'])
 if action=='prelaunch':return adapter.mark_prelaunch(ROOT)
 if action=='retention':
  sys.path.insert(0,str(ROOT/'source/retention_policy'))
  import executor
  return executor.execute_current(ROOT, checkpoint=json.loads((ROOT/'operations/CHECKPOINT_STATE.json').read_text()), acceptance_path=ROOT/'operations/retention_policy/ACCEPTED.json', launch_manifest=ROOT/'source/retention_policy/LAUNCH_PROTECTION.json', launch_sha256='af6d9df47b6d7e67bd5bc9021b688141c8381eff1eeab114380fdfdfd5fb5ccc')
 if action=='restore':
  # Original pure restore validates the frozen source/receipts. The policy planner is separately replayed by adapter checks.
  with recovery.exclusive_lock(ROOT):return legacy.restore_verified(d)
 raise ValueError(action)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');a=p.parse_args();raw=sys.stdin.read();print(json.dumps(main(a.action,json.loads(raw) if raw.strip() else {}),separators=(',',':')))
