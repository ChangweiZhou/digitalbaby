"""Supplement GitHub durability with independently downloaded private Library backups.

Public coverage stays distinct. This adapter only relaxes the public-backlog
stop when exact receipt bytes are already verified in the user's private backup.
The frozen learner, roster, source lock and live-budget driver remain unchanged.
"""
import json
from pathlib import Path
import recovery_local as local

class PrivateRecoveryEnv(local.RecoveryEnv):
 def private(self):
  p=self.root/'results/recovery/PRIVATE_BACKUP_STATUS.json'
  if not p.exists():return {}
  d=json.loads(p.read_text())
  if d.get('schema')!='MINIFLY-A3-VERIFIED-PRIVATE-BACKUP-v1' or not d.get('verified_at_utc') or not d.get('library_file_id'):raise AssertionError('Unverified private backup')
  if d['lock_digest']!=json.loads((self.root/'SOURCE_LOCK.json').read_text())['lock_digest']:raise AssertionError('Private backup lock mismatch')
  for name,h in d['receipt_sha256'].items():
   p=self.root/name
   if not name.startswith('results/science/') or not p.is_file() or local.sha(p.read_bytes())!=h:raise AssertionError('Private-backed receipt bytes changed')
  return d['receipt_sha256']
 def persist(self,message):
  public_backlog=None
  try:super().persist(message)
  except RuntimeError as exc:
   if not str(exc).startswith('Durability backlog reached8'):raise
   public_backlog=exc
  github=self.backup()['receipt_sha256'];private=self.private();combined={**github,**private}
  for name,h in github.items():
   if name in private and private[name]!=h:raise AssertionError('Backup channels disagree about receipt bytes')
  pending={f'results/science/{a}/{w}.json.gz':h for (a,w),h in self.validated.items() if combined.get(f'results/science/{a}/{w}.json.gz')!=h}
  local.write_json(self.root/'results/recovery/OFF_HOST_STATUS.json',{'verified_github_receipts':len(github),'verified_private_recoverable_receipts':len(private),'verified_combined_receipts':len(combined),'unprotected_accepted_receipts':len(pending)})
  print(json.dumps({'github':len(github),'private_recoverable':len(private),'combined':len(combined),'unprotected':len(pending)}),flush=True)
  if len(pending)>=8:raise RuntimeError('Durability backlog reached8 receipts without either verified GitHub or private Library backup')
