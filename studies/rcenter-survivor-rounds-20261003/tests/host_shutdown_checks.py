# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise the actual keeper's token-bound graceful stop in an isolated root."""
import json,subprocess,sys,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run_checks(temp_root):
 root=Path(temp_root)/'host-shutdown';(root/'operations').mkdir(parents=True)
 (root/'operations/RUN_LEDGER.json').write_text('{"active_wall_s":1.0}')
 code="import sys,importlib.util;from pathlib import Path;s=importlib.util.spec_from_file_location('keeper',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.ROOT=Path(sys.argv[2]);m.hold('fresh-test-host')"
 p=subprocess.Popen([sys.executable,'-c',code,str(ROOT/'operations/host_session.py'),str(root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:
  deadline=time.monotonic()+5
  while not (root/'operations/HOST_WALL.json').exists():
   assert p.poll() is None and time.monotonic()<deadline;time.sleep(.02)
  (root/'operations/HOST_STOP.json').write_text('{"host_token":"wrong-host"}');time.sleep(1.1);assert p.poll() is None
  (root/'operations/HOST_STOP.json').write_text('{"host_token":"fresh-test-host"}')
  out,err=p.communicate(timeout=5);assert p.returncode==0,(out,err)
  state=json.loads((root/'operations/HOST_WALL.json').read_text());assert state['active'] is False and state['effective_s']>2.
  rows=[json.loads(q.read_text()) for q in sorted((root/'operations/host_events').glob('*.json'))]
  assert [r['event'] for r in rows]==['host_started','host_closed']
  with (root/'operations/HOST_COORDINATOR.lock').open() as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  return ['actual_keeper_wrong_token_ignored','actual_keeper_graceful_close_recorded_and_lock_released']
 finally:
  if p.poll() is None:p.kill();p.wait()
