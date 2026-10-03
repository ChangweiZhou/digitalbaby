# SPDX-License-Identifier: GPL-3.0-or-later
"""Buffer-free OS-lock keeper for one authenticated tool-host session.

Stop only this exact tool execution session; numeric PIDs are not cross-tool
identities. A new keeper reconciles a stale wall record conservatively after
obtaining the old lock. It does not execute science or archive payloads.
"""
import argparse,fcntl,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source/runtime'))
from durable import replace,journal,validate_journal
CAP=18*3600.
def hold(token):
 p=ROOT/'operations/HOST_WALL.json';events=ROOT/'operations/host_events'
 with (ROOT/'operations/HOST_COORDINATOR.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);validate_journal(events)
  old=json.loads(p.read_text()) if p.exists() else None
  legacy=json.loads((ROOT/'operations/RUN_LEDGER.json').read_text()).get('active_wall_s',0.) if (ROOT/'operations/RUN_LEDGER.json').exists() else 0.
  base=float(old.get('cumulative_s',legacy)) if old else float(legacy)
  if old and old.get('active'):
   upper=max(float(old.get('effective_s',base)),base+max(0.,time.time()-float(old['started_unix_s'])))
   journal(events,{'event':'recovered_unknown_session','old_host_token':old['host_token'],'charged_active_wall_s':upper-base});base=upper
  assert base<CAP,'active wall budget exhausted'
  start=time.monotonic();utc=time.time();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
  state={'schema':'RC-HOST-WALL-v1','active':True,'host_token':token,'boot_id':boot,'started_unix_s':utc,'cumulative_s':base,'effective_s':base,'updated_monotonic_s':start}
  journal(events,{'event':'host_started','host_token':token,'cumulative_s':base,'started_unix_s':utc})
  replace(p,state);print(json.dumps({'ready':True,'host_token':token}),flush=True)
  try:
   while True:
    stop_path=ROOT/'operations/HOST_STOP.json'
    if stop_path.exists() and json.loads(stop_path.read_text()).get('host_token')==token:break
    state.update(effective_s=base+time.monotonic()-start,updated_monotonic_s=time.monotonic());replace(p,state)
    assert state['effective_s']<=CAP,'active wall cap reached';time.sleep(1)
  finally:
   total=base+time.monotonic()-start;state.update(active=False,cumulative_s=total,effective_s=total,updated_monotonic_s=time.monotonic());replace(p,state)
   journal(events,{'event':'host_closed','host_token':token,'cumulative_s':total,'charged_active_wall_s':total-base});print(json.dumps({'closed':True,'effective_s':total}),flush=True)
if __name__=='__main__':
 def stop(signum,frame):raise KeyboardInterrupt()
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 p=argparse.ArgumentParser();p.add_argument('token')
 try:hold(p.parse_args().token)
 except KeyboardInterrupt:pass
