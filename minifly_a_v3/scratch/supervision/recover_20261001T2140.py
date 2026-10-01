from pathlib import Path
import datetime,fcntl,hashlib,json,sys,time,uuid
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'ops'))
import recovery_local as local
handle=(root/'scratch/supervision/supervisor.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=local.RecoveryEnv();driver=local.RecoveryDriver(env)
ledger_path=env.sci/'RUN_LEDGER.json'; original=ledger_path.read_bytes();old=json.loads(original)
assert old['active_wall_s']==106651.41066382588 and old['worker_s']==425234.11058428144,'Unexpected ledger: do not repeat recovery adjustment'
recovery=root/'results/recovery/host_loss_20261001';recovery.mkdir()
with (recovery/'REMOTE_LEDGER.before.json').open('xb') as f:f.write(original)
print(json.dumps({'bootstrap_started':True}),flush=True)
done=driver.bootstrap();assert len(done)==690
now=time.time();start=datetime.datetime(2026,10,1,21,14,tzinfo=datetime.timezone.utc).timestamp();gap=max(0,now-start)
record={'kind':'user_authorized_host_loss_reconstruction','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'User2026-10-01T21:33:56Z requested regular GitHub uploads and increased compute quota, replying to explicit request to regenerate67 accepted-lost trajectories with same lock/seeds and original compute counted.','old_remote_ledger_sha256':hashlib.sha256(original).hexdigest(),'last_observed_active_wall_s':121717.77187667531,'last_observed_worker_s':483839.86000360106,'reservation_since_utc':'2026-10-01T21:14:00Z','additional_wall_reservation_s':gap,'additional_worker_reservation_s':4*gap,'last_observed_validated':757,'recovered_original_receipts':690,'previously_accepted_receipts_to_reconstruct':67,'missing_total_to_run':142,'source_lock_unchanged':True,'lost_clearance_history_note':'Remote history retained. Later lost ledger had no failure at21:15. Known earlier connector ACK timeout6e9e2771e7ce46ff8a992d97a6293998 was cleared under prior user-authorized local-first continuation; exact lost ledger bytes are unavailable. Never present this observation as recovered original ledger.'}
driver.ledger['active_wall_s']=max(old['active_wall_s'],121717.77187667531)+gap
driver.ledger['worker_s']=max(old['worker_s'],483839.86000360106)+4*gap
driver.ledger.setdefault('recovery_adjustments',[]).append(record)
driver.save();local.write_json(recovery/'RECOVERY_ACCOUNTING.json',record,True)
print(json.dumps({'recovered':len(done),'accounting':record}),flush=True)
print(json.dumps(driver.run()),flush=True)
