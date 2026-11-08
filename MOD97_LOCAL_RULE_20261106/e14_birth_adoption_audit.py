"""Clarify scheduler-handover metadata; scientific checkpoint/curve files are untouched."""
import shutil
import e14_campaign as C

def main():
 target=C.R/'E14_BIRTH_ADOPTION_AUDIT.json'
 if target.exists():raise RuntimeError('Already audited')
 archive=C.O/'e14_birth_job_metadata_before_exit_code_clarification';archive.mkdir()
 results=[]
 for j in C.plan()[0]:
  p=C.O/f'E14_JOB_{j["key"]}.json';row=C.read(p);assert row.get('adopted_after_worker_exit')
  C.validate(j);shutil.copy2(p,archive/p.name)
  # The scheduler was suspended, so its wait() did not capture the exit code.
  # Completed, finite checkpoints and the full development curves were verified after exit.
  row.update(returncode=None,exit_code_observed=False,process_exit_observed=True,
             completion_basis='Process observed exited/zombie, finite checkpoint and complete24576-update metadata validated; numeric exit code unavailable',
             initial_returncode_zero_was_inferred=True)
  C.save(p,row)
  results.append(dict(key=j['key'],checkpoint_sha256=C.sha(C.O/j['receipt']),metadata_sha256=C.sha(C.O/j['metadata']),returncode=None,completion_basis=row['completion_basis']))
 C.save(target,dict(reason='Handover initially inferred returncode0 from completed artifacts after process exit. That code was not directly captured; canonical JOB files now explicitly mark it unknown. Initial JOB metadata preserved.',jobs=results,original_checkpoint_and_curve_files_changed=False,live_status_caveat='Running controller cached the initial rows. Finalization must mirror canonical birth JOB rows into E14_STATUS before publishing; do not report46observed zero exit codes.',birth_wall_times='Artifact timestamp lower bounds, as already labeled.'))
 print('Birth artifacts verified; unobserved exit codes honestly marked null')
if __name__=='__main__':main()
