"""Complete DEV analysis and immutable result package. No trajectories."""
import zipfile
import campaign_common as c

def finish():
    import campaign_analysis
    c.verify_lock();index=c.update_index()
    if (index['full_jobs'],index['histories'],index['records'])!=(72,144,147456):raise AssertionError('Incomplete fixed DEV')
    selection,worlds,budget=campaign_analysis.analyze()
    verdict=selection['verdict'];state='COMPLETED_NO_ADVANCE' if verdict=='NO_ADVANCE' else 'DEV_COMPLETE_AWAITING_CONFIRM_AUTHORIZATION'
    c.write(c.RESULTS/'SUMMARY.json',dict(status=state,verdict=verdict,selected=selection['selected'],
        full_jobs=72,continuous_histories=144,external_records=147456,C_dev=budget['C_dev'],C_star=budget['C_star'],
        selection_performed=True,confirmation_started=False,automatic_adoption=False))
    c.write(c.RESULTS/'FINAL_AUDIT.json',dict(accepted=True,scope='Complete eight-world DEV accounting and locked screening',
        certified_full_jobs=72,continuous_histories=144,records=147456,all_job_audits_accepted=True,
        receipt_hashes_checked=True,unchanged_scientific_kernel=True,DEV_A_preserved=True,
        budget_locked_before_behavior_reduction=True,selection_confirmatory=False,confirmation_started=False))
    c.status('PACKAGING',stage='DEV_B',stage_committed_jobs=36,stage_total_jobs=36,pending_jobs=0,active_jobs=[])
    bundle=c.RESULTS/'RESULT_BUNDLE.zip';temp=bundle.with_suffix('.zip.tmp')
    if bundle.exists():raise PermissionError('Never overwrite a completed result bundle')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(c.RESULTS.rglob('*')):
            if p.is_file() and p.suffix not in ('.zip','.pt','.tmp'):
                z.write(p,str(p.relative_to(c.HERE)))
        for p in sorted(list(c.HERE.glob('*.py'))+list(c.OPS.glob('*.py'))+[c.HERE/'SPEC.json',c.HERE/'SOURCE_LOCK.json',c.HERE/'CAMPAIGN_SOURCE_LOCK.json',c.HERE/'OUTPUT_CODEBOOK.json',c.LOCK,c.OPS/'AUTHORIZATION.json',c.OPS/'CAMPAIGN_QUALIFICATION.json']):
            z.write(p,str(p.relative_to(c.HERE)))
    temp.replace(bundle)
    c.write(c.RESULTS/'PACKAGE.json',dict(completed=True,scope='Full eight-world DEV',bundle=str(bundle),sha256=c.sha(bundle),
        binary_checkpoints_retained_locally=True,DEV_A_bundle_preserved=True))
    c.status(state,stage='DEV_B',stage_committed_jobs=36,stage_total_jobs=36,pending_jobs=0,active_jobs=[],workers=0,
        verdict=verdict,selected=selection['selected'],next_stage='CONFIRM_REQUIRES_AUTHORIZATION' if selection['selected'] else None)
if __name__=='__main__':finish()
