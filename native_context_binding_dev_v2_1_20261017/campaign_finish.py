"""Session accounting only; DEV selection remains locked until all eight worlds."""
import gzip,json,zipfile,statistics
import campaign_common as c

def finish():
    c.status('PACKAGING_DEV_A',stage='DEV_A',active_jobs=[])
    index=c.update_index();m=c.manifest('DEV_A')
    if (index['full_jobs'],index['records'])!=(36,73728):raise AssertionError('Incomplete DEV_A')
    rows=[]
    for j in index['jobs']:
        r=c.read(j['receipt']);w=r['histories']['W'];n=r['histories']['N_OLD']
        rows.append(dict(world=r['world'],arm=r['arm'],W_HELD_final={k:v['accuracy'] for k,v in w['points'][-1]['readings'].items()},
            N_OLD_HELD_final={k:v['accuracy'] for k,v in n['points'][-1]['readings'].items()},
            worker_CPU=r['actual_worker_CPU_seconds'],wall_seconds=r['elapsed_seconds']))
    summary=dict(status='DEV_A_COMPLETE_AWAITING_DEV_B_AUTHORIZATION',selection_performed=False,
        reason='Only four of eight locked DEV worlds; no scientific adoption/selection verdict yet',
        full_jobs=36,continuous_histories=72,external_records=73728,rows=rows)
    c.write(c.RESULTS/'SUMMARY.json',summary)
    text='# NATIVE-BIND V2.1 · DEV_A\n\n36 certified full jobs, 72 continuous histories, 73,728 external record applications.\n\nDEV_A is complete. DEV_B requires explicit continuation. No candidate selection, common CPU budget lock, confirmation or adoption decision has been made: those require all eight DEV worlds.\n\nOwn-byte+EOS HELD scores are preserved in the immutable receipts. Taught-item acquisition and retention remain separate from unseen-addition generalization.\n'
    (c.RESULTS/'REPORT.md').write_text(text)
    c.write(c.RESULTS/'FINAL_AUDIT.json',dict(accepted=True,scope='Completed DEV_A session accounting; not full DEV efficacy',
        committed_jobs=36,histories=72,records=73728,receipt_hashes_checked=True,all_job_audits_accepted=True,
        selection_performed=False,kernel_unchanged=True))
    bundle=c.RESULTS/'DEV_A_RESULT_BUNDLE.zip'
    with zipfile.ZipFile(bundle.with_suffix('.zip.tmp'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(c.RESULTS.rglob('*')):
            if p.is_file() and p.name not in ('DEV_A_RESULT_BUNDLE.zip','DEV_A_RESULT_BUNDLE.zip.tmp') and p.suffix not in ('.pt','.tmp'):
                z.write(p,str(p.relative_to(c.HERE)))
        for p in (c.HERE/'CAMPAIGN_SOURCE_LOCK.json',c.HERE/'CAMPAIGN_QUALIFICATION.json',c.HERE/'SOURCE_LOCK.json',c.OPS/'AUTHORIZATION.json'):
            z.write(p,str(p.relative_to(c.HERE)))
    bundle.with_suffix('.zip.tmp').replace(bundle)
    c.write(c.RESULTS/'PACKAGE.json',dict(completed=True,scope='DEV_A',bundle=str(bundle),sha256=c.sha(bundle),binary_checkpoints_retained_locally=True))
