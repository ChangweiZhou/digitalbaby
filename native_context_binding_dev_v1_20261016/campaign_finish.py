"""Finalize already certified science, never train or upload."""
import zipfile
import hashlib
import campaign_common as c


def finish(verdict):
    index=c.update_index()
    expected_jobs=24 if verdict=="NO_ADVANCE" else 88
    expected_records=49152 if verdict=="NO_ADVANCE" else 311296
    if index["full_jobs"]!=expected_jobs or index["records"]!=expected_records:
        raise AssertionError("Wrong completion roster/count")
    c.verify_lock()
    report=c.RESULTS/("DEV_REPORT.md" if verdict=="NO_ADVANCE" else "CONFIRM_REPORT.md")
    text=("# NATIVE-BIND V1 正式实验\n\n"+f"结论：**{verdict}**。已提交{index['full_jobs']}个完整世界×配置jobs、{index['histories']}条相关持续历史、{index['records']:,}条外部记录应用。技术周期的256条记录不计入科学样本。\n\n"
          +report.read_text()+"\n\n目标保持为单位样本和实际计算下的未见加法题完整自主字节输出，以及OLD保持。保留Full151/R_center真实原生记忆，没有算术规则、注意力、Transformer、ELM或独立答案头。\n\n本结果不自动采用新核心。NO_ADVANCE或未确认优势均为有效完成；不延长样本、不替换候选、不在本次运行新增设计。\n")
    (c.RESULTS/"REPORT.md").write_text(text)
    (c.HERE/"REPORT.md").write_text(text)
    summary=dict(schema="NATIVE_BIND_SCIENTIFIC_SUMMARY_V1",verdict=verdict,
                 committed_jobs=index["full_jobs"],continuous_histories=index["histories"],
                 external_record_applications=index["records"],technical_records_excluded=256,
                 automatic_core_adoption=False,report=str(c.RESULTS/"REPORT.md"),github_uploaded=False)
    c.write(c.RESULTS/"SUMMARY.json",summary)
    audit=dict(schema="NATIVE_BIND_CAMPAIGN_FINAL_AUDIT_V1",accepted=True,
               expected_jobs=expected_jobs,committed_jobs=index["full_jobs"],
               expected_records=expected_records,committed_records=index["records"],
               immutable_job_audit_certificates_verified=True,source_lock_verified=True,
               science_training_rerun_for_reporting=False,extra_training_records=0,
               conclusion_scope="E3 only if all locked conjunctive gates pass; E1 scores reported separately",
               verdict=verdict,automatic_core_adoption=False)
    c.write(c.RESULTS/"FINAL_AUDIT.json",audit)
    c.status("COMPLETED_ACCEPTED",verdict=verdict,report=str(c.RESULTS/"REPORT.md"),packaging="IN_PROGRESS")
    bundle=c.RESULTS/"RESULT_BUNDLE.zip"
    lock=c.read(c.HERE/"SOURCE_LOCK.json")
    files=set(c.ROOT/name for name in lock["operative_sources"])
    files.update([c.HERE/"SOURCE_LOCK.json",c.HERE/"CAMPAIGN_QUALIFICATION.json"])
    for folder in (c.HERE,c.DESIGN.parent):
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix in (".zip",".pyc",".tmp"):
                continue
            if any(part in ("scratch","__pycache__",".venv") for part in path.parts):continue
            if path.name=="PACKAGE.json":continue
            files.add(path)
    files.discard(c.RESULTS/"BUNDLE_INVENTORY.json")
    inventory={}
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(files):
            name=str(path.relative_to(c.ROOT));digest=hashlib.sha256()
            # Hash exactly the bytes placed in the archive. Operational heartbeats
            # can continue updating; their archived snapshot stays self-consistent.
            with path.open("rb") as source,archive.open(name,"w",force_zip64=True) as target:
                while block:=source.read(1024*1024):target.write(block);digest.update(block)
            inventory[name]=digest.hexdigest()
        inventory_text=__import__("json").dumps(dict(files=inventory),indent=2)+"\n"
        archive.writestr(str((c.RESULTS/"BUNDLE_INVENTORY.json").relative_to(c.ROOT)),inventory_text)
        (c.RESULTS/"BUNDLE_INVENTORY.json").write_text(inventory_text)
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:raise AssertionError("Result bundle CRC failed")
    c.write(c.RESULTS/"PACKAGE.json",dict(complete=True,sha256=c.sha(bundle),bytes=bundle.stat().st_size,
             path=str(bundle),files=len(files)+1,science_jobs=index["full_jobs"],github_uploaded=False))
    c.status("COMPLETED_ACCEPTED",verdict=verdict,packaging="COMPLETE",
             report=str(c.RESULTS/"REPORT.md"),bundle=str(bundle))
    return summary
