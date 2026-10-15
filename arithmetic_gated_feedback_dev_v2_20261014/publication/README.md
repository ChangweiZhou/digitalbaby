# GATED-FEEDBACK V2 publication

Start with ../REPORT.md and ../FINAL_AUDIT.json. Three technical cycles completed:
6 world/model jobs, 450,560 external record applications, 42 correlated terminal
histories and 96 native state archives. Technical qualification passed; the new
candidate's effect was insufficient. This is not formal DEV/CONFIRM or adoption.

Source, design, reports, original audit certificates, logs and operational
records are directly browsable in Git. The complete 145 MB result archive is
split into files below the hosting limit. It contains all six full native
receipts, all 96 native state archives, two generic-fixture archives and the
environment dependency. Binary copies are not duplicated in Git at their local
paths; ARCHIVE_COVERAGE.json maps every such path to its archive member. No
original local data was removed or altered for publication.

From the experiment folder, in a fresh checkout where RESULT_BUNDLE.zip is absent:

```sh
python3 publication/reassemble_bundle.py --output RESULT_BUNDLE.zip
unzip -n RESULT_BUNDLE.zip -d ..
```

Reassembly only copies the ordered chunks and checks the archive. It does not
run models, audits or experiments. Do not rerun the technical trajectories to
read these results. The PACKAGE.json manifest is external to the archive to
avoid a circular self-hash. PUBLICATION.json is the current upload record;
older authorization fields inside frozen design/source artifacts remain
historical. The user's subsequent upload request is recorded separately.
