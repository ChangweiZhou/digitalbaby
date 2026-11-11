# Three G0 lives interrupted by the container restart of 11 November 2026 (between 06:29 and 06:38 UTC), kept as written

Runner exp20_reference.py at 0e131a29, full dose, seeds 73129, 93683, 116089, started 05:27:47 UTC from the complete births
exp45/e20b/e20_dev_G0_<seed>.pt (hashes verified after the restart). Each receipt holds stages 0 to 4 (through F2.2) with full scores;
no completion record, no final weights. Stopped and reported (amendment 7).

The owner then asked for the optimizer state to be saved so that a restart cannot cost a run again, and authorized resuming the work
once that was tested. exp20_resume.py does this; exp45/e20b_resume_check/RESUME_CHECK.json shows that runs killed with SIGKILL and
resumed end identical to the qualified uninterrupted runs. These three lives were rerun from scratch, from the same births, with
exp20_resume.py (a life without a checkpoint cannot be resumed). The rerun's receipts are in exp45/e20b/ under the standard names.
