# SPDX-License-Identifier: GPL-3.0-or-later
"""Supervised final-analysis bootstrap; modifies only its persistence gate."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/private_official_policy'))
import adapter


def execute():
    adapter.check(os.environ.get('SURVIVOR_MODE') == 'science' and os.environ.get('SURVIVOR_JOB_KEY') == 'analysis/final', 'analysis-only supervised policy bootstrap')
    policy = adapter.verify_policy(ROOT)
    adapter.prelaunch_guard(ROOT, policy)
    # Unmodified require_science inside the original entry still checks PID,
    # reservation/key/host, runtime, source, OFFICIAL_LOCK and LAUNCH_ACCEPTED.
    with adapter.analysis_gate(ROOT):
        runpy.run_path(str(ROOT / 'source/runtime/analysis_entry.py'), run_name='__main__')


if __name__ == '__main__':
    execute()
