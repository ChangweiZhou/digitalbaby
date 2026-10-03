# SPDX-License-Identifier: GPL-3.0-or-later
"""One supervised pure terminal analysis after all64 worlds and acknowledgments."""
import os
import resource
import time
from pathlib import Path

from durable import create, sha, fsync_dir
from integrity import ROOT, require_science, require_runtime, hashes, digest_map, read, require
from analyze import analyze
from validate_analysis import inputs


def execute():
    start = time.monotonic()
    acceptance = require_science()
    source = hashes()
    runtime = require_runtime()
    active = read(ROOT / 'operations/ACTIVE_JOB.json')
    require(active['key'] == os.environ['SURVIVOR_JOB_KEY'] == 'analysis/final', 'terminal analysis supervision required')
    dest = Path(os.environ['SURVIVOR_DEST'])
    require(dest.resolve() == (ROOT / 'results/final').resolve() and not dest.exists(), 'fresh canonical analysis destination required')
    cohort = inputs(ROOT, source, runtime)
    require(cohort['lock_sha256'] == acceptance, 'analysis lock changed since admission')
    result = analyze(ROOT / 'receipts/science', source, dest / 'analysis.json')
    require(result['input_manifest'] == cohort['input_manifest'], 'analysis inputs changed during execution')
    require(hashes() == source, 'analysis source changed during execution')
    part = dest / 'analysis.json'
    manifest = {'schema': 'RC-SURVIVOR-ANALYSIS-RECEIPT-v1', 'complete': True, 'kind': 'analysis',
                'key': 'analysis/final', 'attempt': active['attempt'], 'reservation_id': active['reservation_id'],
                'native_teaching_events': 0, 'source': source, 'source_digest': digest_map(source),
                'runtime': runtime, 'lock_sha256': acceptance, 'input_manifest': cohort['input_manifest'],
                'parts': {'analysis.json': {'sha256': sha(part), 'bytes': part.stat().st_size}},
                'resources': {'wall_s': time.monotonic() - start,
                              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}}
    create(dest / 'manifest.json', manifest)
    fsync_dir(dest)
    return manifest


if __name__ == '__main__':
    import json
    print(json.dumps(execute(), separators=(',', ':')), flush=True)
