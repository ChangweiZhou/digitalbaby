"""Read-only compact status for the fixed V43 run."""
from pathlib import Path
import json
import sys
import time
from collections import Counter

root = Path(sys.argv[1] if len(sys.argv) > 1 else 'v43_final_fb8ebb_results')
files = list((root / 'progress').glob('*.json'))
rows = [json.loads(path.read_bytes()) for path in files]
done = list((root / 'completed').glob('*.json'))
saved = sum(row['cursor'] for row in rows)
print(json.dumps({
    'utc': time.strftime('%H:%M:%S', time.gmtime()),
    'completed': len(done), 'expected_cells': 64, 'started': len(rows),
    'completed_by_load': dict(sorted(Counter(int(f.stem.split('_')[-1]) for f in done).items())),
    'saved_training_percent': round(100 * saved / 131520, 2),
    'saved_records': saved, 'expected_records': 131520,
    'newest_checkpoint_age_seconds': round(time.time() - max((f.stat().st_mtime for f in files), default=time.time()), 1),
    'oldest_active_checkpoint_age_seconds': round(max((time.time() - f.stat().st_mtime for f, r in zip(files, rows) if not r['done']), default=0), 1),
    'max_recorded_worker_rss_MiB': round(max((r['peak_rss_bytes'] for r in rows), default=0) / 2**20, 1),
    'diagnostic_folds': len(list((root / 'diagnostic_checkpoints').glob('*.checkpoint'))),
}))
