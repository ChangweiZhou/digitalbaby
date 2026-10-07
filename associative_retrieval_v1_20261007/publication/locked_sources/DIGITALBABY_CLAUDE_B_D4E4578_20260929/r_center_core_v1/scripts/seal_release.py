"""Developer-only sealing; does not execute a learner or scientific worlds."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from centered_core.integrity import source_map, source_identity

doc = {'schema': 'R_CENTER_ENGINEERING_SOURCE_LOCK_V1', 'source_identity': source_identity(),
       'files': source_map(), 'provenance': 'new engineering release of the public R3 outcome package; not its original scientific LOCK.json'}
(ROOT / 'SOURCE_LOCK.json').write_text(json.dumps(doc, indent=2, sort_keys=True) + '\n')
print(json.dumps({'source_identity': doc['source_identity'], 'files': len(doc['files'])}))
