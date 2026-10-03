import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from centered_core.integrity import verify_sources
print(json.dumps({'verdict': 'PASS', 'source_identity': verify_sources()}))
