"""Read-only terminal monitor. Reports partial work separately from accepted jobs."""
from pathlib import Path
import json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent

def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None

def snapshot():
    status=read(ROOT/'results/STATUS.json') or {'state':'NOT_LAUNCHED'}
    active=[]
    for pattern in ('track_a/ACTIVE.json','track_b/active/*.json'):
        for p in (ROOT/'results').glob(pattern):
            row=read(p)
            if row:active.append(row)
    return dict(status=status,partial_progress=active)

if __name__=='__main__':print(json.dumps(snapshot(),indent=2))
