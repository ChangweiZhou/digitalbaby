"""Recover the reviewed source and scoring patch from the Round 8 session."""
import json
from pathlib import Path
import sys

events=json.loads(Path(sys.argv[1]).read_text())
def command(n):
    return events[n]['input']['command']

def fragment(n):
    return command(n).split('\nEOF',1)[0].split('\n',1)[1]

Path('pred.py').write_text(fragment(1898))
exec(compile(fragment(1903),'predictor-build-fragment','exec'))
pred=Path('pred2.py').read_text().replace("round(run('bag'),3)","round(float(run('bag')),3)")
Path('pred2.py').write_text(pred)
Path('pred3.py').write_text(pred.replace('orders=(1,2,3,4,5)','orders=(1,2,3,4,5,6,7)'))
Path('hop.py').write_text(fragment(1917))
Path('archived_hops.jsonl').write_text(events[1938]['content'][0]['text'])
