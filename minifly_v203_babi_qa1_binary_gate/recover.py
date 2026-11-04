"""Recover the reviewed Round 11 source changes from the session export."""
from pathlib import Path
import json
import os
import sys
import tempfile

root=Path(__file__).resolve().parent
events=json.loads(Path(sys.argv[1]).read_text())
def fragment(i):
    return events[i]['input']['command'].split('\nEOF',1)[0].split('\n',1)[1]

with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    Path('babi13.py').write_bytes((root.parent/'minifly-v202'/'babi13.py').read_bytes())
    exec(compile(fragment(2216),'reviewed-event-2216','exec'))
    p=Path('babi14.py')
    p.write_text(p.read_text().replace('(None if FROZ[0] else TAGS.append)(',
                                     '(TAGS.append if not FROZ[0] else (lambda z: None))('))
    for i in (2225,2240):
        exec(compile(fragment(i),f'reviewed-event-{i}','exec'))
    for n in (14,15,16):
        p=Path(f'babi{n}.py')
        (root/p.name).write_bytes(p.read_bytes())
