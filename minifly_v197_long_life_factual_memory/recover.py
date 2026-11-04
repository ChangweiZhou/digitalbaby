"""Recover only the reviewed Python build fragments from the Round 7 export."""
from pathlib import Path
import json
import sys

events = json.loads(Path(sys.argv[1]).read_text())
def command(index):
    return events[index]['input']['command']

def fragment(index):
    return command(index).split("\nEOF", 1)[0].split("\n", 1)[1]

Path('life.py').write_text(fragment(1619))
for event in (1680, 1709, 1787, 1828, 1852):
    exec(compile(fragment(event), f'round7-event-{event}', 'exec'))
source = Path('life_long3.py').read_text()
Path('round7_recovered.py').write_text(source)
for name in ('life.py', 'life_m.py', 'life_long.py', 'life_long2.py', 'life_long3.py'):
    Path(name).unlink()
