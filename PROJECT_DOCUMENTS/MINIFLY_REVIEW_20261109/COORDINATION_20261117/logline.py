"""Append one row to LOG.md or PLAN.md (in this folder) with the time read from the clock.
Usage, from anywhere:  python3 -I logline.py <LOG|PLAN> "<cells separated by ' | '>"
The row is written as "| HH:MM | <text> |"; the text must not end with a cell separator."""
import datetime
import sys
from pathlib import Path

which, text = sys.argv[1], sys.argv[2].rstrip().rstrip("|").rstrip()
assert which in ("LOG", "PLAN")
now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M")
p = Path(__file__).resolve().parent / ("LOG.md" if which == "LOG" else "PLAN.md")
p.write_text(p.read_text().rstrip("\n") + f"\n| {now} | {text} |\n")
print(now, "appended to", which)
