"""Asserts that section 6 of PLAN.md assigns every entry of the two inventories (D1 to D57, S1 to S50) exactly once.
Run from anywhere:  python3 -I plan_check.py"""
import re, collections, pathlib
text = (pathlib.Path(__file__).resolve().parent / "PLAN.md").read_text()
sec = text.split("## 6. Where each entry of the two inventories goes")[1].split("## 7. State")[0]
rows = [l for l in sec.splitlines() if l.startswith("|") and not l.startswith("|---") and not l.startswith("| Destination")]
count = collections.Counter()
for l in rows:
    entries = l.rstrip("|").rsplit("|", 1)[1]                       # the last cell holds the entries
    for tok in re.findall(r"\b([DS]\d+)\b", entries):
        count[tok] += 1
want = [f"D{i}" for i in range(1, 58)] + [f"S{i}" for i in range(1, 51)]
missing = [w for w in want if count[w] == 0]; twice = [w for w in want if count[w] > 1]; extra = [k for k in count if k not in want]
print(f"rows {len(rows)}; entries assigned {sum(count.values())}; missing {missing}; assigned more than once {twice}; unknown {extra}")
assert not missing and not twice and not extra and sum(count.values()) == 107
print("every entry of the two inventories is assigned exactly once")
