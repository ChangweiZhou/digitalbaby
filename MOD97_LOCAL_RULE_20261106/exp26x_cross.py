"""E26-X: crossing birth weights and later update (the owner's request of 12 November 2026, the memo
MINIFLY_E26_RESULT_AND_NEXT_STEPS_20261112.md). A new file; exp26_candidate.py and every harness source stay as at 360e1b8d.

A life here is exp26_candidate.life, unchanged, with one difference: the birth it starts from and the update it uses are named
separately. `life <birth_arm> <update_arm> <seed>` loads the admitted E26 birth e26_dev_<birth_arm>_<seed>.pt (admitted by
exp26_candidate.admit_e26 for its own arm) and runs the life with the update of <update_arm> (S: local=True, the sweep; B: local=False,
backprop). Nothing else is touched: the same schedule, data streams, panels, optimizer (fresh AdamW, decay 0.25), scorer.

This is done by rebinding three names in exp26_candidate for the duration of the call (birth_path, admit_e26, life_path) and adding the
crossing to the provenance record. With birth_arm == update_arm the life is the E26 life itself (checked at reduced dose by `pilot`).

Output: exp45/e26x/e26x_life_<birth_arm><update_arm><DEVTAG><DRYTAG>_<seed>.json (+ .pt). The receipt's `arm`/`local` are the update's;
stages[0].settings.birth / birth_sha256 name the birth; stages[0].settings.provenance.crossover records both.
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import exp26_candidate as C          # noqa: E402  (imports exp20_reference, which applies the environment gate)
import exp11_cell as E               # noqa: E402

OUT = HERE / "exp45" / "e26x"
BIRTHS = HERE / "exp45" / "e26"


def cross_life(birth_arm, update_arm, seed, out, cmd):
    assert birth_arm in C.ARMS and update_arm in C.ARMS
    out = Path(out)
    orig = (C.birth_path, C.admit_e26, C.life_path, C.provenance)
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    C.birth_path = lambda _out, _arm, s: orig[0](BIRTHS, birth_arm, s)
    C.admit_e26 = lambda bfile, _arm, s: orig[1](bfile, birth_arm, s)
    C.life_path = lambda _out, _arm, s: Path(_out) / f"e26x_life_{birth_arm}{update_arm}{E.DEVTAG}{E.DRYTAG}_{s}.json"
    C.provenance = lambda c: dict(orig[3](c), crossover=dict(birth_arm=birth_arm, update_arm=update_arm, update_local=C.ARMS[update_arm],
                                                              runner="exp26x_cross.py", runner_sha256=me))
    try:
        C.life(update_arm, seed, out, cmd)
    finally:
        C.birth_path, C.admit_e26, C.life_path, C.provenance = orig


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = " ".join([Path(sys.executable).name, Path(__file__).name, *a])
    out = Path(a[a.index("--out") + 1]) if "--out" in a else OUT
    assert a[0] == "life" and len(a) >= 4, "usage: exp26x_cross.py life <S|B birth> <S|B update> <seed> [--out DIR]"
    cross_life(a[1], a[2], int(a[3]), out, cmd)
