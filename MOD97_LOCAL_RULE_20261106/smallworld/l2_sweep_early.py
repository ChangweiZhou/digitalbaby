"""E44, job L2, the sweep's long lives: the reviewer's addition of 17 November 2026 (22:35 UTC) to the reading "Between the rules"; no run
changes. From the score files that l2_sweep_report.py wrote (scores/ and main/, V = 1,024), with the main worker's scorer unchanged:

  early level   the mean of D2 (score by age on reduction items) over the first 64 visits of a life (visits 0 to 63)
  late level    D4 at 256 and at 1,024 visits, as the score files hold it
  for each pair (`sweep` against `naive`, `sweep+replay` against `replay`) and each length: the early difference, the late difference, and
  the change (late difference minus early difference) per world, with the mean and 90% interval over the worlds as D11 forms them;
  *grown*, *not changed* or *shrunk* by that interval.

    python3 l2_sweep_early.py      # -> exp45/e44/L2/sweep/final/E44_L2_SWEEP_EARLY.json and E44_L2_SWEEP_EARLY.md (refused if they exist)
Nothing here trains.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l2_score as L                  # noqa: E402
import l2_sweep_report as RP          # noqa: E402

OUT = RP.SW / "final"
EARLY = 64


def main():
    js, md = OUT / "E44_L2_SWEEP_EARLY.json", OUT / "E44_L2_SWEEP_EARLY.md"
    assert not js.exists() and not md.exists(), "refusing existing outputs"
    J = dict(note=__doc__.strip(), early_visits=[0, EARLY - 1], pairs={})
    lines = ["# E44, job L2, the sweep's long lives: early level and the change of the difference", "",
             "Written by `smallworld/l2_sweep_early.py` from the score files under `exp45/e44/L2/sweep/scores/` and `exp45/e44/L2/sweep/main/` "
             "(the scorer `smallworld/l2_score.py`, unchanged). Early level: the mean score on reduction items (D2) over visits 0 to 63 of a life. "
             "Late level: D4. A difference is the sweep's figure minus back-propagation's; the change is the late difference minus the early "
             "difference. Intervals: two-sided 90%, Student's t over the four worlds (D11). The reviewer's addition of 17 November.", ""]
    rows = []
    for x, y in RP.PAIRS:
        key = f"{x} against {y}"; J["pairs"][key] = {}
        for V in RP.VS:
            per = {}
            for w in RP.WORLDS:
                qx, qy = RP.q_of(x, w, 1024), RP.q_of(y, w, 1024)
                ex, ey = float(np.mean(qx["score_by_age"][:EARLY])), float(np.mean(qy["score_by_age"][:EARLY]))
                lx, ly = RP.q_of(x, w, V)["late_level"], RP.q_of(y, w, V)["late_level"]
                per[w] = dict(early_x=ex, early_y=ey, early_difference=ex - ey, late_x=lx, late_y=ly, late_difference=lx - ly,
                              change=(lx - ly) - (ex - ey))
            ie = L.interval90([per[w]["early_difference"] for w in RP.WORLDS])
            il = L.interval90([per[w]["late_difference"] for w in RP.WORLDS])
            ic = L.interval90([per[w]["change"] for w in RP.WORLDS])
            word = {"higher": "grown", "lower": "shrunk", "not distinguished": "not changed"}[L.verdict(ic)]
            J["pairs"][key][str(V)] = dict(per_world=per, early_difference=ie, late_difference=il, change=ic, change_word=word,
                                          early_verdict=L.verdict(ie), late_verdict=L.verdict(il))
            iv = lambda i: f"{RP.sgn(i['mean'])} [{RP.sgn(i['low'])}, {RP.sgn(i['high'])}]"
            for w in RP.WORLDS:
                p = per[w]
                rows.append([f"`{x}` − `{y}`", V, w, f"{p['early_x']:.3f} − {p['early_y']:.3f} = {RP.sgn(p['early_difference'])}",
                             f"{p['late_x']:.3f} − {p['late_y']:.3f} = {RP.sgn(p['late_difference'])}", RP.sgn(p["change"])])
            rows.append([f"`{x}` − `{y}`", V, "mean [90%]", f"{iv(ie)} {L.verdict(ie)}", f"{iv(il)} {L.verdict(il)}", f"{iv(ic)} **{word}**"])
    lines += [RP.table(["Pair", "Late window at", "World", "Early level (visits 0 to 63)", "Late level (D4)", "Change of the difference"], rows), ""]
    for x, y in RP.PAIRS:
        key = f"{x} against {y}"; c = J["pairs"][key]["1024"]
        lv = {"higher": "ahead", "lower": "behind", "not distinguished": "level"}[c["late_verdict"]]
        lines += [f"**`{x}` against `{y}`, the third clause:** from the first 64 visits to the late window of 1,024 visits the difference has "
                  f"**{c['change_word']}** (change {RP.sgn(c['change']['mean'])} [{RP.sgn(c['change']['low'])}, {RP.sgn(c['change']['high'])}]); "
                  f"early difference {RP.sgn(c['early_difference']['mean'])} ({c['early_verdict']}), late difference "
                  f"{RP.sgn(c['late_difference']['mean'])} ({lv}).", ""]
    js.write_text(json.dumps(J, indent=1)); md.write_text("\n".join(lines))
    print("\n".join(lines[-4:]))


if __name__ == "__main__":
    main()
