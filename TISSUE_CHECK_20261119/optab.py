"""Table of outpart.py runs (out/op/): yes by quarter (last quarter first), speakers, one voice, largest weight-row norm, G_max, th."""
import json, os
O = "/home/user/tissue_check/out/op/"
def row(lab):
    f = O + lab + ".json"
    if not os.path.exists(f): return f"{lab}: -"
    d = json.load(open(f)); q = d["quarters"]
    return (f"{lab:22s} last quarter {100*q[3]:5.2f} (quarters {' '.join(f'{100*x:.1f}' for x in q)}) speakers {d['speakers_mean_last_quarter']:.2f} "
            f"one voice {100*d['one_voice_last_quarter']:.0f}% silent {100*d['silent_last_quarter']:.0f}% |W| {d['max_row_norm']:.3g} G {d['G_max']:.3g} th {d['th']['min']:.3f}-{d['th']['max']:.3f}")
for s in (73129, 93683, 116089):
    print(f"== seed {s}")
    for lab in ("one_s%d_256k", "m8_s%d_256k", "m32_s%d_256k", "one_s%d_64k", "m8_s%d_64k", "m8g1_s%d_64k", "m8g2_s%d_64k", "m8g3_s%d_64k", "m8g4_s%d_64k",
                "m8noinh_s%d_64k", "m8nothr_s%d_64k", "m8neither_s%d_64k", "m8half_s%d_64k", "m8fb_s%d_64k", "onefb_s%d_64k"):
        print("  " + row(lab % s))
