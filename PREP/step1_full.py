"""Step 1 at full length on the fast program (the reviewer's task of 19 November, 05:21 UTC): one tissue a process.

    python3 PREP/step1_full.py <seed> <first|coincidence|fixed> <out.json>

The tissue is TISSUE_EVENT/evt.py's EvTissue (E57's worker's fast program, on origin/claude/e63-tissue-event-017dSE, not edited), with
floor=1/128, tol=1/1024, trace_sender=True, strength_byte=True, over the 262,144 bytes of tissue_step1.stream(seed, 262144). In four
windows of 8,192 bytes, starting at 0, 86,016, 172,032 and 253,952, the output of every cell after the last step of each byte (x) is
recorded, values below 0.01 as zero; at each window's end the distance of every cell (distances(the byte values present in the
window)); the window is read with PREP/readings.py (shares and scores at d = 1, 2, 4; lags 0, 1, 2; C0). Every 16,384 bytes: the
connections (live connections other than from the two feedback lines), their mean a cell, and the CPU-seconds a byte since the last
record. Outside the windows the bytes run in chunks of 1,024 through run_bytes. The process stops itself if its projected CPU time
passes 30 minutes.

The core keeps a log of every connection made and given up (evt_events_made, evt_events_gone); its growth is computed in int and
overflows past 33,554,432 entries (8 * 4 * cap_gone > 2^31), which ended the first two coincidence tissues with a segmentation fault near
byte 250,000. The log is a record and takes no part in a step; the runner reads its length and clears it with the core's own
evt_clear_events() every 16,384 bytes, so it never grows past one record's events. evt.py and evt_core.c are not changed.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "TISSUE_EVENT"))
import evt                            # noqa: E402
import readings as RD                 # noqa: E402

ts = evt.ts
LENGTH, WL = 262144, 8192
STARTS = (0, 86016, 172032, 253952)
EVERY = 16384
LIMIT = 30 * 60


def main(seed, rule, out):
    seed = int(seed)
    t0 = time.process_time()
    st, *_ = ts.stream(seed, LENGTH)
    st = np.asarray(st, dtype=np.uint8)
    T = evt.EvTissue(seed, rule, floor=1 / 128, tol=1 / 1024, trace_sender=True, strength_byte=True)
    tb = time.process_time()
    rec = dict(seed=seed, rule=rule, switches=T.switches, length=LENGTH, stream_sha256=__import__("hashlib").sha256(st.tobytes()).hexdigest(),
               birth_cpu_s=tb - t0, records=[], windows={})
    wins = {s: None for s in STARTS}
    pos, last_t, last_pos = 0, time.process_time(), 0
    events = dict(made=0, gone=0)

    def conn():
        snd, rcv, w, k0 = T.live()
        ordinary = snd < ts.YES_LINE
        return int(ordinary.sum())

    while pos < LENGTH:
        win = next((s for s in STARTS if s <= pos < s + WL), None)
        if win is not None:
            if wins[win] is None:
                wins[win] = np.zeros((WL, ts.N), np.float16)
            for p in range(pos, min(win + WL, (pos // EVERY + 1) * EVERY)):
                T.run_bytes(st[p:p + 1])
                xx = np.asarray(T.x, dtype=np.float64)
                wins[win][p - win] = np.where(xx < 0.01, 0.0, xx).astype(np.float16)
            pos = p + 1
            if pos == win + WL:
                wb = st[win:win + WL]
                dist = T.distances(sorted(set(int(b) for b in wb)))
                r = RD.read_window(wins[win], wb, dist, seed, stream=st, start=win, lags=(0, 1, 2))
                r.pop("c_lt", None)
                rec["windows"][str(win)] = dict(start=win, end=win + WL - 1, readings=r,
                                                cells_by_distance={int(d): int((dist == d).sum()) for d in (1, 2, 4)})
                wins[win] = None
        else:
            nxt = min([s for s in STARTS if s > pos] + [LENGTH, (pos // EVERY + 1) * EVERY])
            for a in range(pos, nxt, 1024):
                T.run_bytes(st[a:min(a + 1024, nxt)])
            pos = nxt
        if pos % EVERY == 0:
            now = time.process_time()
            c = conn()
            made = int(evt.L.evt_events_made(np.zeros(0, np.int64), 0)); gone = int(evt.L.evt_events_gone(np.zeros(0, np.int64), 0))
            evt.L.evt_clear_events()          # the core's own reset of its event log (a record only; it takes no part in a step)
            events["made"] += made; events["gone"] += gone
            rec["records"].append(dict(byte=pos, connections=c, per_cell=c / ts.N, made_since_last=made, given_up_since_last=gone, cpu_ms_a_byte=1000 * (now - last_t) / (pos - last_pos),
                                       cpu_s=now - t0))
            print(f"{rule} {seed} byte {pos} connections {c} ({c / ts.N:.2f} a cell) {1000 * (now - last_t) / (pos - last_pos):.3f} ms a byte", flush=True)
            last_t, last_pos = now, pos
            proj = (now - tb) / pos * LENGTH
            if proj > LIMIT:
                rec["stopped"] = dict(at_byte=pos, projected_cpu_s=proj)
                break
    rec["cpu_s"] = time.process_time() - t0
    rec["counters"] = T.counters()
    rec["events_total"] = events
    rec["complete"] = "stopped" not in rec
    Path(out).write_text(json.dumps(rec, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("done", rule, seed, round(rec["cpu_s"], 1), "s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
