"""Checks of lt6 against the owner's requirements and basic function (25 Nov 2026). Each check states its criterion before it runs.
Usage: python3 checks.py [names...]   (default: all)   ->  one line per check: PASS / FAIL / REPORT and the numbers"""
import sys, os, time, json, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lt6 import Net6 as Net5, run_trace, test_trace, biasonly_test, outstats
D = "/Users/pencilbard/Downloads/digitalbaby_newnet/"
SHAK = np.frombuffer(open(D + "PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/nano/data/tinyshakespeare.txt", "rb").read(), np.uint8)
SMALL = dict(ntis=2048, ninh=512, nrly=512)
out = []


def say(name, ok, **kw):
    tag = "REPORT" if ok is None else ("PASS" if ok else "FAIL")
    line = f"{tag:6s} {name}: " + json.dumps(kw); print(line, flush=True); out.append(line)


def c1_test_is_pure():
    """criterion: a test (learning off) leaves no trace: two tests give identical losses, and a life with tests in it ends exactly
    like the same life without them"""
    a = Net5(SMALL); b = Net5(SMALL)
    a.run(SHAK[:20000]); b.run(SHAK[:20000])
    t1 = a.test(SHAK[300000:305000]); t2 = a.test(SHAK[300000:305000])
    a.run(SHAK[20000:30000]); b.run(SHAK[20000:30000])
    fa, fb = a.test(SHAK[300000:305000]), b.test(SHAK[300000:305000])
    say("c1_test_is_pure", t1 == t2 and fa == fb, two_tests=[t1[0], t2[0]], with_tests=fa[0], without_tests=fb[0])


def c2_requirements():
    """criteria after 64k bytes of English: 0 locality violations; 0 synapses touching answer lines; 0 synapses into interface cells;
    tissue cells have inputs; tissue cells read relay; relay cells have inputs; fan-in is spread (no single value holds > 50% of
    connected cells, so no cap is acting); firing rates differ between cells (p90 > 2 x p10)"""
    n = Net5(SMALL); n.run(SHAK[:64000]); a = n.audit(); h = n.fanin(); conn = h[1:].sum(); top = h[1:].max() / max(conn, 1)
    ok = (a["locality_violations"] == 0 and a["syn_touching_answer"] == 0 and a["syn_into_interface"] == 0 and a["tis_with_input"] > 0 and
          a["tis_reading_relay"] > 0 and a["relay_with_input"] > 0 and top <= 0.5 and a["tis_rate_p90_x1e6"] > 2 * max(a["tis_rate_p10_x1e6"], 1))
    nz = {int(i): int(v) for i, v in enumerate(h) if v}
    say("c2_requirements", ok, audit=a, fanin_hist=nz, largest_share=round(float(top), 3))


def cyc(n): return np.frombuffer((b"abcdefg" * (n // 7 + 1))[:n], np.uint8)


def c3_cycle():
    """criterion: on a repeating 7-byte cycle, held-out loss after 20k bytes < 0.1 nats/byte"""
    n = Net5(SMALL); curve = []
    for k in range(4): n.run(cyc(5000)); curve.append(round(n.test(cyc(2000))[0], 4))
    say("c3_cycle", curve[-1] < 0.1, test_nats_every_5k=curve)


GAP = b"1234567890" * 4            # the answer is 41 bytes after the cue; 41 is not among the default lags (passages 0-7, 12-15, 24-27, 48-51, 96-99, 192-195)


def mem_stream(nlines, seed):
    r = random.Random(seed); s = b"".join(b"A" + GAP + b"B\n" if r.random() < 0.5 else b"C" + GAP + b"D\n" for _ in range(nlines)); return np.frombuffer(s, np.uint8)


def answer_loss(per, x):
    idx = [i for i in range(len(x)) if x[i] in (ord("B"), ord("D")) and per[i] >= 0]; return float(np.mean(per[idx]))


def c4_memory(relay=True):
    """criterion: the last byte of a line depends on its first byte, 41 bytes back, not a lag of any input line, so it must be carried inside the tissue; held-out loss at
    that byte < 0.5 nats (0.69 = knowing only that it is B or D). Lines are 43 bytes; 10,000 bytes per step."""
    n = Net5(SMALL)
    if not relay: n.norelay()
    curve = []
    for k in range(6): n.run(mem_stream(233, k)); te = mem_stream(100, 99); curve.append(round(answer_loss(test_trace(n, te), te), 4))
    return curve


def c4():
    a = c4_memory(True); say("c4_memory_beyond_window", a[-1] < 0.5, answer_nats_every_10k=a)


def c5_relay_ablation():
    """report: the same lives with the relay switched off (feedback reaches the output cells only). If the relay carries useful
    feedback into the tissue, the full net should be lower. Criterion stated: full < no-relay on the memory task by > 0.02 nats"""
    a = c4_memory(True); b = c4_memory(False)
    e1 = Net5(SMALL); e2 = Net5(SMALL); e2.norelay(); e1.run(SHAK[:64000]); e2.run(SHAK[:64000])
    t1, t2 = e1.test(SHAK[300000:310000])[0], e2.test(SHAK[300000:310000])[0]
    say("c5_relay_ablation", a[-1] < b[-1] - 0.02, memory_full=a[-1], memory_norelay=b[-1], english_full=round(t1, 4), english_norelay=round(t2, 4))


def c6_cost():
    """criteria: time per byte grows at most linearly with the number of cells (exponent <= 1.15 over x1, x2, x4), and does not grow
    with bytes read (time per byte at 40-45k within 1.5x of 10-15k); synapse count reported"""
    ts, syn = [], []
    for s in (1, 2, 4):
        n = Net5(SMALL, scale=s); n.run(SHAK[:10000]); t = time.time(); n.run(SHAK[10000:15000]); t1 = (time.time() - t) / 5000
        n.run(SHAK[15000:40000]); t = time.time(); n.run(SHAK[40000:45000]); t2 = (time.time() - t) / 5000
        a = n.audit(); ts.append((t1, t2)); syn.append(sum(a[k] for k in ("syn_exc", "syn_inh", "syn_dis", "syn_out", "syn_rly", "syn_rd")))
    ex = float(np.polyfit(np.log2([1, 2, 4]), np.log2([x[1] for x in ts]), 1)[0]); grow = max(x[1] / x[0] for x in ts)
    say("c6_cost", ex <= 1.15 and grow <= 1.5, us_per_byte=[[round(1e6 * a, 1), round(1e6 * b, 1)] for a, b in ts], exponent_in_cells=round(ex, 2),
        late_over_early=round(grow, 2), synapses_at_45k=syn)


def c7_tissue_reaches_output():
    """criterion: on English, the synaptic drive must matter: held-out loss with all synapses is lower than with output biases
    only by more than 0.1 nats/byte after 64k bytes"""
    n = Net5(SMALL); curve = []
    for k in range(4):
        n.run(SHAK[k * 16000:(k + 1) * 16000]); x = SHAK[300000:310000]; curve.append((round(n.test(x)[0], 4), round(biasonly_test(n, x)[0], 4)))
    say("c7_tissue_reaches_output", curve[-1][1] - curve[-1][0] > 0.1, full_vs_biasonly_every_16k=curve, out=outstats(n))


CHECKS = dict(c1=c1_test_is_pure, c2=c2_requirements, c3=c3_cycle, c4=c4, c5=c5_relay_ablation, c6=c6_cost, c7=c7_tissue_reaches_output)
if __name__ == "__main__":
    names = sys.argv[1:] or list(CHECKS)
    for k in names: t = time.time(); CHECKS[k](); print(f"       ({k} took {time.time() - t:.0f}s)", flush=True)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks_out.txt"), "a").write("\n".join(out) + "\n")
