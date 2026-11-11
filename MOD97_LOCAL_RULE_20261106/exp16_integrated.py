"""E16: the integrated toy learner, the Phase A exit gate (predictions: E45_PREDICTIONS.md, E16).

One E11 cell learner (10-byte window, hidden 512, diagonal memory cell with the exact trace) from the E11 memory birth,
at the adopted settings (decay 0.25 on local arms, fresh-render replay at half of every batch, scan trace), on ONE
stream that carries every family the toy stage has, in six blocks of STAGE_UPDATES updates:
    blocks 1-2  F1: add/sub/max/min mod 97 (codes r t x n), interleaved, k = 0..12 distractor letters (holding)
    blocks 3-4  F2: 12 taught ordered operator pairs on the mod-97 operations, E8 scratchpad form
                    c a a o1 b b o2 c c =  ->  i i ; f f EOS      (no distractors; the window is 10, so the code byte
                    and the first operand leave the window before the final answer, the operators do not)
    blocks 5-6  F3: add/sub/max/min mod 73 (codes s u y z), interleaved, k = 0..12  (a later family: does F1 survive it)
Every block after the first mixes half new items with half replay of stored (family, operands) entries re-rendered fresh.
Scored after every block: F1 and F3 sealed panels at k = 0, 8, 12, 20; the four held cross pairs (+-, -+, ><, <>) and
held triples of taught pairs ("within"); the E11 development operations (mod 89) at k = 12; the silenced-cell audit.
Arms: T  sign rule everywhere + exact trace, decay 0.25, replay (the learner)
      D  backprop through time, each batch drawn from the families seen so far with equal weight, decay 0.3 (the twin)
"""
import os
import re
os.environ.setdefault("E11_WD", "0.25")
os.environ.setdefault("E11_REPLAY", "fresh")
os.environ.setdefault("E11_TRACE", "scan")
import json
import sys
import time
from pathlib import Path
from random import Random

import torch as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp11_cell as E
import exp45 as X

assert E.WD == 0.25 and E.REPLAY == "fresh" and E.TRACE == "scan" and E.GATES == "single"
OUT, EOS, BATCH, STAGE_UPDATES, DRYTAG = E.OUT, E.EOS, X.BATCH, E.STAGE_UPDATES, E.DRYTAG
# E16-D replay controls on arm T (default reproduces E16): uniform = one buffer of 1,024 entries per block, replay drawn uniformly;
# balanced = the same buffer kept per family, each replay draw from a family chosen uniformly among those in the buffer;
# all_balanced = every presented entry kept per family (distinct), draws balanced per family;
# oracle = the whole batch is the twin's joint stream (fresh items of every family seen so far, equal weight), learned by the local rule;
# share = per-family buffers of 1,024 per block, the replay half split equally among the EARLIER families (the current family is never
#         replayed), so each earlier family's exposure rises (F1 50% of the batch during F2; F1 and F2 25% each during F3).
# Arm M (added 8 Nov 12:30 EST, before E16-D's results): backprop through time at decay 0.25 on exactly arm T's uniform-replay batches,
# so the rule is the only difference from T.
# reservoir (E16-R, 8 Nov 14:45 EST): per-family reservoir of at most E16_CAP entries (RCAP; CAP below is the response length cap) filled by Algorithm R over the family's
#         presentations (one counter and one fixed-size list per family: bounded memory, O(1) work per event); draws balanced per family.
# total (E16-R2, 8 Nov 16:25 EST): ONE reservoir of E16_CAP entries over all presentations of the whole life, no family label anywhere
#         (Algorithm R with a single counter); the replay half is drawn uniformly from it. Bounded by a total budget, label-free.
REPLAY16 = os.environ.get("E16_REPLAY", "uniform")
# raw (E16-R3, 8 Nov 17:30 EST): as total, but the reservoir holds the presented BYTES (prompt and answer exactly as they appeared) and
#         replays them verbatim: no parsing into task and operands, no re-rendering, no recomputed answer. The parser is out of the replay path.
assert REPLAY16 in ("uniform", "balanced", "all_balanced", "oracle", "share", "reservoir", "total", "raw")
RCAP = int(os.environ.get("E16_CAP", "4096"))
RTAG = "" if REPLAY16 == "uniform" else (f"_res{RCAP}" if REPLAY16 == "reservoir" else f"_tot{RCAP}" if REPLAY16 == "total" else f"_raw{RCAP}" if REPLAY16 == "raw" else f"_{REPLAY16}")
# E16-S / E16-L (8 Nov 19:10 EST), the environment-side scaffolds, one change each:
#   E16_STREAM=blocks|drift   drift: no stages; each new item's family is drawn from weights that move smoothly through the life
#                             (raised-cosine bumps centred where the blocks were, half-width a third of the life), scored at the same six checkpoints
#   E16_ADMIT=block|online    online: the raw reservoir admits each presented item as it arrives (Algorithm R per item) instead of at checkpoint ends
#   E16_LOSS=answer|all       all: next-byte loss on every byte of every item (prompt and answer), the answer mask removed; scoring unchanged
STREAM16 = os.environ.get("E16_STREAM", "blocks"); ADMIT16 = os.environ.get("E16_ADMIT", "block"); LOSS16 = os.environ.get("E16_LOSS", "answer")
#   E16_STORE=on|off          off: the E8 exact-key store is absent from the local arms (no writes, no logit bias at scoring); the architecture audit (E16-A)
#   E16_ARCH=window|cell1|bank  (E17, 8 Nov 22:30 EST) window: unchanged; cell1 / bank: the window removed, see arch17.py; births e17_dev_{arch}_{seed}
ARCH16 = os.environ.get("E16_ARCH", "window"); assert ARCH16 in ("window", "cell1", "bank", "cell1d2", "bank2")
STORE16 = os.environ.get("E16_STORE", "on" if ARCH16 == "window" else "off"); assert STORE16 in ("on", "off")
if ARCH16 != "window":
    import arch17
    arch17.install(ARCH16)
PFX = "e16" if ARCH16 == "window" else "e17"
#   E16_FREEZE=none|bank1  (E17 Phase 1b): bank1 freezes the first bank's write path (l1, lp, lc, lw) after birth; only the second bank, the
#                           projections from the first bank's state and the readout learn in life. Separates the truncated first-bank trace from the rest.
FREEZE16 = os.environ.get("E16_FREEZE", "none"); assert FREEZE16 in ("none", "bank1", "book")   # book (E17 Phase 3, 9 Nov 22:45 EST): the embedding table alone frozen after birth, both banks learning; tag _frzb
assert FREEZE16 == "none" or ARCH16 == "bank2", "E16_FREEZE=bank1|book is defined for bank2 only"
#   E16_FEEDBACK=sign|exact  (E17 Phase 1c): exact replaces the sign-symmetric feedback through every layer by the true transpose (SignLinear -> linear)
#                           while the state synapses still learn from the eligibility trace (through_time stays off for arm T): separates the error
#                           signal delivered to the state from the trace that credits its synapses. Arm T only; D and M already use exact feedback.
FEEDBACK16 = os.environ.get("E16_FEEDBACK", "sign"); assert FEEDBACK16 in ("sign", "exact")
if FEEDBACK16 == "exact":
    _fwd16 = E.forward
    E.forward = lambda m, x, local=False, through_time=False, keep=False, silence=False: _fwd16(m, x, False, through_time, keep, silence)
#   E16_TRACEUP=none|proj    (E17 Phase 1d): proj extends the scan trace one layer upstream, to the projection feeding each bank's write (lp2; lp when
#                           the first bank learns), replacing its within-step credit by the exact credit through time given Lambda (arch17.TRACEUP).
#   E16_TRACEUP=all          (E17 Phase 4, 10 Nov): credit carried through time at EVERY layer, still delivered through sign(W) at each layer
#                           (the sign rule's own feedback): the first bank's projection, the first layer and the embeddings get the same
#                           through-time credit the write synapses always had, and the first bank receives the second bank's credit through time.
#                           Computed as a backward pass over the item with sign-of-weight feedback (forward kept through time, local=True);
#                           the scan is then redundant and is switched off. Arm T, sign feedback only (with exact feedback it would be arm M). Tag _tall.
TRACEUP16 = os.environ.get("E16_TRACEUP", "none"); assert TRACEUP16 in ("none", "proj", "all")
assert TRACEUP16 == "none" or ARCH16 != "window", "E16_TRACEUP is defined for the E17 architectures only"
assert TRACEUP16 != "all" or FEEDBACK16 == "sign", "E16_TRACEUP=all with exact feedback is backprop through time: use arm M"
if ARCH16 != "window":
    arch17.TRACEUP = TRACEUP16 == "proj"; arch17.FB_SIGN = FEEDBACK16 == "sign"
if TRACEUP16 == "all":
    _fwd16all = E.forward
    E.forward = lambda m, x, local=False, through_time=False, keep=False, silence=False: _fwd16all(m, x, local, (True if local else through_time), (False if local else keep), silence)
    E.trace_grads_scan = E.trace_grads = lambda m, it: None      # the backward pass already carries the credit through time
#   E16_BOOKDECAY=on|off     (9 Nov audit, Fable 5.2 memo section 3): under the local arms the byte embeddings of every prompt symbol that enters only
#                           through the state (operators, letters, task codes) receive no gradient, and AdamW's decay erased them (factor 0.215 per
#                           block), which confounds E17 Phases 1-1c. "off" puts `book` in a no-decay group in lives AND births (birth file tagged
#                           `_nbd`), and the frozen arm freezes `book` with the first bank. Default "on" reproduces every receipt made before.
#   E16_HEADS=top|local      (E18, 10 Nov): local gives the first bank its own next-byte head and cuts every gradient path between the two banks
#                           (arch17.LOCAL_HEADS): each bank learns from its own prediction error, no error is relayed between banks. bank2 only. Tag _lh.
#   E16_BIRTHRULE=D|T        (E18): the rule used by `dev`. D (default) is backprop through time, as every birth before. T is the life's own rule with
#                           the feedback and trace settings in force (E16_FEEDBACK, E16_TRACEUP), decay 0.25: a birth with no backprop anywhere.
#                           Birth files and the lives started from them carry _bT (the birth also its _xfb/_tup/_tall).
#   Added after the 10 November source audit (PROJECT_DOCUMENTS/MINIFLY_E16_E19_HOSTILE_SOURCE_AUDIT_20261110.md); every default reproduces earlier receipts:
#   E16_HEADS=aux            the first bank's own head with the relay between banks kept: the one-factor control for E16_HEADS=local. Tag _ah.
#   E16_PROJFB=raw|scaled    feedback map inside the projection trace (E16_TRACEUP=proj): raw sign(W) as Phases 3-4 ran it, or sign(W) * mean|W|
#                           as every other relay uses. Tag _tups for scaled.
#   E16_FIXEDCODES=on        the byte embeddings stay at their random initial values in birth and life (a labelled restriction: identical fixed
#                           input codes for every arm, so that a rule's ability to learn its input code is separated from the rest). Tag _fc.
#   E16_OUTTAG=<text>        appended to a life's receipt name, so that a repeat on the current panels never overwrites an original receipt.
#   E16_HEADS=relay          (E18c, 11 Nov 2026) as local, except that the first bank's state feeds the second bank's projection with
#                           its gradient (the one read between the banks); the main readout's direct reads of the first bank (Hs -> lm,
#                           z1 -> l2) stay detached, so the main head's error reaches the first bank only through the second. Tag _rh.
HEADS16 = os.environ.get("E16_HEADS", "top"); assert HEADS16 in ("top", "local", "aux", "relay")
assert HEADS16 == "top" or (ARCH16 == "bank2" and FREEZE16 == "none"), "E16_HEADS=local|aux is defined for bank2 with nothing frozen"
PROJFB16 = os.environ.get("E16_PROJFB", "raw"); assert PROJFB16 in ("raw", "scaled")
assert PROJFB16 == "raw" or (TRACEUP16 == "proj" and FEEDBACK16 == "sign"), "E16_PROJFB=scaled applies to E16_TRACEUP=proj with sign feedback"
FIXEDCODES16 = os.environ.get("E16_FIXEDCODES", "off"); assert FIXEDCODES16 in ("on", "off")
assert FIXEDCODES16 == "off" or ARCH16 != "window", "E16_FIXEDCODES is defined for the E17 architectures only"
OUTTAG16 = os.environ.get("E16_OUTTAG", ""); assert re.fullmatch(r"[A-Za-z0-9_]*", OUTTAG16)
#   E16_COMPGAP=<k>          (E20, redesigned after the second audit of 10 Nov) distractor letters around composition prompts in teaching:
#                           each taught composition gets a total of g1 + g2 letters (each 0..k); with probability one half they go INSIDE
#                           (g1 after the first operator, g2 after the second), otherwise they LEAD the prompt. Same lengths either way; inside,
#                           the first operand leaves a ten-byte window, leading, every operand stays in view. The registered composition
#                           panels are unchanged (no letters). Memory use is measured on separate panels (memory_panels): twins that are
#                           byte-identical in the last ten bytes at "=" and differ only in the hidden first operand, scored both-correct on
#                           whole answers without the store, with the same items re-rendered with leading letters as the in-view control.
#                           Raw or oracle replay, or arm D. Tag _cm<k>.
COMPGAP16 = int(os.environ.get("E16_COMPGAP", "0")); assert 0 <= COMPGAP16 <= 12
CGTAG = f"_cm{COMPGAP16}" if COMPGAP16 else ""
MEMORY = [None]
CENSUS_V2 = [None]
LAST_AUX = [None]      # the first bank's head loss of the last update (the logged loss is the main head's)
BIRTHRULE16 = os.environ.get("E16_BIRTHRULE", "D"); assert BIRTHRULE16 in ("D", "T")
assert BIRTHRULE16 == "D" or ARCH16 != "window", "E16_BIRTHRULE=T is defined for the E17 architectures only"
HTAG = ("_lh" if HEADS16 == "local" else "_ah" if HEADS16 == "aux" else "_rh" if HEADS16 == "relay" else "") + ("_fc" if FIXEDCODES16 == "on" else "")
TUPTAG = ("_tups" if PROJFB16 == "scaled" else "_tup") if TRACEUP16 == "proj" else "_tall" if TRACEUP16 == "all" else ""
DEVRULETAG = "" if BIRTHRULE16 == "D" else "_bT" + ("_xfb" if FEEDBACK16 == "exact" else "") + TUPTAG
if HEADS16 != "top":
    arch17.LOCAL_HEADS = True; arch17.LOCAL_DETACH = HEADS16 == "local"
    if HEADS16 == "relay":      # only the read between the banks (first bank's state into the second bank's projection) open
        arch17.LOCAL_DETACH = True; arch17.P2_DETACH = False; arch17.LM_DETACH = True; arch17.Z1_DETACH = True
if ARCH16 != "window":
    arch17.PROJ_SCALED = PROJFB16 == "scaled"
BOOKDECAY16 = os.environ.get("E16_BOOKDECAY", "on"); assert BOOKDECAY16 in ("on", "off")
assert BOOKDECAY16 == "on" or ARCH16 != "window", "E16_BOOKDECAY=off is defined for the E17 architectures only"
#   E16_FARHEAD=<d>          (E23, 11 Nov 2026) a third head on the SECOND bank's state, trained on the byte d steps ahead (d = 1 is the next
#                           byte) at every position where that byte exists in the item, added to the loss with weight 1; births only,
#                           bank2 with local heads (so its error reaches the second bank alone) and the rule-T birth. Tag _fh<d>.
#   E16_FARCTRL=none|shuffle shuffle: the far head's targets are permuted among the valid positions of the SAME item, afresh at each update
#                           (same bytes, same count, same head; the correspondence in time destroyed). Tag suffix s.
#                           The far head is trained on PROMPT bytes only: a position whose far target is an answer byte or EOS is
#                           excluded in every E23 arm (the main head already supervises answers; internal audit of 11 Nov).
#   E16_HEAD1LOSS=answer|all (E23) all: the first bank's head is trained on every next byte (the lower bank predicts the near observation)
#                           while the main head keeps the answer loss. Births only, E16_HEADS=local. Tag _h1all.
HEAD1LOSS16 = os.environ.get("E16_HEAD1LOSS", "answer"); assert HEAD1LOSS16 in ("answer", "all")
assert HEAD1LOSS16 == "answer" or (HEADS16 == "local" and LOSS16 == "answer"), "E16_HEAD1LOSS=all is defined for E16_HEADS=local with the answer loss"
H1TAG = "_h1all" if HEAD1LOSS16 == "all" else ""
FARHEAD16 = int(os.environ.get("E16_FARHEAD", "0")); assert 0 <= FARHEAD16 <= 12
FARCTRL16 = os.environ.get("E16_FARCTRL", "none"); assert FARCTRL16 in ("none", "shuffle")
assert FARCTRL16 == "none" or FARHEAD16 > 0, "E16_FARCTRL needs E16_FARHEAD"
assert FARHEAD16 == 0 or (HEADS16 == "local" and BIRTHRULE16 == "T" and LOSS16 == "answer" and FARHEAD16 <= 4), "E16_FARHEAD (1-4) is defined for E16_HEADS=local E16_BIRTHRULE=T births with the answer loss"
FTAG = (f"_fh{FARHEAD16}" + ("s" if FARCTRL16 == "shuffle" else "")) if FARHEAD16 else ""
if ARCH16 != "window":      # harness merge, 11 Nov: on the E18b/E18c/E23 branch this line ran for every architecture and the window learner could not be imported
    arch17.FAR_HEAD = FARHEAD16 > 0
#   E16_LOSSW=uniform|balanced  (E24, 11 Nov) births with E16_LOSS=all and one head at the top: balanced gives the answer positions half the loss
#                               and every other real position the other half (a diagnostic: it uses the answer mask to set the weights)
LOSSW16 = os.environ.get("E16_LOSSW", "uniform"); assert LOSSW16 in ("uniform", "balanced")
assert LOSSW16 == "uniform" or (LOSS16 == "all" and HEADS16 == "top" and FARHEAD16 == 0), "E16_LOSSW=balanced is defined for E16_LOSS=all births with E16_HEADS=top"
FAR_GEN = [None]       # the shuffle control's own generator (never the global torch RNG); seeded by develop17
LAST_FAR = [None]
BTAG = "_nbd" if BOOKDECAY16 == "off" else ""
ATAG = ("" if ARCH16 == "window" else f"_{ARCH16}") + BTAG + HTAG + ("_frz1" if FREEZE16 == "bank1" else "_frzb" if FREEZE16 == "book" else "") + ("_xfb" if FEEDBACK16 == "exact" else "") + TUPTAG + ("_bT" if BIRTHRULE16 == "T" else "")


def adamw16(m, lr, wd, frozen=()):
    """AdamW over the learnable parameters; with E16_BOOKDECAY=off the embedding table is in a group of its own with no decay."""
    learn = [(n_, p_) for n_, p_ in m.named_parameters() if not n_.startswith(frozen)]
    for n_, p_ in m.named_parameters():
        p_.requires_grad_(not n_.startswith(frozen))
    if BOOKDECAY16 == "off":
        groups = [{"params": [p_ for n_, p_ in learn if not n_.startswith("book.")]},
                  {"params": [p_ for n_, p_ in learn if n_.startswith("book.")], "weight_decay": 0.0}]
        groups = [g for g in groups if g["params"]]
    else:
        groups = [p_ for n_, p_ in learn]
    return T.optim.AdamW(groups, lr=lr, weight_decay=wd, betas=(.9, .98))


@T.no_grad()
def zero_grad_report(m, x):
    """After an update, before zero_grad: which parameters got no gradient at all, and how many embedding rows present in the batch got none
    (9 Nov audit: the check that would have caught the decayed embeddings in E17 Phase 1)."""
    none = [n_ for n_, p_ in m.named_parameters() if p_.requires_grad and (p_.grad is None or not bool(p_.grad.any()))]
    rows = sorted(set(int(v) for v in x.unique().tolist()))
    g = m.book.weight.grad
    dead = [r for r in rows if g is None or not bool(g[r].any())] if m.book.weight.requires_grad else rows
    by_role = {}
    for r in rows:      # 10 Nov audit: the census by symbol role (an active tensor does not show that the rows it needs are learning)
        e = by_role.setdefault(byte_role(r), dict(present=0, with_gradient=0)); e["present"] += 1; e["with_gradient"] += r not in dead
    return dict(params_without_gradient=none, book_rows_in_batch=len(rows), book_rows_in_batch_without_gradient=len(dead),
                dead_rows=[chr(r) if 32 <= r < 127 else r for r in dead], by_role=by_role)


def byte_role(r):
    if r >= 256:
        return "BOS/EOS"
    c = chr(r)
    return ("digit" if c.isdigit() else "equals" if c == "=" else "operator" if c in "+-<>" else "task code" if c.islower()
            else "distractor letter" if c.isupper() else "separator" if c == ";" else "other")


def book_shift(m, book0):
    """Mean distance of each role's embedding rows from their initial values, relative to the mean initial row norm."""
    d = (m.book.weight.detach() - book0).norm(dim=1); n0 = float(book0[:256].norm(dim=1).mean())
    out = {}
    for r in range(32, 127):
        out.setdefault(byte_role(r), []).append(float(d[r]))
    return {k: round(sum(v) / len(v) / n0, 4) for k, v in out.items() if k != "other"}


BIRTH_DEFAULTS = dict(heads="top", birthrule="D", feedback="sign", traceup="none", projfb="raw", fixedcodes="off", loss="answer", farhead=0, head1loss="answer")


def expected_birth():
    """What the birth this life asks for must have been, from the life's own settings (audit of E22, 10 Nov): a matching hash shows which
    file was loaded, not how it was trained."""
    if ARCH16 == "window":
        return dict(birth="M")
    e = dict(arch=ARCH16, bookdecay=BOOKDECAY16, heads=HEADS16, birthrule=BIRTHRULE16, fixedcodes=FIXEDCODES16, loss="answer", final_step=E.DEV_DOSE, farhead=0, head1loss="answer")
    if BIRTHRULE16 == "T":
        e.update(feedback=FEEDBACK16, traceup=TRACEUP16, projfb=PROJFB16)
    return e


def admit_birth(birth_file, sha):
    """An observed hash identifies a birth; admission needs an expected hash AND the expected training configuration. Admitted if its own
    complete receipt names this hash, or if exp45/BIRTH_MANIFEST.json does; in both cases the recorded configuration and final step must
    equal expected_birth(). A configuration mismatch always stops the life. E16_LEGACY_BIRTH=allow covers only a birth with no entry at
    all, and is recorded in the receipt."""
    exp = expected_birth()
    def check(cfg, where):
        got = {k: cfg.get(k, BIRTH_DEFAULTS.get(k)) for k in exp}
        bad = {k: (got[k], v) for k, v in exp.items() if got[k] != v}
        assert not bad, f"{birth_file.name}: the birth recorded in {where} is not the one this life asks for {bad}"
    rec = birth_file.with_suffix(".json")
    if rec.exists():
        r = json.loads(rec.read_text())
        if r.get("complete") and r.get("weights_sha256") == sha:
            # 10 Nov, audit of the E20-B amendments: a checkpoint copied with its receipt under another seed's name passed here
            assert str(r.get("seed")) == birth_file.stem.rsplit("_", 1)[1], f"{birth_file.name}: its receipt is for seed {r.get('seed')}"
            check(dict(r, final_step=r["curve"][-1]["step"]), "its own receipt"); return "own receipt"
    man = OUT / "BIRTH_MANIFEST.json"
    ent = json.loads(man.read_text()).get(birth_file.name, {}) if man.exists() else {}
    if ent.get("sha256") == sha:
        check(ent.get("config", {}), "BIRTH_MANIFEST.json"); return "manifest"
    assert os.environ.get("E16_LEGACY_BIRTH") == "allow", f"{birth_file.name} ({sha[:12]}) is neither bound by its own receipt nor in BIRTH_MANIFEST.json"
    return "legacy, allowed by E16_LEGACY_BIRTH"


def sha_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_weights(m, path):
    """Weights first, by rename, then their hash: a receipt is complete only when it names the hash of weights that exist (10 Nov audit)."""
    tmp = Path(str(path) + ".tmp"); T.save(m.state_dict(), tmp); os.replace(tmp, path)
    return sha_file(path)


# ---------------------------------------------------------------- E21 (10 Nov): checkpoints at every scored stage, resume by replay
#   E16_RESUME=1   continue a life or birth from its last checkpoint: the data pipeline is replayed with the learner switched off (every batch
#                  built, every admission made, the score-time store written; no update, no evaluation) up to the checkpoint, its digest is
#                  compared with the saved one, then the weights and optimizer are loaded and the run continues as an ordinary one. Exact because
#                  nothing in the data pipeline reads the weights. Without it a run behaves as before, apart from writing <receipt stem>.ckpt
#                  (and .ckpt.prev, .ckpt.log) beside its receipt. Not a setting of the run: excluded from the configuration digest.
RESUME16 = os.environ.get("E16_RESUME", "0"); assert RESUME16 in ("0", "1")
CKPT_MAGIC = b"E21CKPT1"


class Counted:
    """A stream that counts what it has yielded (E21: part of the pipeline digest). Yields exactly what the wrapped generator yields."""
    def __init__(self, g):
        self.g, self.n = g, 0

    def __iter__(self):
        return self

    def __next__(self):
        self.n += 1
        return next(self.g)


def stream_state(s):
    """Yield count and the generator's own state: its Random, its shuffled order and its place in it (entry_stream, E.stream, joint_stream)."""
    if s is None:
        return None
    n = s.n if isinstance(s, Counted) else None
    g = s.g if isinstance(s, Counted) else s
    f = g.gi_frame
    if f is None:
        return [n, "not running"]
    loc = f.f_locals
    if "st" in loc:          # E.batches: the state is the inner stream's
        return [n, stream_state(loc["st"])]
    return [n, loc["rng"].getstate() if "rng" in loc else None, loc.get("order"), loc.get("i")]


def digest_parts(parts):
    """{component: sha256 of its repr}, and the overall digest of those. repr of ints, strings, bytes, tuples and lists is deterministic."""
    import hashlib
    d = {k: hashlib.sha256(repr(v).encode()).hexdigest() for k, v in parts.items()}
    return dict(overall=hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest(), parts=d)


def store_state(store):
    if store is None:
        return None
    return [store.order, [sorted(store.mem[k].items()) for k in store.order], len(store.mem)]


def store_write_only(store, batch):
    """What update16 / E.update write to the score-time store, under the same mask as the loss, without touching the weights (E21 replay)."""
    x, y, mask = X.batch_tensor(batch)
    if LOSS16 == "all":
        mask = y != E.BOS
    store.write(x, y, mask)


def run_config(**extra):
    """Every E11_* / E16_* / EXP45_* setting in force, by effective value, and any other such variable as given (E16_RESUME excluded)."""
    cfg = dict(E11_WD=E.WD, E11_REPLAY=E.REPLAY, E11_SCHEDULE=E.SCHEDULE, E11_TRACE=E.TRACE, E11_GATES=E.GATES, E11_STAGE_UPDATES=STAGE_UPDATES,
               E11_DEV_UPDATES=E.DEV_DOSE, EXP45_WINDOW=X.W, EXP45_HIDDEN=X.H, E16_REPLAY=REPLAY16, E16_CAP=RCAP, E16_STREAM=STREAM16, E16_ADMIT=ADMIT16,
               E16_LOSS=LOSS16, E16_ARCH=ARCH16, E16_STORE=STORE16, E16_FREEZE=FREEZE16, E16_FEEDBACK=FEEDBACK16, E16_TRACEUP=TRACEUP16,
               E16_HEADS=HEADS16, E16_PROJFB=PROJFB16, E16_FIXEDCODES=FIXEDCODES16, E16_OUTTAG=OUTTAG16, E16_COMPGAP=COMPGAP16,
               E16_BIRTHRULE=BIRTHRULE16, E16_BOOKDECAY=BOOKDECAY16, E16_RSHARE=RSHARE16, E16_LEGACY_BIRTH=os.environ.get("E16_LEGACY_BIRTH"),
               batch=BATCH, block=X.BLOCK, torch_threads=T.get_num_threads(),
               torch_version=T.__version__, dtype=str(T.get_default_dtype()), sources=source_hashes())      # 11 Nov audit: a checkpoint is bound to its sources and its torch
    for k, v in sorted(os.environ.items()):
        if k.startswith(("E11_", "E16_", "EXP45_")) and k not in cfg and k not in ("E16_RESUME", "E16_RESUME_UNQUALIFIED"):
            cfg[k] = v
    cfg.update(extra)
    return json.loads(json.dumps(cfg))


def ckpt_files(ck):
    return ck, Path(str(ck) + ".prev"), Path(str(ck) + ".log")


def ckpt_write(ck, obj):
    """Temporary file, then rename; the previous checkpoint is kept as .prev. The file carries the sha256 of its own body."""
    import hashlib, io
    buf = io.BytesIO(); T.save(obj, buf); body = buf.getvalue()
    _, prev, _ = ckpt_files(ck); tmp = Path(str(ck) + ".tmp")
    with open(tmp, "wb") as f:
        f.write(CKPT_MAGIC + hashlib.sha256(body).digest() + body); f.flush(); os.fsync(f.fileno())
    if ck.exists():
        os.replace(ck, prev)
    os.replace(tmp, ck)
    return len(body) + 40


def ckpt_read(path):
    import hashlib, io
    raw = Path(path).read_bytes()
    if raw[:8] != CKPT_MAGIC or len(raw) < 40:
        raise ValueError("not a checkpoint, or truncated before its header")
    if hashlib.sha256(raw[40:]).digest() != raw[8:40]:
        raise ValueError("body does not match its sha256 (truncated or damaged)")
    return T.load(io.BytesIO(raw[40:]), weights_only=True)


def ckpt_load(ck):
    """The newest readable checkpoint: (object, file used), or (None, None) if there is none at all. A damaged .ckpt falls back to .ckpt.prev,
    and says so; with every existing file damaged the run stops (it never starts over silently from a damaged checkpoint)."""
    cur, prev, _ = ckpt_files(ck)
    bad = []
    for p in (cur, prev):
        if not p.exists():
            continue
        try:
            obj = ckpt_read(p)
        except Exception as e:      # noqa: BLE001  (any failure to read is a damaged file)
            bad.append(f"{p.name}: {e}"); print(f"e21 checkpoint {p.name} is unreadable ({e})", flush=True); continue
        if bad:
            print(f"e21 falling back to {p.name}", flush=True)
        return obj, p
    if bad:
        raise SystemExit("e21: every checkpoint is damaged, refusing to start over: " + "; ".join(bad))
    return None, None


def ckpt_check_config(obj, cfg, used):
    old = json.loads(obj["config"])
    diff = {k: (old.get(k), cfg.get(k)) for k in sorted(set(old) | set(cfg)) if old.get(k) != cfg.get(k)}
    if diff:
        raise SystemExit(f"e21: {used.name} was written under a different configuration; changed: "
                         + ", ".join(f"{k} (checkpoint {a!r}, now {b!r})" for k, (a, b) in diff.items()))


def ckpt_check_digest(saved, now, used, what):
    if saved["overall"] != now["overall"]:
        bad = [k for k in sorted(set(saved["parts"]) | set(now["parts"])) if saved["parts"].get(k) != now["parts"].get(k)]
        raise SystemExit(f"e21: the replayed data pipeline does not match {used.name} at {what}; differing components: {bad}")


def ckpt_log(ck, **row):
    with open(ckpt_files(ck)[2], "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def ckpt_fresh(ck):
    """A run that does not resume starts its checkpoint files afresh (a stale checkpoint of an earlier run of the same name is removed)."""
    for p in ckpt_files(ck):
        if p.exists():
            p.unlink()


RESUME_SOURCES = ("exp16_integrated.py", "arch17.py", "exp11_cell.py", "exp45.py", "exp3_state.py")
RESUME_GATE = [None]


def source_hashes():
    here = Path(__file__).parent
    return {n: sha_file(here / n) for n in RESUME_SOURCES}


def resume_gate():
    """Codex's audit of E21 (11 Nov): the acceptance checks were passed on the worker's branch, not on the merged sources. E16_RESUME=1 is refused
    unless exp45/RESUME_QUALIFIED.json names exactly these source hashes. E16_RESUME_UNQUALIFIED=allow is for the qualification run itself; it is
    written into the receipt's `resumed` entries and is not a setting of the run."""
    q = OUT / "RESUME_QUALIFIED.json"
    if q.exists() and json.loads(q.read_text()).get("sources") == source_hashes():
        RESUME_GATE[0] = "qualified"
    elif os.environ.get("E16_RESUME_UNQUALIFIED") == "allow":
        RESUME_GATE[0] = "unqualified, allowed by E16_RESUME_UNQUALIFIED"
    else:
        raise SystemExit("e21: E16_RESUME=1 is refused: resume has not been qualified on these sources (exp45/RESUME_QUALIFIED.json is absent or names other sources)")


def resume_record(from_stage, used, cpu_replay, note=None):
    import datetime, socket
    r = dict(from_stage=from_stage, checkpoint=(used.name if used else None), host=socket.gethostname(),
             time=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), cpu_replay=cpu_replay)
    if note:
        r["note"] = note
    if RESUME_GATE[0]:
        r["qualification"] = RESUME_GATE[0]
    return r
assert STREAM16 in ("blocks", "drift") and ADMIT16 in ("block", "online") and LOSS16 in ("answer", "all")
assert ADMIT16 == "block" or REPLAY16 == "raw", "online admission is implemented for the raw reservoir only"
#   E16_RSHARE=<fraction>  (E16-M3, 9 Nov 21:20 EST) the share of every batch drawn from the raw/total reservoir once it holds anything;
#                           default 0.5 reproduces every receipt made before. Other values tag the receipt `_rs<percent>`. Raw/total replay only.
RSHARE16 = float(os.environ.get("E16_RSHARE", "0.5")); assert 0.0 <= RSHARE16 <= 1.0
assert RSHARE16 == 0.5 or REPLAY16 in ("raw", "total"), "E16_RSHARE is defined for the raw/total reservoirs only"
NREP = round(BATCH * RSHARE16)
assert RSHARE16 == 0.5 or NREP == round(BATCH * round(100 * RSHARE16) / 100), "this replay share and another give the same _rs tag but different replay rows per batch (10 Nov audit); use a share that is a whole number of rows"
RSTAG = "" if RSHARE16 == 0.5 else f"_rs{round(100 * RSHARE16)}"
RTAG += ("_drift" if STREAM16 == "drift" else "") + ("_online" if ADMIT16 == "online" else "") + ("_lossall" if LOSS16 == "all" else "") + ("_nostore" if STORE16 == "off" and ARCH16 == "window" else "") + RSTAG   # off is the default for the windowless architectures
F1 = E.LIFE                                                                        # mod 97: r t x n
F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]   # mod 73 (E8's second four)
SYM_OP = {v: k for k, v in E.SYM.items()}
PAIRS = [(a, b) for a in E.SYM.values() for b in E.SYM.values()]
HELD_PAIRS = [("+", "-"), ("-", "+"), (">", "<"), ("<", ">")]
TAUGHT_PAIRS = [p for p in PAIRS if p not in HELD_PAIRS]
KMAX, KS = 12, E.KS_PANEL
CAP = 6   # longest answer: i i ; f f EOS


# ---------------------------------------------------------------- items: an entry is replayable; render makes the bytes
def render_pair(o1, o2, a, b, c, rng=None, inside=None, lead=b""):
    """inside = (letters after the first operator, letters after the second); lead = letters before the prompt. With rng and E16_COMPGAP the
    placement is drawn: total length g1 + g2, inside or leading with probability one half each."""
    i = E.apply(SYM_OP[o1], 97, a, b); f = E.apply(SYM_OP[o2], 97, i, c)
    if inside is None:
        inside = (b"", b"")
        if COMPGAP16 and rng is not None:
            g = [bytes(rng.choice(E.DISTRACT) for _ in range(rng.randint(0, COMPGAP16))) for _ in range(2)]
            inside, lead = ((g[0], g[1]), b"") if rng.random() < .5 else ((b"", b""), g[0] + g[1])
    prompt = list(lead + b"c" + f"{a:02d}".encode() + o1.encode() + inside[0] + f"{b:02d}".encode() + o2.encode() + inside[1] + f"{c:02d}".encode() + b"=")
    return prompt, list(f"{i:02d}".encode()) + [ord(";")] + list(f"{f:02d}".encode()) + [EOS]


def render(entry, rng):
    if entry[0] == "raw":          # verbatim replay of presented bytes
        return list(entry[1]), list(entry[2])
    if entry[0] == "op":
        _, task, a, b = entry
        return E.render(task, a, b, rng.randint(0, KMAX), rng)
    _, o1, o2, a, b, c = entry
    return render_pair(o1, o2, a, b, c, rng)


def triples(pair, seed, n, exclude=frozenset(), split="taught"):
    """Taught triples keep the original seed (training streams unchanged). Held and within panels are keyed by split name and pair
    (audit of 8 Nov 2026: `seed + 1` for the held +- pair collided with the taught -> pair's seed, so those 600 operand triples were shared;
    the pairing and prompts were still held out). Pass exclude = every taught triple so the panels are operand-disjoint from all teaching."""
    rng = Random(seed * 31 + ord(pair[0]) * 7 + ord(pair[1])) if split == "taught" else Random(f"{split}|{seed}|{pair[0]}{pair[1]}")
    out, seen = [], set(exclude)
    while len(out) < n:
        tr = (rng.randrange(97), rng.randrange(97), rng.randrange(97))
        if tr not in seen:
            seen.add(tr); out.append(tr)
    return out


def family_entries(fam, seed):
    if fam == "F2":
        return [("comp", o1, o2, a, b, c) for o1, o2 in TAUGHT_PAIRS for a, b, c in triples((o1, o2), seed, 2000)]
    return [("op", t, a, b) for t in (F1 if fam == "F1" else F3) for a, b in E.split(t, .25, seed)[0]]


def entry_stream(entries, seed):
    rng = Random(seed + 77)
    while True:
        order = list(range(len(entries)))
        rng.shuffle(order)
        for i in order:
            yield render(entries[i], rng)


def joint_stream(streams, seed):
    """Arm D: each item from a family chosen uniformly among those seen so far (ground truth, equal weight per family)."""
    rng = Random(seed + 78)
    while True:
        yield next(streams[rng.randrange(len(streams))])


def update16(m, opt, items, rule, store):
    """E.update, with the answer mask replaced by every-byte prediction when E16_LOSS=all (default path calls E.update unchanged)."""
    if LOSS16 == "answer" and HEADS16 == "top":
        LAST_AUX[0] = None
        return E.update(m, opt, items, rule, store)
    x, y, mask = X.batch_tensor(items)
    amask = mask             # the answer positions
    if LOSS16 == "all":
        mask = y != E.BOS        # every real byte of prompt and answer; padding (BOS) excluded
    logits, it = E.forward(m, x, local=(rule != "D"), through_time=(rule == "D"), keep=(rule == "T"))
    if LOSSW16 == "balanced":    # E24: half the loss on the answer positions, half on all the others
        other = mask & ~amask
        loss = .5 * E.cross_entropy(logits[amask], y[amask]) + .5 * E.cross_entropy(logits[other], y[other])
    else:
        loss = E.cross_entropy(logits[mask], y[mask])
    m1 = (y != E.BOS) if HEAD1LOSS16 == "all" else mask      # E23: the first bank's head on every next byte
    aux = E.cross_entropy(it["logits1"][m1], y[m1]) if HEADS16 != "top" else None     # E18: the first bank's own error
    LAST_AUX[0] = float(aux.detach()) if aux is not None else None
    total = loss + aux if aux is not None else loss
    LAST_FAR[0] = None
    if FARHEAD16:      # E23: the byte FARHEAD16 steps ahead of each input byte, y[:, t + d - 1]; positions past the item's end excluded
        yf, mf = far_targets(y, mask, FARHEAD16)
        if FARCTRL16 == "shuffle":
            yf = yf.clone()
            for i in range(yf.shape[0]):
                idx = mf[i].nonzero().squeeze(1)
                yf[i, idx] = yf[i, idx[T.randperm(len(idx), generator=FAR_GEN[0])]]
        far = E.cross_entropy(it["logits_far"][mf], yf[mf])
        LAST_FAR[0] = float(far.detach()); total = total + far
    opt.zero_grad(); total.backward()
    if rule == "T":
        (E.trace_grads_scan if E.TRACE == "scan" else E.trace_grads)(m, it)
    opt.step()
    if store is not None:
        store.write(x, y, mask)
    return float(loss.detach())


def drift_weights(u, total):
    """Family weights at update u of total: raised-cosine bumps centred at 1/6, 1/2, 5/6 of the life, half-width 1/2
    (F1 alone at the start, F1 and F2 equal at a third, F2 alone at the middle, F2 and F3 equal at two thirds, F3 alone at the end)."""
    import math
    f = u / total
    return {fam: max(0.0, math.cos(math.pi * (f - c) / 0.5)) if abs(f - c) < 0.5 else 0.0 for fam, c in (("F1", 1 / 6), ("F2", 1 / 2), ("F3", 5 / 6))}


# ---------------------------------------------------------------- panels
def score(m, items, store=None, silence=False):
    outs = E.respond(m, [p for p, _ in items], store, cap=CAP, silence=silence)
    return dict(n=len(items), exact=sum(o == t for o, (_, t) in zip(outs, items)),
                final=sum(o[-3:] == t[-3:] for o, (_, t) in zip(outs, items)))


FAR_POS_D = 4      # E23: the far head is trained at the SAME input positions in every arm: those whose byte FAR_POS_D ahead is a prompt byte


def far_targets(y, mask, d):
    """E23 (v3, after the second internal audit): target = the byte d steps ahead of input position t (y[:, t + d - 1]); trained positions =
    those at which the byte FAR_POS_D ahead is still a prompt byte (not an answer digit, not EOS, not padding), identical for d = 1 and d = 4."""
    D = FAR_POS_D; assert 1 <= d <= D <= y.shape[1]
    shift = lambda z, k, fill: T.cat([z[:, k - 1:], T.full_like(z[:, :k - 1], fill)], 1)
    pos = (shift(y, D, E.BOS) != E.BOS) & ~shift(mask, D, False)
    return shift(y, d, E.BOS), pos


@T.no_grad()
def far_accuracy(m, items):
    """E23 diagnostic, never a whole-answer score: the far head's top-1 on its own (unshuffled) prompt-byte targets, teacher forced, with a
    structural baseline (on the same items, the most frequent target given the class of the input byte and the class of the target's
    position: an upper bound on what structure alone gives). Also the stratum whose target is a digit."""
    x, y, mask = X.batch_tensor(items)
    _, it = E.forward(m, x, local=True, through_time=False)
    yf, mf = far_targets(y, mask, FARHEAD16)
    hit = (it["logits_far"].argmax(-1) == yf) & mf
    dig = mf & (yf >= 48) & (yf <= 57)
    cls = lambda b: T.where(b == E.BOS, 5, T.where((b >= 48) & (b <= 57), 0, T.where((b >= 65) & (b <= 90), 1, T.where(b >= 97, 2, T.where(b == 61, 3, 4)))))
    key = cls(x) * 8 + cls(yf)        # input class x target class (the target class is not the target byte)
    base = 0
    for k_ in key[mf].unique():
        sel = mf & (key == k_); base += int(T.bincount(yf[sel]).max())
    return dict(prompt=[int(hit.sum()), int(mf.sum())], digit=[int((hit & dig).sum()), int(dig.sum())], structural_baseline=[base, int(mf.sum())])


def head1(fn, *a):
    """E18: a score taken from the first bank's own head (what the first bank alone can answer)."""
    arch17.HEAD1_OUT = True
    try:
        return fn(*a)
    finally:
        arch17.HEAD1_OUT = False


def panels(seed):
    ks = E.KS_PANEL + ((40,) if ARCH16 != "window" else ())                                   # E17 adds k = 40, beyond any window
    P = {E.NAME(*t): E.panel(t, E.split(t, .25, seed)[1], seed, ks=ks) for t in F1 + F3}   # held pairs, k = 0, 8, 12, 20 (, 40)
    P.update({E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(12,), n=300) for t in E.DEV})
    taught = {p: set(triples(p, seed, 2000)) for p in TAUGHT_PAIRS}
    all_taught = frozenset().union(*taught.values())
    comp = {f"{o1}{o2}": [render_pair(o1, o2, *tr) for tr in triples((o1, o2), seed, 600, exclude=all_taught, split="held")] for o1, o2 in HELD_PAIRS}
    comp["within"] = [render_pair(*p, *tr) for p in TAUGHT_PAIRS for tr in triples(p, seed, 50, exclude=all_taught, split="within")]
    MEMORY[0] = memory_panels(seed, all_taught) if COMPGAP16 else None
    return P, comp


def exposure_sources(seed):
    """Every route by which a single-operation relation is supervised somewhere in a life (second audit of 10 Nov): standalone teaching in
    each family and at birth, and both steps of every taught composition. Keys are operation names; values are sets of ordered pairs."""
    src = {"standalone": {}, "substep": {}, "other_modulus": {}}
    for t in F1 + F3:
        src["standalone"][E.NAME(*t)] = set(E.split(t, .25, seed)[0])
    for t in E.DEV:
        src["standalone"][E.NAME(*t)] = set(E.split(t, .5, seed)[0])
    for o1, o2 in TAUGHT_PAIRS:
        for a, b, c in triples((o1, o2), seed, 2000):
            src["substep"].setdefault(SYM_OP[o1], set()).add((a, b)); src["substep"].setdefault(SYM_OP[o2], set()).add((E.apply(SYM_OP[o1], 97, a, b), c))
    for t in F1 + F3 + list(E.DEV):      # max and min give the same answer under every modulus that contains both operands
        if t[0] in ("max", "min"):
            src["other_modulus"][E.NAME(*t)] = set().union(*[src["standalone"][E.NAME(*u)] for u in F1 + F3 + list(E.DEV) if u[0] == t[0] and u[1] != t[1]])
    return src


def exposure_flags(t, a, b, src):
    """Which equivalent supervision exists for held pair (a, b) of task t. Commutative operations are checked in both orientations."""
    comm = t[0] in ("add", "max", "min"); name = E.NAME(*t); f = []
    if comm and a != b and (b, a) in src["standalone"][name]:
        f.append("reversal taught")
    sub = src["substep"].get(t[0], ()) if t[1] == 97 else ()
    if (a, b) in sub or (comm and (b, a) in sub):
        f.append("step of a taught composition")
    om = src["other_modulus"].get(name, ())
    if (a, b) in om or (b, a) in om:
        f.append("taught under another modulus")
    return f


def strict_panels(seed):
    """Held single-operation pairs with no equivalent supervision by any route in exposure_sources (corrected after the second audit of
    10 Nov: the first version checked composition steps in one orientation only and ignored max/min under other moduli, which left most
    equivalent supervision in). Scored at k = 12 beside the registered panels; the registered panels and the split are unchanged.
    Small by construction; the count is recorded."""
    src = exposure_sources(seed); out, counts = {}, {}
    for t in F1 + F3:
        held = E.split(t, .25, seed)[1]
        ok = [(a, b) for a, b in held if not exposure_flags(t, a, b, src)]
        counts[E.NAME(*t)] = dict(held=len(held), strict=len(ok))
        out[E.NAME(*t)] = E.panel(t, ok, seed + 11, ks=(12,), n=min(400, len(ok)))[12]
    return out, counts


def memory_panels(seed, all_taught, n=150):
    """E20 assay. For each held operator pair, n twins: two prompts with the same operators, second and third operands, and the same letters
    in the same places, differing only in the first operand, which is outside the last ten bytes at "=". The two worked answers differ in
    both the intermediate and the final result. A predictor that sees only the last ten bytes gives the same answer to both, so it can get
    at most one of a twin and never both. `lead` re-renders the same items with the same letters moved in front: same length, every operand
    in view. Also, with the first operand hidden: new instances of taught operator pairs, and taught triples themselves. Checked here,
    fail-closed, every time the panels are built."""
    g = Random(f"memory|{seed}|{COMPGAP16}"); out = dict(twins={}, lead={}, census={})
    def letters():
        while True:
            a, b = (bytes(g.choice(E.DISTRACT) for _ in range(g.randint(0, COMPGAP16))) for _ in range(2))
            if len(a) + len(b) >= 3:
                return a, b
    for o1, o2 in HELD_PAIRS:
        tw, ld = [], []
        while len(tw) < n:
            b_, c_, a1, a2 = (g.randrange(97) for _ in range(4)); ins = letters()
            x, y = render_pair(o1, o2, a1, b_, c_, inside=ins), render_pair(o1, o2, a2, b_, c_, inside=ins)
            if a1 == a2 or x[1][:2] == y[1][:2] or x[1][3:5] == y[1][3:5] or (a1, b_, c_) in all_taught or (a2, b_, c_) in all_taught:
                continue
            assert x[0][-10:] == y[0][-10:] and x[0] != y[0] and x[1] != y[1], "twins must share the last ten bytes and differ in the answer"
            tw.append((x, y)); ld.append((render_pair(o1, o2, a1, b_, c_, lead=ins[0] + ins[1]), render_pair(o1, o2, a2, b_, c_, lead=ins[0] + ins[1])))
            assert len(ld[-1][0][0]) == len(x[0]) and bytes(ld[-1][0][0][-10:]).startswith(b"c"), "the control must have the same length with the whole prompt in view"
        out["twins"][f"{o1}{o2}"], out["lead"][f"{o1}{o2}"] = tw, ld
        out["census"][f"{o1}{o2}"] = dict(twins=n, second_operand_hidden=sum(len(x[0]) - 10 - (len(x[0]) - x[0].index(ord(o2), 4) - 1 - 3) > 5 for x, _ in tw),
                                           mean_prompt_bytes=round(sum(len(x[0]) for x, _ in tw) / n, 2))
    def hidden(pairs, split, m_):
        res = []
        for p_ in pairs:
            for tr in (triples(p_, seed, 2000)[:m_] if split == "taught" else triples(p_, seed, m_, exclude=all_taught, split="memory-within")):
                res.append(render_pair(*p_, *tr, inside=letters()))
        return res
    out["within_hidden"] = hidden(TAUGHT_PAIRS, "within", 25)       # new operand triples of taught operator pairs
    def gaps(p_, o2_):
        j = p_.index(ord(o2_), 4)
        return j - 6, len(p_) - j - 4          # letters after the first operator, letters after the second
    CENSUS_V2[0] = {pair: dict(twins=len(tw), second_operand_fully_out_of_view=sum(gaps(x[0], pair[1])[1] >= 6 for x, _ in tw),
                               second_operand_partly_in_view=sum(gaps(x[0], pair[1])[1] == 5 for x, _ in tw),
                               note="replaces census.second_operand_hidden, which counted the letters after the first operator (third audit, 10 Nov); not part of the panel hash")
                    for pair, tw in out["twins"].items()}
    out["taught_hidden"] = hidden(TAUGHT_PAIRS, "taught", 25)       # triples that are taught (letters are fresh)
    # the assay must reject a last-ten-bytes predictor and accept the true answers
    table = {}
    both = 0
    for tw in out["twins"].values():
        for x, y in tw:
            guess = table.setdefault(bytes(x[0][-10:]), x[1]); both += (guess == x[1]) and (guess == y[1])
    assert both == 0, "a suffix-only predictor must score zero both-correct twins"
    return out


def score_memory(m, mem, silence=False):
    """Whole answers, no store, for every arm alike. Twins: both members correct."""
    res = {}
    for key in (("twins",) if silence else ("twins", "lead")):
        res[key] = {}
        for pair, tw in mem[key].items():
            flat = [it for xy in tw for it in xy]
            outs = E.respond(m, [p for p, _ in flat], None, cap=CAP, silence=silence)
            ok = [o == t for o, (_, t) in zip(outs, flat)]
            res[key][pair] = dict(pairs=len(tw), items_correct=sum(ok), both_correct=sum(ok[2 * i] and ok[2 * i + 1] for i in range(len(tw))))
    if not silence:
        for key in ("within_hidden", "taught_hidden"):
            res[key] = score(m, mem[key], None)
    return res


def panel_sha(P, comp, strict):
    import hashlib
    return hashlib.sha256(json.dumps([P, comp, strict], sort_keys=True, default=list).encode()).hexdigest()


def shortcut_ceilings(comp):
    """Final answer from a visible-only rule: ignore the first operator (f = o2(a, c), f = o2(b, c)) or the second (f = o1(a, b))."""
    out = {}
    for name, items in comp.items():
        if name == "within":
            continue
        o1, o2 = name
        hits = {"ignore_o1_use_a": 0, "ignore_o1_use_b": 0, "ignore_o2": 0, "answer_is_c": 0}
        for p, t in items:
            a, b, c = int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9])); f = int(bytes(t[3:5]))
            hits["ignore_o1_use_a"] += E.apply(SYM_OP[o2], 97, a, c) == f
            hits["ignore_o1_use_b"] += E.apply(SYM_OP[o2], 97, b, c) == f
            hits["ignore_o2"] += E.apply(SYM_OP[o1], 97, a, b) == f
            hits["answer_is_c"] += c == f
        out[name] = {k: v / len(items) for k, v in hits.items()}
    return out


# ---------------------------------------------------------------- life
SCHEDULE = [("F1", 1), ("F1", 2), ("F2", 1), ("F2", 2), ("F3", 1), ("F3", 2)]


def life(arm, seed):
    assert FARHEAD16 == 0 and HEAD1LOSS16 == "answer", "E16_FARHEAD and E16_HEAD1LOSS are defined for births only (E23)"
    assert LOSSW16 == "uniform", "E16_LOSSW=balanced is defined for births only (E24)"
    DTAG = ("_drift" if STREAM16 == "drift" else "") + ("_lossall" if LOSS16 == "all" else "")   # arm D carries the stream/loss tags, not the replay tag
    path = OUT / f"{PFX}_life_{arm}{ATAG}{RTAG if arm != 'D' else DTAG}{CGTAG}{OUTTAG16}{DRYTAG}_{seed}.json"
    assert COMPGAP16 == 0 or arm == "D" or REPLAY16 in ("raw", "oracle"), "E16_COMPGAP needs raw or oracle replay (other stores parse fixed positions)"
    m = E.new_net(seed)
    birth_file = OUT / (f"e11_dev_M_{seed}.pt" if ARCH16 == "window" else f"e17_dev_{ARCH16}{BTAG}{HTAG}{DEVRULETAG}{E.DEVTAG}_{seed}.pt")
    import hashlib
    birth_sha = hashlib.sha256(birth_file.read_bytes()).hexdigest()
    birth_admission = admit_birth(birth_file, birth_sha)
    m.load_state_dict(T.load(birth_file))
    assert STREAM16 == "blocks" or arm == "D" or REPLAY16 in ("raw", "total", "oracle"), "the drifting stream is implemented for the raw/total reservoirs and the twin only"
    local = arm not in ("D", "M")
    admits = local or arm == "M"       # arm M keeps T's whole data pipeline, including online admission (9 Nov: before this line M admitted at block ends only)
    FROZEN = ((("book.",) if BOOKDECAY16 == "off" else ()) + ("l1.", "lp.", "lc.", "lw.")) if FREEZE16 == "bank1" else ()
    if FREEZE16 == "book" or FIXEDCODES16 == "on": FROZEN = ("book.",)
    opt = adamw16(m, .001, (E.WD if arm != "D" else .3), FROZEN)
    store = X.Store(seed) if (local and STORE16 == "on") else None
    rule = "D" if arm == "M" else arm      # arm M: T's data pipeline, D's update
    rng = Random(seed + 9)
    P, comp = panels(seed)
    SP, strict_counts = strict_panels(seed)
    f1c, f3c = {t[2][0] for t in F1}, {t[2][0] for t in F3}
    code_of = lambda p_: next((b_ for b_ in p_ if not 65 <= b_ <= 90), p_[0])      # the task code is the first byte that is not a distractor letter (third audit, 10 Nov)
    fam_of = lambda p_: "F2" if code_of(p_) == ord("c") else "F1" if code_of(p_) in f1c else "F3" if code_of(p_) in f3c else "other"
    entries = {f: family_entries(f, seed) for f in ("F1", "F2", "F3")}
    buffer, seen_streams = [], []
    fbuf, fset, fseen = {}, {}, {}      # balanced modes: per-family entry lists (sets for all_balanced's distinct entries; counters for reservoir)
    t0 = time.process_time(); stages = []
    # E21: checkpoint / resume. K is the stage the checkpoint was written after (0: start from the birth).
    ck = path.with_suffix(".ckpt"); K, ckobj, ckused, resumed, saved = 0, None, None, [], None
    if RESUME16 == "1":
        resume_gate()
        ckobj, ckused = ckpt_load(ck)
        if ckobj is None:
            print(f"e21 no checkpoint for {path.name}: starting from the birth", flush=True)
            resumed.append(resume_record(0, None, 0.0, note="no checkpoint: started from the birth"))
    else:
        ckpt_fresh(ck)

    def receipt(**extra):
        d = dict(arm=arm, seed=seed, schedule=SCHEDULE, stages=stages)
        if resumed:
            d["resumed"] = resumed
        d.update(extra)
        return d

    def evaluate():
        res = {n: {k: score(m, it, store) for k, it in pk.items()} for n, pk in P.items()}
        res["compose"] = {n: score(m, it, store) for n, it in comp.items()}
        res["silenced_k12"] = {n: score(m, pk[12], store, silence=True) for n, pk in P.items() if not n.endswith("89")}
        res["silenced_compose"] = {n: score(m, it, store, silence=True) for n, it in comp.items() if n != "within"}
        if HEADS16 != "top":
            res["head1_k12"] = {n: head1(score, m, pk[12], store) for n, pk in P.items()}
        if MEMORY[0] is not None:
            res["memory"] = score_memory(m, MEMORY[0]); res["memory_silenced"] = score_memory(m, MEMORY[0], silence=True)
        res["strict_k12"] = {n: score(m, it, store) for n, it in SP.items()}          # 10 Nov audit: no reversal taught, no taught sub-step
        if store is not None:       # 10 Nov audit: what the network answers without the score-time store, so arms with and without one can be compared
            res["nostore_k12"] = {n: score(m, pk[12], None) for n, pk in P.items()}
            res["nostore_compose"] = {n: score(m, it, None) for n, it in comp.items()}
        return res

    cfg21 = run_config(arm=arm, seed=seed, receipt=path.name, birth=birth_file.name, birth_sha256=birth_sha, panel_sha256=panel_sha(P, comp, SP),
                       memory_sha256=(__import__("hashlib").sha256(json.dumps(MEMORY[0], sort_keys=True, default=list).encode()).hexdigest() if MEMORY[0] else None))
    if ckobj is not None:
        ckpt_check_config(ckobj, cfg21, ckused)
        K, saved = int(ckobj["stage"]), json.loads(ckobj["stages"]); resumed = json.loads(ckobj["resumed"])
        assert len(saved) == K + 1, f"{ckused.name}: {len(saved)} stage records for stage {K}"
        print(f"e21 resuming {path.name} from stage {K} ({ckused.name}): replaying the data pipeline with the learner off", flush=True)
    if K > 0:      # stage 0's record (scores of the birth, settings, first-update census) is the saved one
        stages.append(json.loads(json.dumps(saved[0])))
        cpu_score_total = [0.0]      # merge of E21 with the 10 Nov accounting: on a resume the scoring total is read from the checkpoint at the resume point
    else:
        c_e0 = time.process_time(); scores0 = evaluate(); cpu_score_total = [time.process_time() - c_e0]      # the scoring before the first block is scoring, not updating (third audit, 10 Nov)
        stages.append(dict(stage=0, task=None, scores=scores0, shortcut_ceilings=shortcut_ceilings(comp),
                       entries={f: len(v) for f, v in entries.items()}, settings=dict(wd=opt.defaults["weight_decay"], replay=E.REPLAY, trace=E.TRACE, stage_updates=STAGE_UPDATES, replay16=REPLAY16, cap=(RCAP if REPLAY16 in ("reservoir", "total", "raw") else None), stream=STREAM16, admit=ADMIT16, loss=LOSS16, arch=ARCH16, freeze=FREEZE16, frozen=list(FROZEN), bookdecay=BOOKDECAY16, feedback=FEEDBACK16, traceup=TRACEUP16, rshare=RSHARE16, nrep=NREP, batch=BATCH, heads=HEADS16, birthrule=BIRTHRULE16, projfb=PROJFB16, fixedcodes=FIXEDCODES16, outtag=OUTTAG16, compgap=COMPGAP16,
                                     memory_panels=(dict(design="v2 twins/lead, letters inside or leading 50/50 in teaching", census=MEMORY[0]["census"], census_v2=CENSUS_V2[0],
                                                         sha256=__import__("hashlib").sha256(json.dumps(MEMORY[0], sort_keys=True, default=list).encode()).hexdigest()) if MEMORY[0] else None),
                                     panel_sha256=panel_sha(P, comp, SP), strict_panel_counts=strict_counts, birth=birth_file.name, birth_sha256=birth_sha, birth_admission=birth_admission, panels="held/within keyed by split, operand-disjoint from all taught triples (8 Nov audit)",
                                     arm=arm, rule=rule, store=store is not None)))   # E16-V: settings.wd is now the optimizer's actual decay (arm D runs at .3)))
    fstreams = {f: Counted(entry_stream(entries[f], seed + 200 + i)) for i, f in enumerate(("F1", "F2", "F3"))} if STREAM16 == "drift" else None
    drng = Random(seed + 300); TOTAL = STAGE_UPDATES * len(SCHEDULE); nseen = {}
    def drift_item(u):
        w = drift_weights(u, TOTAL); fams = [f for f in ("F1", "F2", "F3") if w[f] > 0] or ["F3"]
        r = drng.random() * sum(w[f] for f in fams); acc = 0.0
        for f in fams:
            acc += w[f]
            if r <= acc:
                return next(fstreams[f])
        return next(fstreams[fams[-1]])
    joint = None

    def pipeline_digest(presented, expo):      # E21: everything the data pipeline carries, at a stage's end (after its admissions)
        return digest_parts(dict(rng=rng.getstate(), drng=drng.getstate(), cur=stream_state(cur), joint=stream_state(joint),
                                 seen_streams=[stream_state(s_) for s_ in seen_streams], fstreams=({f: stream_state(s_) for f, s_ in fstreams.items()} if fstreams else None),
                                 buffer=buffer, fbuf=fbuf, fset={f: sorted(map(repr, v)) for f, v in fset.items()}, fseen=fseen, nseen=nseen,
                                 presented=presented, exposure=expo, store=store_state(store)))
    for si, (fam, block) in enumerate(SCHEDULE, 1):
        replaying = si <= K     # E21: the learner is off; batches, admissions, store writes and bookkeeping run exactly as written
        cur = Counted(entry_stream(entries[fam], seed + si))
        if arm == "D" or REPLAY16 == "oracle":
            if STREAM16 == "drift":
                for f in ("F1", "F2", "F3"):     # the twin's joint stream admits a family once its weight is positive at this checkpoint's start
                    if drift_weights((si - 1) * STAGE_UPDATES, TOTAL)[f] > 0 and f not in nseen:
                        nseen[f] = True; seen_streams.append(Counted(entry_stream(entries[f], seed + 100 + len(nseen))))
            elif block == 1:
                seen_streams.append(Counted(entry_stream(entries[fam], seed + 100 + si)))
            joint = Counted(joint_stream(seen_streams, seed + si))
        losses, presented, expo, aux_losses = [], [], {}, []
        for u in range(STAGE_UPDATES):
            npres_before = len(presented)
            if arm == "D" or REPLAY16 == "oracle":
                batch = [next(joint) for _ in range(BATCH)]
            elif fbuf and REPLAY16 == "share" and sorted(fbuf) != [fam]:
                fams = [f for f in sorted(fbuf) if f != fam]
                rp = [render(rng.choice(fbuf[fams[i % len(fams)]]), rng) for i in range(BATCH // 2)]
                new = [next(cur) for _ in range(BATCH // 2)]
                batch = new + rp; presented += new
            elif REPLAY16 in ("total", "raw") and fbuf:
                rp = [render(rng.choice(fbuf["all"]), rng) for _ in range(NREP)]
                new = [(drift_item(si * STAGE_UPDATES - STAGE_UPDATES + u) if STREAM16 == "drift" else next(cur)) for _ in range(BATCH - NREP)]
                batch = new + rp; presented += new
            elif fbuf and REPLAY16 != "share":
                fams = sorted(fbuf)
                rp = [render(rng.choice(fbuf[fams[rng.randrange(len(fams))]]), rng) for _ in range(BATCH // 2)]
                new = [next(cur) for _ in range(BATCH // 2)]
                batch = new + rp; presented += new
            elif buffer:
                rp = [render(e, rng) for e in rng.sample(buffer, BATCH // 2)]   # fresh-render replay: stored entries, new letters and k
                new = [next(cur) for _ in range(BATCH // 2)]
                batch = new + rp; presented += new
            else:
                batch = [(drift_item(si * STAGE_UPDATES - STAGE_UPDATES + u) if STREAM16 == "drift" else next(cur)) for _ in range(BATCH)]; presented += batch
            nnew = len(presented) - npres_before if admits else 0
            nfresh = len(presented) - npres_before
            for i_, (p_, t_) in enumerate(batch):      # 10 Nov audit: rows and supervised tokens by origin and family (logging only)
                key = f"{'joint' if (arm == 'D' or REPLAY16 == 'oracle') else 'fresh' if i_ < nfresh else 'replay'}/{fam_of(p_)}"
                e_ = expo.setdefault(key, [0, 0]); e_[0] += 1; e_[1] += (len(t_) if LOSS16 == "answer" else len(p_) + len(t_))   # bytes under the loss mask
            if replaying:
                if store is not None:
                    store_write_only(store, batch)
            else:
                losses.append(update16(m, opt, batch, rule, store))
            if not replaying and LAST_AUX[0] is not None:
                aux_losses.append(LAST_AUX[0])
            if si == 1 and u == 0 and not replaying:    # 9 Nov audit: record what received no gradient on the first update (grads persist until the next zero_grad)
                stages[0]["settings"]["zero_grad_first_update"] = zero_grad_report(m, X.batch_tensor(batch)[0])
                print("e16 zero-grad check:", stages[0]["settings"]["zero_grad_first_update"], flush=True)
            if ADMIT16 == "online" and admits:     # raw reservoir, Algorithm R per presented item, as it arrives (new items are the first nnew of the batch)
                fl = fbuf.setdefault("all", []); n = fseen.get("all", 0)
                for p_, t_ in batch[:nnew]:
                    n += 1
                    if len(fl) < RCAP:
                        fl.append(("raw", tuple(p_), tuple(t_)))
                    else:
                        j = rng.randrange(n)
                        if j < RCAP:
                            fl[j] = ("raw", tuple(p_), tuple(t_))
                fseen["all"] = n
        if replaying:     # the stage's record is the saved one (its bookkeeping below is recomputed and compared at the resume point)
            if json.dumps(expo, sort_keys=True) != json.dumps(saved[si]["exposure_rows_tokens"], sort_keys=True):
                raise SystemExit(f"e21: the replayed exposure of stage {si} differs from {ckused.name}")
            stages.append(json.loads(json.dumps(saved[si])))
        else:
            c_before = time.process_time(); sc_now = evaluate(); c_after = time.process_time(); cpu_score_total[0] += c_after - c_before
            stages.append(dict(stage=si, task=f"{fam}.{block}", loss=sum(losses[-512:]) / min(512, len(losses)), cpu=c_before - t0, scores=sc_now,
                               cpu_scoring_so_far=cpu_score_total[0], cpu_updates_so_far=c_after - t0 - cpu_score_total[0],
                               exposure_rows_tokens=expo, exposure_note=f"[rows, bytes under the loss mask]; loss={LOSS16}; heads applying each byte: {1 if HEADS16 == 'top' else 2}"))
            if aux_losses:
                stages[-1]["loss_head1"] = sum(aux_losses[-512:]) / min(512, len(aux_losses))
            sc = stages[-1]["scores"]
            print("e16", arm, seed, "after", f"{fam}.{block}",
                  "F1@k", {n: [sc[n][k]["exact"] for k in KS] for n in sc if n.endswith("97")},
                  "F3@12", [sc[n][12]["exact"] for n in sc if n.endswith("73")],
                  "comp.final", {n: v["final"] for n, v in sc["compose"].items()},
                  "dev89@12", [sc[n][12]["exact"] for n in sc if n.endswith("89")],
                  f"cpu {stages[-1]['cpu']:.0f}s", flush=True)
        if not replaying:
            path.write_text(json.dumps(receipt(), indent=1))
        if ADMIT16 == "online" and admits:
            stages[-1]["buffer"] = {f: len(v) for f, v in fbuf.items()}; stages[-1]["reservoir_distinct"] = {f: len(set(v)) for f, v in fbuf.items()}
            stages[-1]["reservoir_bytes"] = sum(len(e[1]) + len(e[2]) for e in fbuf["all"]); stages[-1]["admitted"] = fseen.get("all", 0)
            if not replaying:
                path.write_text(json.dumps(receipt(), indent=1))
        elif (local or arm == "M") and REPLAY16 != "oracle":
            # the stage's presented items as replayable entries (family + operands), 1,024 of them, as E11-D/E15
            ents = []
            for p, t in presented:
                if REPLAY16 == "raw":
                    ents.append(("raw", tuple(p), tuple(t))); continue
                if p[0] == ord("c"):
                    ents.append(("comp", chr(p[3]), chr(p[6]), int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9]))))
                else:
                    task = next(t for t in F1 + F3 if t[2][0] == p[0])
                    ents.append(("op", task, int(bytes(p[1:3])), int(bytes(p[-3:-1]))))
            if REPLAY16 == "uniform":
                buffer += rng.sample(ents, min(1024, len(ents)))
            elif REPLAY16 in ("balanced", "share"):
                fbuf.setdefault(fam, []).extend(rng.sample(ents, min(1024, len(ents))))
            elif REPLAY16 in ("reservoir", "total", "raw"):   # Algorithm R over presentations: per family (reservoir) or one for the whole life (total, raw)
                key = fam if REPLAY16 == "reservoir" else "all"
                fl = fbuf.setdefault(key, []); n = fseen.get(key, 0)
                for e in ents:
                    n += 1
                    if len(fl) < RCAP:
                        fl.append(e)
                    else:
                        j = rng.randrange(n)
                        if j < RCAP:
                            fl[j] = e
                fseen[key] = n
                stages[-1]["reservoir_distinct"] = {f: len(set(v)) for f, v in fbuf.items()}
                if REPLAY16 == "total":   # how the label-free reservoir happens to be composed, by family of entry (for reading only)
                    stages[-1]["reservoir_by_family"] = {f: sum(1 for e in set(fbuf["all"]) if (e[0] == "comp") == (f == "F2") and (f == "F2" or e[1][1] == (97 if f == "F1" else 73))) for f in ("F1", "F2", "F3")}
                elif REPLAY16 == "raw":   # by first byte of the stored prompt (for reading only; the learner never sees this grouping)
                    f1c, f3c = {t[2][0] for t in F1}, {t[2][0] for t in F3}
                    stages[-1]["reservoir_by_family"] = {f_: sum(1 for e in set(fbuf["all"]) if fam_of(e[1]) == f_) for f_ in ("F1", "F2", "F3")}
                    stages[-1]["reservoir_bytes"] = sum(len(e[1]) + len(e[2]) for e in fbuf["all"])
            else:   # all_balanced: every distinct presented entry of the family
                fs = fset.setdefault(fam, set()); fl = fbuf.setdefault(fam, [])
                for e in ents:
                    if e not in fs:
                        fs.add(e); fl.append(e)
            stages[-1]["buffer"] = {f: len(v) for f, v in fbuf.items()} if fbuf else len(buffer)
            if not replaying:
                path.write_text(json.dumps(receipt(), indent=1))   # E16-V: keep the last block's buffer count too
        # E21: the checkpoint, after the stage's receipt and every stage-end admission
        dg = pipeline_digest(presented, expo)
        if replaying:
            ckpt_log(ck, event="replay", stage=si, pipeline=dg["overall"], parts=dg["parts"])
        if si == K:       # the resume point: the replay must reproduce the saved pipeline and the saved stage records exactly
            ckpt_check_digest(json.loads(ckobj["pipeline"]), dg, ckused, f"stage {K}")
            bad = [s_["stage"] for s_, o_ in zip(stages, saved) if json.dumps(s_, sort_keys=True) != json.dumps(o_, sort_keys=True)]
            if bad:
                raise SystemExit(f"e21: the replayed stage-end bookkeeping differs from {ckused.name} at stages {bad}")
            m.load_state_dict(ckobj["model"]); opt.load_state_dict(ckobj["optimizer"]); T.set_rng_state(ckobj["torch_rng"])
            cpu_replay = time.process_time() - t0
            t0 = time.process_time() - float(ckobj["cpu_elapsed"]); cpu_score_total[0] = float(ckobj["cpu_scoring"])
            resumed.append(resume_record(K, ckused, cpu_replay))
            print(f"e21 resumed at stage {K}: pipeline digest {dg['overall'][:12]} matches; replay cpu {cpu_replay:.1f}s", flush=True)
        elif not replaying:
            c_ck = time.process_time(); w_ck = time.perf_counter()
            nbytes = ckpt_write(ck, dict(version=1, stage=si, model=m.state_dict(), optimizer=opt.state_dict(), torch_rng=T.get_rng_state(),
                                         stages=json.dumps(stages), config=json.dumps(cfg21, sort_keys=True), pipeline=json.dumps(dg),
                                         cpu_elapsed=c_ck - t0, cpu_scoring=cpu_score_total[0], resumed=json.dumps(resumed)))
            ckpt_log(ck, event="checkpoint", stage=si, pipeline=dg["overall"], parts=dg["parts"], bytes=nbytes, seconds_to_write=round(time.perf_counter() - w_ck, 4))
    wsha = save_weights(m, path.with_suffix(".pt"))   # final weights, so panels can be re-scored later (8 Nov audit)
    path.write_text(json.dumps(receipt(complete=dict(stages=len(stages), weights=path.with_suffix(".pt").name, weights_sha256=wsha)), indent=1))


def qualify():
    """Mechanics only; the trace itself is qualified in QUALIFY_E11.json."""
    out = {}
    rng = Random(1)
    seed = X.SEEDS[0]
    ents = {f: family_entries(f, seed) for f in ("F1", "F2", "F3")}
    out["entries"] = {f: len(v) for f, v in ents.items()}
    # composition rendering and the round trip from bytes back to an entry
    for e in rng.sample(ents["F2"], 200):
        p, t = render(e, rng)
        raw_len = len(p); p = [b_ for b_ in p if not 65 <= b_ <= 90]      # distractor letters (E16_COMPGAP) are capitals; the rest is the prompt
        assert raw_len <= 10 + 2 * COMPGAP16 and (COMPGAP16 or raw_len == 10)
        assert len(p) == 10 and len(t) == 6 and t[2] == ord(";") and t[-1] == EOS
        back = ("comp", chr(p[3]), chr(p[6]), int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9])))
        assert back == e, (back, e)
        i = E.apply(SYM_OP[e[1]], 97, e[3], e[4]); assert int(bytes(t[:2])) == i and int(bytes(t[3:5])) == E.apply(SYM_OP[e[2]], 97, i, e[5])
    for e in rng.sample(ents["F1"] + ents["F3"], 200):
        p, t = render(e, rng)
        task = next(tk for tk in F1 + F3 if tk[2][0] == p[0])
        assert ("op", task, int(bytes(p[1:3])), int(bytes(p[-3:-1]))) == e
    # held pairs and held triples are disjoint from the taught sets
    P, comp = panels(seed)
    taught = {p: set(triples(p, seed, 2000)) for p in TAUGHT_PAIRS}
    all_taught = frozenset().union(*taught.values())
    for name, items in comp.items():
        for p, _ in items:
            pair = (chr(p[3]), chr(p[6])); tr = (int(bytes(p[1:3])), int(bytes(p[4:6])), int(bytes(p[7:9])))
            assert (pair in HELD_PAIRS) if name != "within" else (pair in TAUGHT_PAIRS)
            assert tr not in all_taught, "panel triple appears in some taught pair's training triples"   # operand-level holdout
    if COMPGAP16:      # the memory assay's own qualification (second audit of 10 Nov): structured round trip, disjointness, visibility
        strip = lambda p_: [b_ for b_ in p_ if not 65 <= b_ <= 90]
        parse = lambda p_: ((chr(p_[3]), chr(p_[6])), (int(bytes(p_[1:3])), int(bytes(p_[4:6])), int(bytes(p_[7:9]))))
        mem = MEMORY[0]; chk = dict(twins=0, lead=0)
        for key in ("twins", "lead"):
            for pair, tw in mem[key].items():
                for x, y in tw:
                    for it in (x, y):
                        pr, tr = parse(strip(it[0])); i_ = E.apply(SYM_OP[pr[0]], 97, tr[0], tr[1])
                        assert pr in HELD_PAIRS and "".join(pr) == pair and tr not in all_taught
                        assert int(bytes(it[1][:2])) == i_ and int(bytes(it[1][3:5])) == E.apply(SYM_OP[pr[1]], 97, i_, tr[2]) and it[1][-1] == EOS
                        first_operand_in_view = bytes(it[0][-10:]).find(bytes(it[0][it[0].index(ord("c")) + 1:it[0].index(ord("c")) + 3])) == 1 and it[0][-10] == ord("c")
                        assert first_operand_in_view == (key == "lead"), "twins hide the first operand at '='; the leading control shows the whole prompt"
                        chk[key] += 1
        for it in mem["within_hidden"]:
            pr, tr = parse(strip(it[0])); assert pr in TAUGHT_PAIRS and tr not in all_taught
        for it in mem["taught_hidden"]:
            pr, tr = parse(strip(it[0])); assert pr in TAUGHT_PAIRS and tr in taught[pr]
        out["memory_panels"] = dict(checked_items=chk, census=mem["census"], within_hidden=len(mem["within_hidden"]), taught_hidden=len(mem["taught_hidden"]),
                                    suffix_only_predictor_both_correct=0, sha256=__import__("hashlib").sha256(json.dumps(mem, sort_keys=True, default=list).encode()).hexdigest())
    out["panel_sizes"] = {n: len(v) for n, v in comp.items()}
    out["shortcut_ceilings"] = shortcut_ceilings(comp)
    # window visibility on the composition prompt: positions whose prediction no longer sees the code byte / first operand
    out["composition_prompt_len"] = 10; out["window"] = E.W
    out["code_byte_visible_until_output_index"] = E.W - 10     # 0: visible only for the first output byte
    # a batch of mixed items passes through update on every arm
    m = E.new_net(seed); m.load_state_dict(T.load(OUT / f"e11_dev_M_{seed}.pt"))
    out["birth_compose_final_before_life"] = {n: score(m, it)["final"] for n, it in comp.items()}   # scored BEFORE the smoke updates (audit, 8 Nov)
    opt = T.optim.AdamW(m.parameters(), lr=.001)
    mixed = [render(e, rng) for e in rng.sample(ents["F1"], 20) + rng.sample(ents["F2"], 24) + rng.sample(ents["F3"], 20)]
    sm = {}
    for arm in ("D", "T"):
        t0 = time.process_time()
        for _ in range(10):
            E.update(m, opt, mixed, arm, None)
        sm[arm] = (time.process_time() - t0) / 10 * 1000
    out["ms_per_update_mixed"] = sm
    t0 = time.process_time(); score(m, comp["+-"]); out["s_per_compose_panel600"] = time.process_time() - t0
    out["compose_final_after_smoke_updates"] = {n: score(m, it)["final"] for n, it in comp.items()}
    (OUT / "QUALIFY_E16.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def develop17(seed):
    """Birth for a windowless architecture: exp11's development (backprop through time on the four mod-89 operations with 0-12
    distractor letters, 24,576 updates, sealed panels at k = 0/4/8/12/20) on the E16_ARCH net; writes e17_dev_{arch}_{seed}."""
    assert ARCH16 != "window"
    # E18b (10 Nov): a birth under the every-byte loss (E16_LOSS=all) has its own name (_lossall) and always goes through update16, which applies
    # the mask; the plain backprop path would ignore it, so it is never taken with that loss. Lives still start from answer-loss births.
    stem = f"e17_dev_{ARCH16}{BTAG}{HTAG}{DEVRULETAG}{'_lossall' if LOSS16 == 'all' else ''}{'_bal' if LOSSW16 == 'balanced' else ''}{H1TAG}{FTAG}{E.DEVTAG}_{seed}"
    path = OUT / f"{stem}.pt"
    cfg = dict(arch=ARCH16, bookdecay=BOOKDECAY16, heads=HEADS16, birthrule=BIRTHRULE16, feedback=FEEDBACK16, traceup=TRACEUP16, projfb=PROJFB16,
               fixedcodes=FIXEDCODES16, decay=(.3 if BIRTHRULE16 == "D" else E.WD), loss=LOSS16, updates=E.DEV_DOSE,
            **(dict(farhead=FARHEAD16, farctrl=FARCTRL16) if FARHEAD16 else {}), **(dict(head1loss=HEAD1LOSS16) if HEAD1LOSS16 != "answer" else {}),
               **(dict(lossw=LOSSW16) if LOSSW16 != "uniform" else {}))
    assert not (RESUME16 == "1" and FARCTRL16 == "shuffle"), "E16_RESUME is not defined for E16_FARCTRL=shuffle: the shuffle's own generator is not in the checkpoint (harness merge, 11 Nov)"
    ck = OUT / f"{stem}.ckpt"; K, ckobj, ckused, resumed = 0, None, None, []
    # E21: weights written but the receipt not yet (a kill between the two) is resumable from the last checkpoint, which rewrites both
    if path.exists() and not (RESUME16 == "1" and not (OUT / f"{stem}.json").exists() and (ck.exists() or Path(str(ck) + ".prev").exists())):     # 10 Nov audit: never accept a checkpoint on its file name alone
        jp = OUT / f"{stem}.json"
        assert jp.exists(), f"{path.name} exists without its receipt: remove or move it before a new birth"
        old = json.loads(jp.read_text()); dflt = dict(heads="top", birthrule="D", feedback="sign", traceup="none", projfb="raw", fixedcodes="off", decay=.3, loss="answer", updates=E.DEV_DOSE, farhead=0, farctrl="none", head1loss="answer", lossw="uniform")
        bad = {k: (old.get(k, dflt.get(k)), v) for k, v in cfg.items() if old.get(k, dflt.get(k)) != v}
        assert not bad, f"{path.name} exists with a different configuration {bad}"
        assert "weights_sha256" not in old or old["weights_sha256"] == sha_file(path), f"{path.name} does not match the hash in its receipt"
        print("e17 dev", stem, "exists with the same configuration; nothing to do", flush=True)
        return
    if RESUME16 == "1":
        resume_gate()
        ckobj, ckused = ckpt_load(ck)
        if ckobj is None:
            print(f"e21 no checkpoint for {stem}: starting from the beginning", flush=True)
            resumed.append(resume_record(0, None, 0.0, note="no checkpoint: started from the beginning"))
    else:
        ckpt_fresh(ck)
    m = E.new_net(seed)
    book0 = m.book.weight.detach().clone()
    if FARHEAD16:
        FAR_GEN[0] = T.Generator().manual_seed(seed + 23)
    plain = HEADS16 == "top" and BIRTHRULE16 == "D" and FIXEDCODES16 == "off" and LOSS16 == "answer"   # every birth before E18: E.update with backprop, decay .3
    step = (lambda items: E.update(m, opt, items, "D")) if plain else (lambda items: update16(m, opt, items, BIRTHRULE16, None))
    opt = adamw16(m, .001, .3 if BIRTHRULE16 == "D" else E.WD, ("book.",) if FIXEDCODES16 == "on" else ())
    census = None      # E16_BOOKDECAY=off: the embedding table without decay, so rows of symbols unseen at birth keep their init
    tp = [(t, E.split(t, .5, seed)[0]) for t in E.DEV]
    pan = {E.NAME(*t): E.panel(t, E.split(t, .5, seed)[1], seed, ks=(0, 4, 8, 12, 20), n=300) for t in E.DEV}
    cfg21 = run_config(stem=stem, seed=seed, birth=cfg, kmax=E.KMAX["M"],
                       panel_sha256=__import__("hashlib").sha256(json.dumps(pan, sort_keys=True, default=list).encode()).hexdigest())
    if ckobj is not None:
        ckpt_check_config(ckobj, cfg21, ckused)
        K = int(ckobj["step"]); resumed = json.loads(ckobj["resumed"])
        print(f"e21 resuming {stem} from update {K} ({ckused.name}): drawing and discarding the batches already used", flush=True)
    bt = Counted(E.batches(tp, seed, E.KMAX["M"]))
    t0 = time.process_time(); curve = []
    for stop in range(X.BLOCK, E.DEV_DOSE + 1, X.BLOCK):
        if stop <= K:      # E21 replay: the stream does not depend on the weights; its batches are drawn and discarded
            for _ in range(X.BLOCK):
                next(bt)
            if stop % 4096 == 0 or stop == E.DEV_DOSE:
                dg = digest_parts(dict(stream=stream_state(bt)))
                ckpt_log(ck, event="replay", step=stop, pipeline=dg["overall"], parts=dg["parts"])
            if stop == K:
                ckpt_check_digest(json.loads(ckobj["pipeline"]), dg, ckused, f"update {K}")
                m.load_state_dict(ckobj["model"]); opt.load_state_dict(ckobj["optimizer"]); T.set_rng_state(ckobj["torch_rng"])
                curve, census = json.loads(ckobj["curve"]), json.loads(ckobj["census"]); row = curve[-1]
                cpu_replay = time.process_time() - t0; t0 = time.process_time() - float(ckobj["cpu_elapsed"])
                resumed.append(resume_record(K, ckused, cpu_replay))
                print(f"e21 resumed at update {K}: stream digest {dg['overall'][:12]} matches; replay cpu {cpu_replay:.1f}s", flush=True)
            continue
        losses, aux_losses, far_losses = [], [], []
        for _ in range(X.BLOCK):
            items = next(bt); losses.append(step(items))
            if not plain and LAST_AUX[0] is not None:
                aux_losses.append(LAST_AUX[0])
            if not plain and LAST_FAR[0] is not None:
                far_losses.append(LAST_FAR[0])
            if census is None:     # which embedding rows the first update reached, by symbol role (grads persist until the next zero_grad)
                census = zero_grad_report(m, X.batch_tensor(items)[0]); print("e17 dev credit census:", census["by_role"], flush=True)
        if stop % 4096 == 0 or stop == E.DEV_DOSE:
            row = dict(step=stop, loss=sum(losses) / len(losses), cpu=time.process_time() - t0,
                       sealed={n: {k: E.score(m, it) for k, it in pk.items()} for n, pk in pan.items()}, book_shift_by_role=book_shift(m, book0))
            if aux_losses:
                row["loss_head1"] = sum(aux_losses) / len(aux_losses)
            if far_losses:
                row["loss_far"] = sum(far_losses) / len(far_losses)
            curve.append(row)
            print("e17 dev", ARCH16 + HTAG + DEVRULETAG, seed, stop, {n: [v[k]["exact"] for k in (0, 4, 8, 12, 20)] for n, v in row["sealed"].items()}, f"cpu {row['cpu']:.0f}s", flush=True)
            dg = digest_parts(dict(stream=stream_state(bt))); c_ck = time.process_time(); w_ck = time.perf_counter()      # E21: the checkpoint at every curve row
            nbytes = ckpt_write(ck, dict(version=1, step=stop, model=m.state_dict(), optimizer=opt.state_dict(), torch_rng=T.get_rng_state(),
                                         curve=json.dumps(curve), census=json.dumps(census), config=json.dumps(cfg21, sort_keys=True),
                                         pipeline=json.dumps(dg), cpu_elapsed=c_ck - t0, resumed=json.dumps(resumed)))
            ckpt_log(ck, event="checkpoint", step=stop, pipeline=dg["overall"], parts=dg["parts"], bytes=nbytes, seconds_to_write=round(time.perf_counter() - w_ck, 4))
    row["silenced_k12"] = {n: E.score(m, pk[12], silence=True) for n, pk in pan.items()}
    if HEADS16 != "top":
        row["head1"] = {n: {k: head1(E.score, m, it) for k, it in pk.items()} for n, pk in pan.items()}
    if FARHEAD16:
        row["far_head_k12"] = {n: far_accuracy(m, pk[12]) for n, pk in pan.items()}
    wsha = save_weights(m, path)
    (OUT / f"{stem}.json").write_text(json.dumps(dict(seed=seed, kmax=E.KMAX["M"], **cfg, credit_census_first_update=census,
                                                      weights=path.name, weights_sha256=wsha, complete=True, curve=curve,
                                                      **({"resumed": resumed} if resumed else {})), indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if sys.argv[1] == "qualify":
        qualify()
    elif sys.argv[1] == "dev":
        develop17(int(sys.argv[2]))
    else:
        life(sys.argv[1], int(sys.argv[2]))
