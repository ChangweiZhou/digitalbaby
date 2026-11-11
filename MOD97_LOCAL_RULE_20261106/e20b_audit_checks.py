"""Coordinator's checks of Codex's audit of the E20-B amendments (PROJECT_DOCUMENTS/MINIFLY_E20B_AMENDMENTS_AUDIT_20261110.md), 10 Nov 2026.
Each claim about the inherited source is run here, not read. Trains nothing; writes exp45/E20B_AUDIT_CHECKS.json.
Run from this folder: python3 e20b_audit_checks.py"""
import os, sys, json, shutil, subprocess, tempfile, pathlib
out = {}

# 1. a bare `exp16_integrated.py D <seed>` is the window network with embedding decay on; an unknown architecture name stops at import
env = {k: v for k, v in os.environ.items() if not k.startswith(("E16_", "E11_"))}
r = subprocess.run([sys.executable, "-c", "import exp16_integrated as X; print(X.ARCH16, X.BOOKDECAY16, X.PFX, X.COMPGAP16)"], env=env, capture_output=True, text=True)
out["1_bare_defaults"] = r.stdout.split()
r = subprocess.run([sys.executable, "-c", "import exp16_integrated"], env=dict(env, E16_ARCH="gru"), capture_output=True, text=True)
out["3_unknown_arch_stops_at_import"] = r.returncode != 0 and "AssertionError" in r.stderr

os.environ.update(E16_ARCH="bank2", E16_BOOKDECAY="off", E11_STAGE_UPDATES="4", E11_DEV_UPDATES="512")
import torch as T, torch.nn as nn
import exp16_integrated as X, exp11_cell as E
P, comp = X.panels(73129)
out["1_reference_names"] = dict(arch=X.ARCH16, bookdecay=X.BOOKDECAY16, prefix=X.PFX, atag=X.ATAG, drytag=X.DRYTAG, devtag=E.DEVTAG)
out["3_k40_panels_present_when_windowless"] = any(str(k).endswith("40") or (isinstance(v, dict) and 40 in v) for k, v in P.items())

# 2. the birth scorer generates at most four bytes; a worked composition target has six
items = [it for name, its in comp.items() if name != "within" for it in its][:64]
tl = sorted({len(t) for _, t in items})
m = E.new_net(73129)
gen = E.respond(m, [p for p, _ in items])                       # the birth helper's default cap
gen6 = E.respond(m, [p for p, _ in items], cap=X.CAP)
out["2_target_lengths"] = tl
out["2_birth_scorer_max_generated"] = max(len(g) for g in gen)
out["2_life_scorer_cap"] = X.CAP
out["2_birth_scorer_can_score_composition_exact"] = max(len(g) for g in gen) >= min(tl)
# the three fixtures, through the life scorer with the responder replaced by a stub
real = E.respond
def fixture(make):
    E.respond = lambda m_, prompts, store=None, cap=4, silence=False: [make(t) for _, t in items]
    try: return X.score(None, items)
    finally: E.respond = real
wrong_mid = lambda t: [(t[0] - 48 + 1) % 10 + 48] + list(t[1:])
out["2_fixture_fully_correct"] = fixture(lambda t: list(t))
out["2_fixture_wrong_intermediate_right_suffix"] = fixture(wrong_mid)
out["2_fixture_missing_eos"] = fixture(lambda t: list(t[:-1]) + [t[-2]])
# a `taught` key in the dictionary given to shortcut_ceilings
try:
    X.shortcut_ceilings({"taught": items}); out["2_shortcut_ceilings_taught_key"] = "accepted"
except ValueError as e:
    out["2_shortcut_ceilings_taught_key"] = f"ValueError: {e}"

# 3. the optimizer helper finds embeddings by the name prefix `book.` only
class Other(nn.Module):
    def __init__(s):
        super().__init__(); s.emb = nn.Embedding(258, 8); s.lin = nn.Linear(8, 8)
class Named(nn.Module):
    def __init__(s):
        super().__init__(); s.book = nn.Embedding(258, 8); s.lin = nn.Linear(8, 8)
def emb_decay(mod, attr):
    opt = X.adamw16(mod, .001, .3); w = getattr(mod, attr).weight
    return [g["weight_decay"] for g in opt.param_groups if any(p is w for p in g["params"])][0]
out["3_embedding_decay_when_named_book"] = emb_decay(Named(), "book")
out["3_embedding_decay_when_named_otherwise"] = emb_decay(Other(), "emb")
out["3_bos_row_is_padding"] = m.book.padding_idx == E.BOS

# 4. admission does not look at the seed: a complete receipt and its weights, copied under another seed's name (a fixture: the weights are an
#    untrained net of seed 73129 and the receipt is written here; what is exercised is admit_birth, not a birth)
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    own = td / "e17_dev_bank2_nbd_dry512_73129.pt"; T.save(m.state_dict(), own)
    rec = dict(arch="bank2", seed=73129, bookdecay="off", heads="top", birthrule="D", fixedcodes="off", loss="answer",
               curve=[dict(step=512)], complete=True, weights_sha256=X.sha_file(own))
    own.with_suffix(".json").write_text(json.dumps(rec))
    fake = td / "e17_dev_bank2_nbd_dry512_93683.pt"
    shutil.copy(own, fake); shutil.copy(own.with_suffix(".json"), fake.with_suffix(".json"))
    res = {}
    for name, f in (("own_name", own), ("copied_as_seed_93683", fake)):
        try: res[name] = f"admitted ({X.admit_birth(f, X.sha_file(f))})"
        except AssertionError as e: res[name] = f"refused: {e}"
    out["4_admission"] = dict(receipt_seed=73129, **res)
    out["4_admission_before_the_fix"] = dict(at_commit="522174bb", own_name="admitted (own receipt)", copied_as_seed_93683="admitted (own receipt)",
                                             note="recorded from this script's run before admit_birth compared the receipt's seed with the file name")
out["4_expected_birth_fields"] = sorted(X.expected_birth())
# was the hole ever used? every life receipt on file that records its birth's hash: is any birth recorded under two seeds, and by which path was it admitted
import re, glob, collections
by = collections.defaultdict(set); how = collections.Counter()
for j in glob.glob("exp45/*_life_*.json"):
    txt = open(j).read(); mm = re.search(r'"birth_sha256": "([0-9a-f]{64})"', txt)
    if not mm: continue
    by[mm.group(1)].add(re.findall(r"_(\d+)(?:_killed)?\.json$", j)[0])
    a_ = re.search(r'"birth_admission": "([^"]*)"', txt); how[a_.group(1) if a_ else "none recorded (before admission existed)"] += 1
out["4_births_on_file"] = dict(life_receipts_with_birth_hash=sum(how.values()), distinct_births=len(by),
                               births_used_under_more_than_one_seed=sum(len(v) > 1 for v in by.values()), admission_path=dict(how))

# scientific note 1: the size of G3's added matrix
out["5_g3_added_weights"] = dict(shape=[E.D, E.C], weights=E.D * E.C, bias="none (the brief specifies one added matrix)", bank2_parameters=sum(p.numel() for p in m.parameters()))
pathlib.Path("exp45/E20B_AUDIT_CHECKS.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
