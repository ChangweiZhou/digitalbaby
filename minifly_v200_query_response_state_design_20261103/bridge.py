"""Replay the pinned historical evaluator with read-only tracing."""
import hashlib
import json
import os
from pathlib import Path

assert os.environ["PYTHONHASHSEED"] == "0"
source = Path("parent_hop.py").read_text()
assert hashlib.sha256(source.encode()).hexdigest() == "6ec531682970b2362d44ca2053a1bcedf09d1c8757b4f5f225f34bb6910a4c8a"
source = source[:source.index("for name, QQ in (")]
source = source.replace("    one1=", """
    before = hashlib.sha256(W.tobytes()+IMP.tobytes()).digest()
    forced = []
    matched_stop = []
    first_errors = [0]*6
    for f,s in zip(first,second):
        prefix=Q(f)
        out=gen(prefix)
        gold=s+b'.'
        for j,y in enumerate(gold):
            if out[j:j+1]!=bytes([y]):
                first_errors[j]+=1
                break
        guesses=[int(W[bag(prefix+gold[:j])].sum(0).argmax()) for j in range(6)]
        forced.append(guesses)
        if out[:5]==s:
            matched_stop.append(out[5:]==b'.')
    assert before == hashlib.sha256(W.tobytes()+IMP.tobytes()).digest()
    trace=dict(gold_stop=sum(p[-1]==46 for p in forced), first_errors=first_errors,
               exact_payload_n=len(matched_stop), stop_given_payload=sum(matched_stop))
    one1=""").replace("return {'strict_two_hop'", "return {**trace,'strict_two_hop'")
env = {"hashlib": hashlib}
exec(compile(source, "parent_hop.py", "exec"), env)
forms = {
    "trained": lambda k: b"the code of "+k+b" is ",
    "remember": lambda k: b"remember "+k+b": ",
    "tell_me": lambda k: b"tell me, what is "+k+b"? ",
}
rows = {}
for name, form in forms.items():
    env["QQ"] = form
    rows[name] = env["run"](1)
assert [rows[n]["strict_two_hop"] for n in forms] == [122/128, 0, 0]
assert [rows[n]["key_prefix"] for n in forms] == [122/128,54/128,0]
assert [rows[n]["hop1_eos_rate"] for n in forms] == [1,1/128,1]
assert [rows[n]["hop1_key_to_digit"] for n in forms] == [1,.984,0]
assert rows["trained"]["strict_two_hop"] > .85
assert rows["remember"]["key_prefix"]-rows["remember"]["hop1_key_to_key_with_EOS"] >= .10
Path("CYCLE1_TEST.json").write_text(json.dumps(dict(status="PASS", presentations=12288, targets=49152, rows=rows),indent=2)+"\n")
