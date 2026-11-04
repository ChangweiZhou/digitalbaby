"""Source-backed V214B runtime with explicit causal observation and decoding."""
from pathlib import Path
from hashlib import sha256
from copy import deepcopy
from textwrap import indent, dedent

SOURCE = Path(__file__).resolve().parent.parent / "minifly_v214b_round20_fresh_seed_20261104/cmr.py"
raw = SOURCE.read_bytes()
if sha256(raw).hexdigest() != "ed9d054140a06641f24f7762a7cd09cb2532e6aa49b486582b360bb14930c739":
    raise RuntimeError("Unexpected V214B source")
source = raw.decode()
header, loop = source.split("for i,byte in enumerate(data):\n", 1)
header = header.replace("data=open(sys.argv[1],'rb').read()", "")
header = header.replace("BITS=np.zeros(len(data)); ", "")
pre, body = loop.split("    for b in range(7,-1,-1):\n", 1)
body, post = body.split("    hist.append(byte)", 1)
post = "    hist.append(byte)" + post.split("\nprint(", 1)[0]
body = dedent(body)
body = body.replace("bit=(byte>>b)&1\n", "")
prediction, writes = body.split("BITS[i]+=", 1)
writes = writes.split("\n", 1)[1].rsplit("c0=", 1)[0]
for name, default in (("w=W[k]", "np.full(len(x),0.3/len(x)*8)"),
                      ("w2=W2[(c0,econf)]", "np.full(len(x2),1/len(outs))")):
    prediction = prediction.replace(f"{name}={default}", f"{name.split('=')[0]}={default}")
# Keep the source's exact dictionary insertion semantics on observed bits.
writes = """for gate,W,(w,_) in zip(GATES1,W1,keys):
    k=gate(hist,c0,econf)
    if k not in W: W[k]=w
if (c0,econf) not in W2: W2[(c0,econf)]=w2
""" + writes
post = post.replace("if LINE: end_line(LINE)", "if LINE and learn: end_line(LINE)")
step = """def step(observed=None, learn=True):
    global H1,H2,LINE
    loss=0.0
""" + pre + """    for b in range(7,-1,-1):
""" + indent(prediction, "        ") + """        bit = int(p > 0.5) if observed is None else (observed >> b) & 1
        loss += -math.log2(p if bit else 1-p)
        if learn:
""" + indent(writes, "            ") + """        c0=(c0<<1)|bit
    byte=c0 & 255
""" + post + """
    return byte, loss
"""
RUNTIME = compile(header + "\n" + step, str(SOURCE), "exec")
STATE = ("P", "C", "BAG", "DF", "NL", "rnd", "W1", "W2", "WORD", "LINE", "hist", "H1", "H2")
LONG_TERM = STATE[:5] + STATE[6:8]


class Learner:
    def __init__(self, mode="1"):
        self.state = {}
        exec(RUNTIME, self.state)
        self.state.update(REC=mode, LAM=0.7)

    def observe(self, data, learn=True):
        return [self.state["step"](byte, learn)[1] for byte in data]

    def generate(self, limit=32):
        output = bytearray()
        for _ in range(limit):
            byte, _ = self.state["step"](None, False)
            output.append(byte)
            if byte == 10:
                break
        return bytes(output)

    def fork(self):
        child = Learner(self.state["REC"])
        child.state.update({key: deepcopy(self.state[key]) for key in STATE})
        return child
