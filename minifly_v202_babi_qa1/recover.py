"""Rebuild the reviewed Round 10 Python source from its session export."""
import json
from pathlib import Path
import sys

def recover(export):
    events=json.loads(export.read_text())
    def fragment(event):
        command=events[event]['input']['command']
        return command.split('\nEOF',1)[0].split('\n',1)[1]
    Path('babi1.py').write_text(fragment(2070))
    for event in (2080,2085,2090,2100,2105,2110,2135,2140,2145,2150,2165,2175):
        if event==2100:
            file=Path('babi4.py')
            s=file.read_text().replace(
                "cut=max(h.rfind(b'.'),h.rfind(b'?'),h.rfind(b'\\n'))",
                "hh=h[:-2]; cut=max(hh.rfind(b'.'),hh.rfind(b'?'),hh.rfind(b'\\n'))")
            file.write_text(s)
        if event==2135:
            file=Path('babi7.py')
            s=file.read_text().replace('if gw>0.05:', 'if gw>GTH:')
            s=s.replace("CREDIT=os.environ.get('CREDIT','match')",
                        "CREDIT=os.environ.get('CREDIT','match'); GTH=float(os.environ.get('GTH','0.05')); OSC=float(os.environ.get('OSC','1'))")
            s=s.replace("gw=1.0 if bytes(hist[-4:])==b'the '", "gw=OSC if bytes(hist[-4:])==b'the '")
            file.write_text(s)
        if event==2175:
            file=Path('babi12.py')
            file.write_text(file.read_text().replace("print(json.dumps({'store':STOREMODE,",
                                                     "print(json.dumps({'gth':GTH,'valid':VALID,'store':STOREMODE,"))
        exec(compile(fragment(event),f'reviewed-event-{event}','exec'))
    return Path('babi13.py').read_text()

if __name__=='__main__':
    import tempfile
    import os
    original=Path.cwd()
    export=Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        source=recover(export)
        os.chdir(original)
    source=source.replace("run('/tmp/babi/qa1_single-supporting-fact_train.txt','/tmp/babi/qa1_single-supporting-fact_test.txt')",
                          "run(sys.argv[1], sys.argv[2])")
    Path('babi13.py').write_text(source)
