"""Check source availability and provenance without authorizing a replay."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

def gate(root):
    required=("cmr.py","generator.py","HISTORICAL_RECEIPT.json")
    if any(not (root/p).is_file() for p in required):
        return "MISSING_ROUND20_PARENT"
    receipt=json.loads((root/required[-1]).read_text())
    if any(sha256((root/p).read_bytes()).hexdigest()!=receipt.get("sources",{}).get(p)
           for p in required[:-1]):
        return "SOURCE_MISMATCH"
    return "NEEDS_INDEPENDENT_REPLAY"

def main():
    actual=gate(Path("parent"))
    assert actual=="MISSING_ROUND20_PARENT"
    with TemporaryDirectory() as directory:
        p=Path(directory)
        sources={}
        for n in ("cmr.py","generator.py"):
            data=b"# nonexecuted provenance fixture\n"
            (p/n).write_bytes(data)
            sources[n]=sha256(data).hexdigest()
        (p/"HISTORICAL_RECEIPT.json").write_text(json.dumps(dict(sources=sources)))
        assert gate(p)=="NEEDS_INDEPENDENT_REPLAY"
        (p/"cmr.py").write_bytes(b"changed")
        assert gate(p)=="SOURCE_MISMATCH"
    Path("CYCLE1_TEST.json").write_text(json.dumps(dict(
        technical_check="PASS",status=actual,parent_byte_applications=0))+"\n")

if __name__=="__main__":
    main()
