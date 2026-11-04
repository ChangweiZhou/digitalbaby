import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from bridge import LONG_TERM, SOURCE, Learner


def fingerprint(learner):
    return hashlib.sha256(pickle.dumps([learner.state[k] for k in LONG_TERM])).hexdigest()


def exact(output, answer):
    return output == answer + b"\n"


def main():
    sample = b"Ana went to the hall.\nAna got the cup.\nWhere is the cup? hall\n" * 7
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "stream"
        path.write_bytes(sample)
        for mode in ("0", "1", "shuf"):
            subprocess.run([sys.executable, str(SOURCE), str(path)],
                           env=dict(os.environ, REC=mode, LAM="0.7"),
                           check=True, capture_output=True)
            expected = np.load(str(path) + f".bits_{mode}.npy")
            actual = Learner(mode).observe(sample)
            assert np.array_equal(actual, expected), (mode, np.max(np.abs(actual - expected)))
    learner = Learner()
    learner.observe(sample + b"Where is the cup? ")
    before = fingerprint(learner)
    output = learner.generate()
    assert before == fingerprint(learner)
    assert output and len(output) <= 32
    assert exact(b"hall\n", b"hall")
    assert all(not exact(x, b"hall") for x in (b"", b"hall", b"hal\n", b"hall \n", b"hall\n\n"))
    print(json.dumps(dict(cycle=1, parity_modes=3, parity_bytes_per_mode=len(sample),
                          frozen_write_check=True, generated_hex=output.hex(), exact_boundary_cases=6)))


if __name__ == "__main__":
    main()
