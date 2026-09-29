"""Immutable inherited EVENT model; V79 owns its configuration independently."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
sys.path.insert(0, str(BASE / 'iteration23/src'))
from common78 import (np, pd, json, time, copy, sha, atomic_json, Fly,
    FrozenReader, READER, observed_activity, state_arrays, state_error, DEN,
    geometry, pair_events, step_encoded, ParentLearner, measured)
from model73 import advance73
sys.path.insert(0, str(ROOT / 'src'))
CFG = json.loads((ROOT / 'config.json').read_text())
