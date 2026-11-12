"""Sizing sweep: the configurations (brief OPUS_TASK_TARGET_SIZING_SWEEP_20261111.md rev. 3, "What is varied").

Only drivers import this file. The generator, oracle, solver and checker never do: each is handed the configuration as an argument and
computes from it for itself. The default configuration is step 0's SIZES, value for value (checked against TARGET_STEP0_CHECK.json["sizes"]).
"""
import copy
import hashlib
import json

_DEFAULT = dict(
    contexts=24, cues_per_family=12, modulus=97, window=10,
    taught_per_family=216, first_encounter_per_family=48, first_encounter_sets=("acquire", "retain", "revise"), per_set=16,
    sealed_twins_per_family=12, renderings_per_panel=150, panel_filler=(10, 14), teaching_filler=(0, 14),
    first_encounter_filler=(10, 14),
    changed_cues=2, revision_pairs_with_changed_cue=6,
    acquire_reps=4, absent_reps=2, return_reps=1, revise_reps=2,
    acquire_first_from=0.5, retain_within=0.25, revise_first_from=0.4, change_at_fraction=0.10,
    line_delays=(0, 1, 2, 3),
    control_twins=2,
    distinct_pair_share=0.8,
    evidence_after_change=3,
)
MODULI = (97, 31, 11)
PRESENTATIONS = (4, 32, 256)
ROSTER_WORLDS = (101, 102, 103)
PROFILE_WORLD = 104


def default():
    return copy.deepcopy(_DEFAULT)


def make(modulus=97, presentations=4):
    c = default(); c["modulus"] = modulus; c["acquire_reps"] = presentations
    return c


def chash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def settings():
    """The nine (modulus, presentations) settings of the grid."""
    return [(m, p) for m in MODULI for p in PRESENTATIONS]
