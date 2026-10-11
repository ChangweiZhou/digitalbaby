"""Technical-cycle admission only; no DEV or confirmation permission."""
import math
ARMS=('STABLE','U042-WG1','U021-SF1')
DOSES=(10,20,40)
BRANCHES=('W','N_OLD','SHUFFLED')
PHASES=('OLD_END','DAY1','NEW_END','FINAL')
TECHNICAL_WORLDS=tuple(range(834001,834007))
DT=30./14.
DAY_TICKS=40320
DAY=86400.
ALPHABET=b'012345678'
SCALE=1.3452365735750882
ORDINAL_ETA=1.
SLOW_SHARE=.8
ROW_BOUND=2.
VISIT_CAP=65535
VERSION='ADDITION_EFFICIENCY_TECHNICAL_V1'
def require_world(world):
    if type(world) is not int or world not in TECHNICAL_WORLDS:
        raise ValueError('Technical-cycle IDs only; DEV and confirmation not authorized')
def valid_time(t,minimum):
    if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<minimum:
        raise ValueError('finite monotonic clock required')
    return float(t)
