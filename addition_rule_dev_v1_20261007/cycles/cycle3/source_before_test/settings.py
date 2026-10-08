"""Four declared DEV worlds; no science admission or automatic extension."""
import math
ALPHABET=b'012345678'
DT=30./14.
DAY=86400.
REPEATS=40
BRANCHES=('W','N_OLD','SHUFFLED')
ARMS=('NATIVE','RESIDUAL')
DEV_WORLDS=(831001,831101,831201,831202)
SCALE=1.3452365735750882
ETA=4.*SCALE
FAST_TAU=3600.
SLOW_TAU=30.*DAY
SLOW_SHARE=.8
ROW_BOUND=2.
VISIT_CAP=65535
VERSION='ADDITION_SHARED_LOCAL_RESIDUAL_V3_ORDINAL_HUBER'
ORDINAL_ETA=1.
PHASES=('old','day1','new','day2')
def require_dev(world):
    if type(world) is not int or world not in DEV_WORLDS:
        raise ValueError('Only four declared DEV worlds; science is not authorized')
def valid_time(t,minimum):
    if isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<minimum:
        raise ValueError('finite monotonic clock required')
    return float(t)
