import math
ALPHABET=b'0123'
SCALE=1.3452365735750882
DT=30./14.
DAY=86400.
REPEATS=8
PROBE_REPEATS=2
DEV_WORLDS=(822001,822101,822201,822202)
SCIENCE_WORLDS=tuple(range(823001,823033))
BRANCHES=('W','N_OLD','N_REV','N_ALL')
ETA=6.*SCALE
SLOW_SHARE=.8
FAST_TAU=3600.
SLOW_TAU=30.*DAY
ROW_BOUND=2.
VERSION='U045_LOCAL_RESIDUAL_CONTENT_DEV_V2'
def require_dev(world):
    if type(world) is not int or world not in DEV_WORLDS:
        raise ValueError('Only declared technical DEV worlds; science not authorized')
