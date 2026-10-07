"""Technical development configuration; science is not authorized by this module."""
ALPHABET = b'0123'
WINDOW = 4
CONTACTS = 24
SCALE = 1.3452365735750882
BYTE_SECONDS = 30. / 14.
DAY_SECONDS = 86400.
OLD_KEYS = 16
NEW_KEYS = 8
REVISED_KEYS = 8
REPEATS = 8
PROBE_REPEATS = 2
DEV_WORLDS = (810001, 810101, 810201, 810202, 810203, 810204)
SCIENCE_WORLDS = tuple(range(811001, 811033))
VERSION = 'A_NATIVE_OBSERVATION_DEV_1'


def require_dev(world):
    if type(world) is not int or world not in DEV_WORLDS:
        raise ValueError('Only the six declared DEV worlds are authorized; science is locked')
