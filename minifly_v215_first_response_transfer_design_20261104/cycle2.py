import json

from controls_cycle2 import controlled, qualified
from worlds import family


def main():
    histories, interference, _, pairs = family(2150201)
    left, right = controlled(), controlled(True)
    prefix = histories[0] + interference + pairs[0][0].prompt
    left.observe(prefix)
    right.observe(prefix)
    l, r = left.control[3], right.control[3]
    audit, accepted = qualified(l, r)
    assert l["h1_visits"] and r["h1_visits"]
    assert l["h1_calls"] == r["h1_calls"]
    assert len(pairs) == 8 and all(len(x) == 2 for x in pairs)
    print(json.dumps(dict(cycle=2, engineering_pairs=len(pairs), oracle_exact=16,
                          matched_history_bytes=len(histories[0]), true=dict(l),
                          sham=dict(r), matching=audit, sham_qualified=accepted), indent=2))


if __name__ == "__main__":
    main()
