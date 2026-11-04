"""Common bounded recall for V215 engineering arms."""
from collections import Counter, deque
from itertools import islice

from bridge import Learner

ROW_LIMIT = 16
CUE_LIMIT = 12
POOL_LIMIT = 64


def attach(learner, shuffle=False, seed=215):
    from random import Random

    state = learner.state
    rng = Random(seed)
    recent = deque(maxlen=POOL_LIMIT)
    counts = Counter()
    end_line = state["end_line"]

    def write(words):
        end_line(words)
        recent.extend(sorted(set(words)))

    def visit(key, blocked, stage):
        bag = state["BAG"]
        row = bag.get(key, {})
        degree = min(len(row), ROW_LIMIT)
        match = []
        if degree:
            counts[f"{stage}_available"] += 1
        # Both arms perform the same bounded search and draw on every call.
        for candidate in recent:
            if (candidate != key and candidate not in blocked and len(candidate) == len(key)
                    and min(len(bag[candidate]), ROW_LIMIT) == degree and degree):
                match.append(candidate)
        counts[f"{stage}_matched"] += bool(match)
        donor = rng.choice(match) if match else None
        selected = bag.get(donor, {}) if shuffle else row
        counts[f"{stage}_calls"] += 1
        counts[f"{stage}_slots"] += ROW_LIMIT
        counts[f"{stage}_visits"] += min(len(selected), ROW_LIMIT)
        items = list(islice(selected.items(), ROW_LIMIT))
        items.extend([(None, 0.0)] * (ROW_LIMIT - len(items)))
        return items

    def recall(cues):
        idf = state["idf"]
        h1, h2 = {}, {}
        for key in sorted(cues)[:CUE_LIMIT]:
            weight = idf(key)
            for word, strength in visit(key, cues, "h1"):
                if word is not None and word not in cues:
                    h1[word] = h1.get(word, 0) + weight * strength * idf(word)
        for key, weight in sorted(h1.items(), key=lambda pair: -pair[1])[:6]:
            for word, strength in visit(key, cues, "h2"):
                if word is not None and word not in cues and word != key:
                    h2[word] = h2.get(word, 0) + weight * strength * idf(word)
        counts["h1_active"] += bool(h1)
        counts["h2_active"] += bool(h2)
        return h1, h2

    state.update(cands=recall, end_line=write)
    learner.control = (shuffle, rng, recent, counts)
    return counts


def controlled(shuffle=False, seed=215):
    learner = Learner()
    attach(learner, shuffle, seed)
    return learner


def fork(learner):
    child = learner.fork()
    if hasattr(learner, "control"):
        shuffle, rng, recent, _ = learner.control
        attach(child, shuffle)
        child.control[1].setstate(rng.getstate())
        child.control[2].extend(recent)
    return child


def qualified(left, right):
    by_hop = {}
    for hop in ("h1", "h2"):
        available = right[f"{hop}_available"]
        matched = right[f"{hop}_matched"]
        visits = left[f"{hop}_visits"]
        delta = abs(visits - right[f"{hop}_visits"])
        by_hop[hop] = dict(coverage=matched / available if available else 0,
                           relative_visit_delta=delta / visits if visits else 1,
                           slots_equal=left[f"{hop}_slots"] == right[f"{hop}_slots"])
    return by_hop, all(row["coverage"] >= .95 and row["relative_visit_delta"] <= .05
                       and row["slots_equal"] for row in by_hop.values())
