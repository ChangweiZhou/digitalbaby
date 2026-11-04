# Cycle 2 · Matching

Refinement. Fix the salted orderings and use a domain-separated SHA-256 hash per short gram with a four-row fan-out for REHASH. Draw replacement rows from outside the original short code. Use the same feasible q and the same drops for both changed arms.

Audit. Check the exact short-code reference, shared drop mask, causal suffix, absence of four-gram dependence in the rehash proposals, coefficient equality, and bounded proposal counts. Count hashing separately from row traffic.

Test. 797 empty, short, repeated and random prefixes passed. Both replacement arms changed the code on 698 prefixes and matched the baseline's active-row count and coefficients on every prefix.
