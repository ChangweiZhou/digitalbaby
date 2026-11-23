"""net12_rcrep: net12_rc (persistent rule cells read by the relay gate) combined with net12_rep (the replaying flag and the optional
fast/slow gate split) by cooperative multiple inheritance (23 November 2026, helper rcseeds). No new mechanism: MRO is
net12_rep.Tissue -> net12_rc.Tissue -> net12.Tissue, so with slow=0 net12_rep's step only passes through to net12_rc's step, and the
targeted replay itself lives in the driver (t23_life_idea12_rc.py, a copy of t23_life_idea12.py importing this module)."""
import net12_rep, net12_rc


class Tissue(net12_rep.Tissue, net12_rc.Tissue):
    pass
