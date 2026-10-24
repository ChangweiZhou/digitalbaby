# V43 protocol erratum — EVENT-ORACLE capture

Recorded before V43 scientific execution. This corrects one mathematical interpretation sentence in the adopted independent design review, reproduced in `PROTOCOL.md` under “Endpoint/gate denominators,” Q8:

> Capture can exceed1 because a probability mixture may beat the event oracle that chooses only between fixed experts; do not clip it.

The claim that capture can exceed one for that reason is incorrect. For the specified route-G probability mixture with any weight \(0\le w_t\le1\),

\[
p_m(y_t)=w_t p_S(y_t)+(1-w_t)p_F(y_t)
\le \max\{p_S(y_t),p_F(y_t)\}.
\]

Therefore \(\ell_m(t)\ge\min\{\ell_S(t),\ell_F(t)\}=\ell_O(t)\). Summing over exactly the same route-G events gives \(L_m\ge L_O\). With positive headroom \(H=L_H-L_O>0\), the defined capture

\[
C_m=\frac{L_H-L_m}{L_H-L_O}
\]

is at most one, up to floating-point error. It can be negative, with no universal finite lower bound as positive headroom becomes small. Zero headroom leaves the ratio undefined. These facts also apply to an equal-weight average of defined seed-level capture ratios.

`EVENT-ORACLE` selects the better expert separately on each event. `SLOT-ORACLE` selects one fixed expert for an entire slot generation. A changing mixture can outperform SLOT-ORACLE across that generation; this does not permit it to outperform the specified EVENT-ORACLE.

The frozen source, saved-state identities, scoring formula, Q8 threshold and other gates remain unchanged. Capture values must still be reported without clipping. A value above one beyond numerical tolerance should trigger an audit of matched event sets, expert losses, units and denominators, rather than be interpreted as beating EVENT-ORACLE.

Independent audit recommendation: for every completed seed/load and every reported route-G policy, reconstruct matched-event \(L_m,L_H,L_O\); verify \(L_m\ge L_O\) within a documented floating-point tolerance and, when \(H>0\), verify \(C_m\le1\) with a tolerance scaled by \(H\). Check the equal-seed mean separately. Do not silently replace, truncate or retune a reported ratio.

This erratum supersedes only the quoted interpretation sentence. It does not state or predict any V43 scientific outcome.
