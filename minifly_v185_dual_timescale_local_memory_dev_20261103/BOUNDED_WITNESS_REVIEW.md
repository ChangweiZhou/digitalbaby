# V185 bounded qualification follow-up

The consolidated branch still registers V185 as the latest design. This exercise tested its revised C0 readout qualification against the existing deterministic roster and native projection.

1. **Design.** Use one direct ridge solve over a 65-column feature matrix, including bias, as a cheap constructive witness. The earlier exercise charged 480 labeled sites and four synthetic target updates against the 768 technical target reserve. Charge the new 240 fit sites separately from read-only verification. This leaves insufficient reserve for another full fit.
2. **Audit and test.** The witness used the unchanged 80 keys and 240 target sites. Float32 greedy decoding returned the exact two bytes and own EOS for 44/80 keys. Its minimum correct-class logit margin was −0.360495. Peak RSS was 245,743,616 bytes and wall time was 0.825 seconds. The witness consumed 80 fitting records and 240 fitting targets, followed by 240 read-only margin checks and 80 read-only decoding probes. The iterative constraint loop was disabled for this pass.
3. **Revision.** Keep the existing admission gate closed. A ridge fit optimizes squared error, so this partial score leaves multiclass separability unresolved. A future constraint-based test needs an explicit budget for all repeated label-conditioned checks, a measured memory guard, and a validated 80/80 own-EOS witness. The current cumulative technical charge is 240 fitting records and 724 fitting or synthetic target exposures. Candidate scientific optimizer records remain zero.

Disposition: REPRESENTATION_CEILING_UNRESOLVED. The bounded witness resolves the immediate memory cost for this solver only. Full roster admission remains pending.
