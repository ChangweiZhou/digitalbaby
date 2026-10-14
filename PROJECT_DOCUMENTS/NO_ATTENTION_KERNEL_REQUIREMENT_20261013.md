# Kernel architecture requirement — 13 October 2026

The user explicitly requires no Transformers or attention mechanisms. Apply this to every newly proposed kernel and comparison baseline, including encoding, recurrent computation, memory access, readout and learned update controllers.

Exclude self/cross/linear attention, query–key scoring, content-addressed key/value reads, attention pooling and equivalent fast-weight retrieval. Ordinary GRU neuron gates, fixed local connectivity, elementwise neural modulation and a vocabulary softmax are permitted; they must not conceal attention over positions or memory entries.

Arithmetic meaning must still be learned from raw byte feedback. No supplied numeral features, operand roles, arithmetic laws or intermediate calculation targets. The objective stays retained unseen four-operation/mixed arithmetic ability per sample and compute.

Current revision: [KERNEL-SEARCH V3](../arithmetic_kernel_no_attention_design_v3_20261013/EXPERIMENT.md). Old Transformer/attention proposals and papers remain historical records, excluded from the active roster. This restriction is a design choice, not negative experimental evidence about their mechanism families. It does not rewrite frozen results.
