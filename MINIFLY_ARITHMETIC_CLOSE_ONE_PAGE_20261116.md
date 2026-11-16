# MiniFly arithmetic line: where it stands (one page, 16 November 2026)

For Mr. Xu. Written by the coordinator from the review (`MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md`, sections 9.122 to 9.146). Everything below is evidence level E2: small recurrent networks on byte-level modular arithmetic, three seeds unless stated, scored on operand pairs never taught. "The sweep" is the project's local rule (errors sent back through weight signs); backpropagation is the reference.

## What is established

1. **At birth the sweep matches backpropagation** when only answer bytes are targets: 98 to 100% on untaught pairs and 96 to 100% at three times the trained gap (E26).
2. **On untaught combinations of operations the sweep is behind**, and the loss is in the recurrent map's backward pass. Learned feedback there (Kolen–Pollack) recovers most of it on three seeds; this has not been replicated (E31).
3. **With every byte a target the sweep fails to extrapolate in length**, and the cause is error from the unpredictable distractor positions, on three seeds (E33). Scaling each byte's error by the network's own confidence repairs it on two of three seeds and slows early learning; it does the same job under no rule-specific mechanism (E34).
4. **Experience makes new learning far cheaper.** A network that has lived learns a new family from a tenth of its pairs in about 64 updates; a fresh one memorises and stays near 10% (E37, E40). All three rules agree.
5. **Nothing unused is kept.** A family is lost within tens of updates of learning anything else, contradicting or unrelated (E38, E43), and within about 2,000 updates of learning nothing, by weight decay alone. After a whole life it comes back in 64 to 128 updates of reminding (E36), so at least in that case it was not destroyed.
6. **Hand-wired circuits did not help.** They generalise when their inputs are handed over; backpropagation does not learn to hand them over from the byte stream, even when forced. Both inventories closed by their registered stop rule (E35).

## What is not established

- **Any advantage of the local rule.** Where the sweep differs from backpropagation it is weaker. The original hypothesis, that a local rule generalises better, is not supported.
- **Whether a lost family is destroyed or only not expressed** when the later learning is unrelated. In E43 every old answer became the end symbol, which a whole-answer score cannot interpret.
- **Anything over a long life.** The small world built for that failed twice: its tasks had too few examples for a learner without experience. Its replacement is drafted (E44).
- **Scale.** Width and data as axes are measured on one seed so far (E41).

## Open items

| Item | State |
|---|---|
| E43 additions: weight files, a reminder, plain gradient steps | released; decides where the loss in E43 occurs |
| E41, seeds 93683 and 116089 | released; no worker assigned |
| E42, cells with limited local connections and periods | running; first seed being checked by the reviewer |
| E44, the long-life world in the full-scale format | drafted; audit running |
| Kolen–Pollack result | a candidate on three seeds, unreplicated |
| E34's later tables and E41's first seed | merged; not yet read by me in detail |

## Direction (adopted by Mr. Xu, 16 November)

Retention is at the centre, and the question is restated: under fixed memory and continued learning, can a learner whose changes are confined to the connections in use keep what it has learned while learning new things, at no greater computation to threshold than backpropagation? Two conditions of mine stand with it: the comparison includes backpropagation with a plain replay store at equal memory; and weight decay is dealt with first, since it alone removes unused knowledge (E46, drafted).
