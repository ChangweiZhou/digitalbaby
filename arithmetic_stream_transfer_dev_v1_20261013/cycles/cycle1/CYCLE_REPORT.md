# Cycle 1

## Design

Same generic recurrent core for counting, addition and multiplication; four objective/decay recipes.

## Audit

All-byte loss, bounded per-target gradients, actual EOS update, pure generation and no-write including decay.

## Test run

12 numerical/interface tests; 4 stream fixtures / 104 external applications including disabled records.

## Revise

Implement bounded online per-target credit for long counting strings; task changes never reset parameters or optimizer.

All execution in this cycle is technical qualification, not formal DEV or confirmation science.
