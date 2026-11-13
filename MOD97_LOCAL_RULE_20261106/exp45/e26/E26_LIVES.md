# E26 stage 3: the six lives

Written by verify_lives.py from the receipts. Percent whole answers (`exact`). within = new instances of taught pairs (600); each held pair 600; plus-minus = +- and -+ pooled (1,200); F1/F3/dev89 = family means at k=12; hard89 = add89/sub89 mean at k=12.

## Checks

- S_73129: {'complete_record': True, 'seven_stages': True, 'optimizer_steps_36864': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'birth_is_admitted_birth': True, 'sources_equal_360e1b8d': True, 'arm_local': True, 'finite_losses_every_stage': True, 'weight_decay_read_back': [0.25, 0.0], 'taught_panel_only_taught_presented': True}
- B_73129: {'complete_record': True, 'seven_stages': True, 'optimizer_steps_36864': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'birth_is_admitted_birth': True, 'sources_equal_360e1b8d': True, 'arm_local': True, 'finite_losses_every_stage': True, 'weight_decay_read_back': [0.25, 0.0], 'taught_panel_only_taught_presented': True}
- data_digest_S_equals_B_73129: True
- S_93683: {'complete_record': True, 'seven_stages': True, 'optimizer_steps_36864': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'birth_is_admitted_birth': True, 'sources_equal_360e1b8d': True, 'arm_local': True, 'finite_losses_every_stage': True, 'weight_decay_read_back': [0.25, 0.0], 'taught_panel_only_taught_presented': True}
- B_93683: {'complete_record': True, 'seven_stages': True, 'optimizer_steps_36864': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'birth_is_admitted_birth': True, 'sources_equal_360e1b8d': True, 'arm_local': True, 'finite_losses_every_stage': True, 'weight_decay_read_back': [0.25, 0.0], 'taught_panel_only_taught_presented': True}
- data_digest_S_equals_B_93683: True
- S_116089: {'complete_record': True, 'seven_stages': True, 'optimizer_steps_36864': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'birth_is_admitted_birth': True, 'sources_equal_360e1b8d': True, 'arm_local': True, 'finite_losses_every_stage': True, 'weight_decay_read_back': [0.25, 0.0], 'taught_panel_only_taught_presented': True}
- B_116089: {'complete_record': True, 'seven_stages': True, 'optimizer_steps_36864': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'birth_is_admitted_birth': True, 'sources_equal_360e1b8d': True, 'arm_local': True, 'finite_losses_every_stage': True, 'weight_decay_read_back': [0.25, 0.0], 'taught_panel_only_taught_presented': True}
- data_digest_S_equals_B_116089: True

## Readings

- P26.3: {"values": [83.5, 83.83333333333333, 81.0], "seeds_meeting": [], "holds": false}
- P26.4: {"values": [54.25, 44.5, 37.25], "seeds_meeting": [73129, 93683], "holds": true}
- P26.5: {"within": [84.66666666666667, 93.66666666666667, 87.16666666666667], "plus_minus": [63.25, 89.16666666666667, 66.33333333333333], "seeds_meeting_both": [93683, 116089], "holds": true}
- candidate_pair_S: {"seeds_meeting_both": [], "exists": false}
- paired_S_minus_B_end: {"73129": {"within": -1.1666666666666714, "plus_minus": -9.0}, "93683": {"within": -9.833333333333343, "plus_minus": -44.66666666666667}, "116089": {"within": -6.166666666666671, "plus_minus": -29.08333333333333}}

## Life checkpoints

| arm | seed | stage | taught | within | +- | -+ | >< | <> | plus-minus | F1 | F3 | dev89 | hard89 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S | 73129 | 0 birth | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 46.9 | 47.7 | 98.5 | 98.0 |
| S | 73129 | 1 F1.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 95.8 | 42.8 | 72.3 | 49.0 |
| S | 73129 | 2 F1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 95.2 | 45.9 | 71.5 | 47.8 |
| S | 73129 | 3 F2.1 | 80.2 | 78.0 | 35.7 | 34.5 | 81.7 | 70.0 | 35.1 | 91.4 | 31.2 | 62.2 | 44.2 |
| S | 73129 | 4 F2.2 | 88.2 | 82.2 | 48.0 | 53.5 | 74.2 | 79.0 | 50.8 | 95.0 | 21.3 | 53.8 | 33.2 |
| S | 73129 | 5 F3.1 | 87.0 | 83.5 | 46.8 | 59.2 | 83.3 | 79.7 | 53.0 | 94.2 | 90.8 | 55.3 | 31.0 |
| S | 73129 | 6 F3.2 | 87.5 | 83.5 | 49.3 | 59.2 | 81.0 | 76.5 | 54.2 | 95.7 | 89.2 | 55.5 | 33.0 |
| B | 73129 | 0 birth | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 23.8 | 9.6 | 98.6 | 98.5 |
| B | 73129 | 1 F1.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 95.5 | 17.9 | 53.3 | 44.2 |
| B | 73129 | 2 F1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 96.6 | 15.9 | 51.0 | 47.5 |
| B | 73129 | 3 F2.1 | 80.0 | 75.2 | 50.2 | 51.3 | 73.8 | 75.5 | 50.8 | 95.5 | 27.3 | 46.1 | 35.5 |
| B | 73129 | 4 F2.2 | 84.0 | 77.8 | 60.7 | 60.7 | 88.5 | 73.5 | 60.7 | 93.5 | 36.1 | 51.2 | 34.5 |
| B | 73129 | 5 F3.1 | 83.5 | 76.2 | 42.8 | 63.2 | 86.3 | 83.5 | 53.0 | 95.1 | 91.1 | 58.8 | 40.2 |
| B | 73129 | 6 F3.2 | 90.5 | 84.7 | 60.5 | 66.0 | 91.2 | 86.5 | 63.2 | 97.4 | 92.5 | 57.9 | 37.3 |
| S | 93683 | 0 birth | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 15.0 | 23.8 | 98.4 | 98.2 |
| S | 93683 | 1 F1.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 98.6 | 18.4 | 30.8 | 6.8 |
| S | 93683 | 2 F1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 98.0 | 22.6 | 33.3 | 2.0 |
| S | 93683 | 3 F2.1 | 80.5 | 78.5 | 37.7 | 36.7 | 75.3 | 59.0 | 37.2 | 94.2 | 26.1 | 42.8 | 1.0 |
| S | 93683 | 4 F2.2 | 90.5 | 86.7 | 50.3 | 66.0 | 78.8 | 68.8 | 58.2 | 96.6 | 23.2 | 44.2 | 0.2 |
| S | 93683 | 5 F3.1 | 90.8 | 85.7 | 52.7 | 59.2 | 72.3 | 69.0 | 55.9 | 93.9 | 87.6 | 48.6 | 11.7 |
| S | 93683 | 6 F3.2 | 83.2 | 83.8 | 42.7 | 46.3 | 69.8 | 73.3 | 44.5 | 94.6 | 91.9 | 39.9 | 17.7 |
| B | 93683 | 0 birth | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7.3 | 23.7 | 98.5 | 98.7 |
| B | 93683 | 1 F1.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 95.1 | 14.2 | 5.8 | 1.2 |
| B | 93683 | 2 F1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 96.6 | 16.8 | 7.2 | 1.0 |
| B | 93683 | 3 F2.1 | 89.5 | 86.3 | 67.7 | 71.8 | 84.7 | 80.5 | 69.8 | 95.9 | 21.2 | 44.2 | 0.7 |
| B | 93683 | 4 F2.2 | 96.0 | 93.8 | 85.3 | 87.7 | 92.5 | 87.7 | 86.5 | 98.5 | 5.2 | 46.5 | 0.0 |
| B | 93683 | 5 F3.1 | 95.2 | 93.0 | 86.2 | 83.8 | 93.3 | 93.0 | 85.0 | 98.8 | 96.2 | 59.2 | 23.2 |
| B | 93683 | 6 F3.2 | 97.3 | 93.7 | 88.2 | 90.2 | 91.7 | 90.7 | 89.2 | 98.6 | 96.2 | 52.7 | 26.7 |
| S | 116089 | 0 birth | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 9.6 | 26.9 | 98.5 | 99.8 |
| S | 116089 | 1 F1.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 96.9 | 27.4 | 62.1 | 34.0 |
| S | 116089 | 2 F1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 97.9 | 25.4 | 62.0 | 29.3 |
| S | 116089 | 3 F2.1 | 79.7 | 70.7 | 16.0 | 27.0 | 57.5 | 77.5 | 21.5 | 95.7 | 45.3 | 34.1 | 9.2 |
| S | 116089 | 4 F2.2 | 90.2 | 82.2 | 21.2 | 35.2 | 66.3 | 82.2 | 28.2 | 95.3 | 43.4 | 36.6 | 3.2 |
| S | 116089 | 5 F3.1 | 86.3 | 82.3 | 26.5 | 48.0 | 85.3 | 81.3 | 37.2 | 95.4 | 93.5 | 49.4 | 14.0 |
| S | 116089 | 6 F3.2 | 83.3 | 81.0 | 37.2 | 37.3 | 78.7 | 80.0 | 37.2 | 96.0 | 95.1 | 45.9 | 13.5 |
| B | 116089 | 0 birth | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 | 17.9 | 96.1 | 93.8 |
| B | 116089 | 1 F1.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 96.6 | 14.6 | 62.5 | 32.7 |
| B | 116089 | 2 F1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 96.5 | 14.8 | 58.9 | 22.7 |
| B | 116089 | 3 F2.1 | 90.0 | 86.2 | 66.2 | 68.8 | 83.5 | 89.8 | 67.5 | 95.8 | 55.4 | 41.7 | 34.0 |
| B | 116089 | 4 F2.2 | 91.2 | 87.8 | 64.7 | 77.5 | 77.8 | 87.7 | 71.1 | 96.9 | 61.8 | 38.2 | 26.3 |
| B | 116089 | 5 F3.1 | 91.3 | 87.8 | 72.2 | 65.8 | 79.0 | 78.3 | 69.0 | 97.0 | 94.8 | 49.5 | 38.0 |
| B | 116089 | 6 F3.2 | 88.2 | 87.2 | 68.3 | 64.3 | 76.7 | 84.2 | 66.3 | 95.1 | 94.9 | 42.6 | 36.3 |

## Paired S minus B by checkpoint (within / plus-minus)

| stage | 73129 | 93683 | 116089 |
|---|---|---|---|
| 0 | +0.0 / +0.0 | +0.0 / +0.0 | +0.0 / +0.0 |
| 1 | +0.0 / +0.0 | +0.0 / +0.0 | +0.0 / +0.0 |
| 2 | +0.0 / +0.0 | +0.0 / +0.0 | +0.0 / +0.0 |
| 3 | +2.8 / -15.7 | -7.8 / -32.6 | -15.5 / -46.0 |
| 4 | +4.3 / -9.9 | -7.2 / -28.3 | -5.7 / -42.9 |
| 5 | +7.3 / +0.0 | -7.3 / -29.1 | -5.5 / -31.8 |
| 6 | -1.2 / -9.0 | -9.8 / -44.7 | -6.2 / -29.1 |

## Silenced readout, composition, end of life

- S 73129: {"+-": 0.0, "-+": 0.0, "><": 0.0, "<>": 0.0}
- B 73129: {"+-": 0.0, "-+": 0.0, "><": 0.0, "<>": 0.0}
- S 93683: {"+-": 0.0, "-+": 0.0, "><": 0.0, "<>": 0.0}
- B 93683: {"+-": 0.0, "-+": 0.0, "><": 0.0, "<>": 0.0}
- S 116089: {"+-": 0.0, "-+": 0.0, "><": 0.0, "<>": 0.0}
- B 116089: {"+-": 0.0, "-+": 0.0, "><": 0.0, "<>": 0.0}

## CPU seconds

| life | total | updates | scoring |
|---|---|---|---|
| S 73129 | 5161 | 4787 | 374 |
| B 73129 | 4869 | 4466 | 403 |
| S 93683 | 5163 | 4771 | 392 |
| B 93683 | 4868 | 4447 | 421 |
| S 116089 | 5103 | 4723 | 381 |
| B 116089 | 4809 | 4425 | 384 |
