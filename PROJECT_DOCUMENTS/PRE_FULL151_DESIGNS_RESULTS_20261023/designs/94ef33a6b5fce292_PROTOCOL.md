# V49 protocol

Version `v49.0.0-raw-banc-mb-core`.
Design SHA256 `40c5c8649c56a6fe6c1cb7ae4d76b7538ef2db9f1461170e03461e1ee505f878`.

Primary evidence source: embedded blocks extracted directly from the supplied BANC v888 Feather files.

Block A tests PN→KC/KC→MBON identity alignment, support topology and count organization.
Block B tests raw non-self MBON recurrence separately.

Fixed output-code normalization: residual multiplied by sqrt(104); eta=4.0; tau=0.25.
Block B recurrence is linear h=f+Rh with spectral radius 0.25.
Direct interference audit uses target-byte acquisition writes only.
