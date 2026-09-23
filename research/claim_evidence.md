# Claim and evidence ledger

| Claim | Current evidence | Status |
| --- | --- | --- |
| F, St, S tangent maps and exact returns obey small-matrix formulas | 44 local CPU tests | checked locally |
| Practical F and St steps avoid full SVD | monkeypatched local step tests | checked locally |
| Practical Stiefel full `msign` is close to orthogonal | 384x384 local synthetic residual about `2e-5` before dimension normalization | preliminary |
| Practical spectral PI10/top-8 has acceptable constraint error | 128x128 synthetic relative defect about `0.015` | not supported |
| FP32 master survives Megatron BF16 wrapping and checkpoint resume | direct metadata and optimizer state tests only; full H20 integration pending | open |
| Any method improves held-out NLL or GPU-hours | no matched long-run result in this branch | not tested |
