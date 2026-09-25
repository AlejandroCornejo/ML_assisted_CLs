# Validation-only audit: multicavity feature-count sweep

This frozen audit consolidates the 84 constrained fits at `m = 2, 4, 6, 8, 16, 24, 32`. It reads final training reports and checkpoint hashes only; it does not load test or reserved-path labels, model weights, or FOM predictions.

The original unstarted `m=40` block was replaced prospectively by the low-count amendment (`m=2,4,6`) using validation-only observations. The retained `m=8,16,24,32` fits use the legacy frozen rule; all low-count fits use the amendment rule. This distinction is retained below.

- Validated terminal fits: **84 / 84**
- Unstarted/cancelled `m=40` fits: **12**
- Test labels loaded by any fit: **no**
- Reserved path labels loaded by any fit: **no**

## Median validation stress error by model and feature count

Lower is better. Ranges span the three declared seeds.

| Model | `m` | Median | Min--max | Median Adam steps | Median wall time (h) |
|---|---:|---:|---:|---:|---:|
| ICNN-fixed | 2 | 4.616e-03 | 4.616e-03--4.616e-03 | 6780 | 0.09 |
| ICNN-fixed | 4 | 9.115e-05 | 9.097e-05--9.120e-05 | 10530 | 0.12 |
| ICNN-fixed | 6 | 1.057e-05 | 1.032e-05--1.502e-05 | 54510 | 0.58 |
| ICNN-fixed | 8 | 1.037e-05 | 9.663e-06--1.097e-05 | 61020 | 0.53 |
| ICNN-fixed | 16 | 7.014e-06 | 6.299e-06--7.053e-06 | 50000 | 0.67 |
| ICNN-fixed | 24 | 2.108e-06 | 2.074e-06--2.143e-06 | 93640 | 1.58 |
| ICNN-fixed | 32 | 1.904e-06 | 1.849e-06--1.975e-06 | 134940 | 2.98 |
| ICNN-learned | 2 | 1.422e-05 | 1.258e-05--2.950e-05 | 32350 | 0.39 |
| ICNN-learned | 4 | 4.449e-06 | 3.314e-06--5.254e-06 | 44800 | 0.54 |
| ICNN-learned | 6 | 2.048e-07 | 1.785e-07--2.715e-07 | 200000 | 2.48 |
| ICNN-learned | 8 | 1.595e-07 | 1.572e-07--1.785e-07 | 200000 | 2.11 |
| ICNN-learned | 16 | 1.320e-07 | 1.317e-07--1.325e-07 | 200000 | 3.16 |
| ICNN-learned | 24 | 1.108e-07 | 1.009e-07--1.119e-07 | 200000 | 4.18 |
| ICNN-learned | 32 | 1.102e-07 | 1.055e-07--1.142e-07 | 200000 | 5.62 |
| ICKAN-fixed | 2 | 3.648e-03 | 3.646e-03--3.648e-03 | 10750 | 0.61 |
| ICKAN-fixed | 4 | 6.975e-05 | 6.950e-05--7.000e-05 | 10440 | 0.58 |
| ICKAN-fixed | 6 | 8.927e-06 | 8.862e-06--9.089e-06 | 42210 | 2.18 |
| ICKAN-fixed | 8 | 8.661e-06 | 8.646e-06--8.764e-06 | 61240 | 2.78 |
| ICKAN-fixed | 16 | 5.726e-06 | 5.689e-06--5.785e-06 | 72330 | 5.51 |
| ICKAN-fixed | 24 | 1.931e-06 | 1.871e-06--2.007e-06 | 86050 | 8.31 |
| ICKAN-fixed | 32 | 1.772e-06 | 1.641e-06--1.813e-06 | 119160 | 13.98 |
| ICKAN-learned | 2 | 1.284e-05 | 1.276e-05--1.300e-05 | 39250 | 2.23 |
| ICKAN-learned | 4 | 2.718e-06 | 1.780e-06--2.724e-06 | 58150 | 3.29 |
| ICKAN-learned | 6 | 1.712e-07 | 1.642e-07--1.807e-07 | 200000 | 6.43 |
| ICKAN-learned | 8 | 1.619e-07 | 1.473e-07--1.666e-07 | 195180 | 10.30 |
| ICKAN-learned | 16 | 1.375e-07 | 1.238e-07--1.556e-07 | 200000 | 15.43 |
| ICKAN-learned | 24 | 1.142e-07 | 1.103e-07--1.197e-07 | 200000 | 20.42 |
| ICKAN-learned | 32 | 1.309e-07 | 1.143e-07--1.496e-07 | 200000 | 21.06 |

## Paired learned/fixed validation ratio

Each ratio pairs the same core and seed. A value below one favors learned directional features.

| Core | `m` | Median learned/fixed | Min--max |
|---|---:|---:|---:|
| ICNN | 2 | 0.00308 | 0.00273--0.00639 |
| ICNN | 4 | 0.0488 | 0.0364--0.0576 |
| ICNN | 6 | 0.0181 | 0.0169--0.0198 |
| ICNN | 8 | 0.0163 | 0.0154--0.0163 |
| ICNN | 16 | 0.0188 | 0.0187--0.021 |
| ICNN | 24 | 0.0522 | 0.0479--0.0534 |
| ICNN | 32 | 0.0578 | 0.057--0.0579 |
| ICKAN | 2 | 0.00352 | 0.0035--0.00356 |
| ICKAN | 4 | 0.0389 | 0.0255--0.0391 |
| ICKAN | 6 | 0.0192 | 0.0181--0.0204 |
| ICKAN | 8 | 0.0185 | 0.017--0.0192 |
| ICKAN | 16 | 0.024 | 0.0214--0.0273 |
| ICKAN | 24 | 0.0591 | 0.055--0.064 |
| ICKAN | 32 | 0.0739 | 0.063--0.0911 |

## Scope

This document is a training/validation audit, not the independent-test or loading-path analysis. It must not be used to claim generalization or to select a final reported model from test outcomes.
