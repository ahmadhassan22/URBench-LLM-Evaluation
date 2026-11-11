# D11 eval458 readout lines (Amendment 5)

Written by `eval/error_analysis_tests/efbpt/d11_eval458_readout.py --set eval458` (MD5 `8fb00396fc676a1066d924dfdb3ed874`); Amendment 5 `45c118de9804390d12853f231e7632a0`; E3 from `d11_scorer_audit.py` `246cf6c9687b3fa67324722cb5cd16b2` (audit commit e3fa2ca); E0 = the frozen extractor. n = 458 (gold yes 221 / no 237). b = the first-named arm right and the second wrong; c = the reverse; p = exact two-sided McNemar; unparsed = wrong.

| arm | file | MD5 | seal |
|---|---|---|---|
| A_r2_eval | `outputs/efbpt/d11/student/eval458/d11_armA_r2_eval.jsonl` | `f139ba9247cdd9fef191f00d404d8212` | outputs/efbpt/d11/student/eval458/S_SEAL_A_r2_eval.json (md5 245130c9adaaee0471000e5f9331652a; slurm 115529; 2026-11-10 16:26:39; 2 files verified) |
| TR | `outputs/efbpt/d11/student/eval458/d11_armTR.jsonl` | `e38d64114089a88b2deb202de0240ca9` | outputs/efbpt/d11/student/eval458/S_SEAL_TR.json (md5 e5deadcc0bc732fd6081e581ccee8f72; slurm 115575; 2026-11-10 19:07:03; 2 files verified) |
| TR_wrong_eval | `outputs/efbpt/d11/student/eval458/d11_armTR_wrong_eval.jsonl` | `39d6200fec7ad4016f9e4411f213eb10` | outputs/efbpt/d11/student/eval458/S_SEAL_TR_wrong_eval.json (md5 6352a4e242b8a86cb2c3c57d938b62f2; slurm 115576; 2026-11-10 20:45:36; 3 files verified) |
| TR_L1_eval | `outputs/efbpt/d11/student/eval458/d11_armTR_L1_eval.jsonl` | `4a1b2118bbcaa9348713a600b9281733` | outputs/efbpt/d11/student/eval458/S_SEAL_TR_L1_eval.json (md5 b175e5b851a5b39c95a428a8285e17a3; slurm 115577; 2026-11-10 23:08:25; 2 files verified) |

Prompts: 1832 sealed prompts equal the prompts rebuilt from the sealed teacher facts.

## Per arm, under E0 and E3 (A5.3)

| arm | scorer | acc % (correct) | unparsed (%) | predicted yes % | acc on gold-yes % | acc on gold-no % | balanced acc % |
|---|---|---|---|---|---|---|---|
| A_r2_eval | E0 | 54.80 (251) | 15 (3.28) | 45.63 | 52.04 | 57.38 | 54.71 |
| A_r2_eval | E3 | 55.46 (254) | 19 (4.15) | 59.61 | 67.42 | 44.30 | 55.86 |
| TR | E0 | 68.78 (315) | 6 (1.31) | 31.66 | 51.13 | 85.23 | 68.18 |
| TR | E3 | 74.24 (340) | 8 (1.75) | 44.32 | 70.14 | 78.06 | 74.10 |
| TR_wrong_eval | E0 | 52.62 (241) | 22 (4.80) | 23.14 | 27.60 | 75.95 | 51.78 |
| TR_wrong_eval | E3 | 53.71 (246) | 26 (5.68) | 32.10 | 38.46 | 67.93 | 53.20 |
| TR_L1_eval | E0 | 64.63 (296) | 7 (1.53) | 33.84 | 49.77 | 78.48 | 64.13 |
| TR_L1_eval | E3 | 70.96 (325) | 10 (2.18) | 46.29 | 69.68 | 72.15 | 70.92 |

Balanced accuracy = mean of the gold-yes and gold-no accuracies (descriptive; separates knowledge from a shift to "no"). Rows where E0 and E3 differ: A_r2_eval 81; TR 69; TR_wrong_eval 60; TR_L1_eval 70.

## Comparisons, four lines each (A5.3)

**TR vs A_r2_eval — PRIMARY (gate)**

| line | n | acc TR | acc A_r2_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  E0, all rows (pre-registered gate) | 458 | 68.78 | 54.80 | +13.97 | 117 | 53 | 1.021e-06 |
| L2  E3, all rows (validated scorer) | 458 | 74.24 | 55.46 | +18.78 | 133 | 47 | 1.042e-10 |
| L3  E0, rows parsed in both arms (A3.1) | 437 | 70.02 | 56.06 | +13.96 | 108 | 47 | 1.059e-06 |
| L4  rows where E0 = E3 in both arms (descriptive) | 327 | 76.76 | 55.96 | +20.80 | 93 | 25 | 2.127e-10 |

**TR vs TR_wrong_eval — CONTENT CONTROL (A5.5)**

| line | n | acc TR | acc TR_wrong_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  E0, all rows (pre-registered gate) | 458 | 68.78 | 52.62 | +16.16 | 111 | 37 | 8.747e-10 |
| L2  E3, all rows (validated scorer) | 458 | 74.24 | 53.71 | +20.52 | 138 | 44 | 1.708e-12 |
| L3  E0, rows parsed in both arms (A3.1) | 431 | 69.84 | 55.45 | +14.39 | 97 | 35 | 6.358e-08 |
| L4  rows where E0 = E3 in both arms (descriptive) | 337 | 75.37 | 55.19 | +20.18 | 88 | 20 | 2.322e-11 |

**TR_wrong_eval vs A_r2_eval — wrong facts vs floor (reported, A5.5)**

| line | n | acc TR_wrong_eval | acc A_r2_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  E0, all rows (pre-registered gate) | 458 | 52.62 | 54.80 | -2.18 | 94 | 104 | 0.5225 |
| L2  E3, all rows (validated scorer) | 458 | 53.71 | 55.46 | -1.75 | 90 | 98 | 0.6098 |
| L3  E0, rows parsed in both arms (A3.1) | 425 | 55.76 | 56.47 | -0.71 | 90 | 93 | 0.8825 |
| L4  rows where E0 = E3 in both arms (descriptive) | 329 | 54.10 | 54.71 | -0.61 | 66 | 68 | 0.9312 |

**TR_L1_eval vs A_r2_eval — SECONDARY (A2.2, A3.2)**

| line | n | acc TR_L1_eval | acc A_r2_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  E0, all rows (pre-registered gate) | 458 | 64.63 | 54.80 | +9.83 | 107 | 62 | 0.0006677 |
| L2  E3, all rows (validated scorer) | 458 | 70.96 | 55.46 | +15.50 | 126 | 55 | 1.37e-07 |
| L3  E0, rows parsed in both arms (A3.1) | 437 | 65.68 | 56.29 | +9.38 | 98 | 57 | 0.001241 |
| L4  rows where E0 = E3 in both arms (descriptive) | 329 | 71.43 | 55.62 | +15.81 | 85 | 33 | 1.843e-06 |

## A3.1 and verdict tier (A5.4)

- A3.1 primary: |unparsed E0 TR − A_r2_eval| = |1.31 − 3.28| = 1.97 pp ≤ 5.0 → the pre-registered gate is read on **L1**.
- A3.1 control: unparsed E0 TR_wrong_eval 4.80 % vs A_r2_eval 3.28 % (difference 1.53 pp); it never changes the primary.
- Gate (≥ +5.0 pp and p < 0.05) on L1: Δ +13.97 pp, b 117 / c 53, p = 1.021e-06 → **PASS**.
- Validated scorer, L2: Δ +18.78 pp, b 133 / c 47, p = 1.042e-10 → passes.

**VERDICT: CONFIRMED**

A2.7: no second eval458 run in any case.
- Secondary TR_L1_eval vs A_r2_eval (A3.2: p < 0.05 with a positive delta; tested in tier 1 on L1 (A3.1: unparsed difference 1.75 pp)): Δ +9.83 pp, b 107 / c 62, p = 0.0006677 → **SIGNIFICANT**.
- Content control TR vs TR_wrong_eval (A5.5): L1 Δ +16.16 pp (b 111 / c 37, p = 8.747e-10); L2 Δ +20.52 pp (b 138 / c 44, p = 1.708e-12).
- TR_wrong_eval vs A_r2_eval (reported): L1 Δ -2.18 pp (p = 0.5225); L2 Δ -1.75 pp (p = 0.6098). A5.5: under E3 on DEV200 this is −4.5 pp (p 0.38), so "wrong facts harm the baseline" is not claimed under E3.

## A2.6 predictions

| prediction | observed, E0 | E0 | observed, E3 (descriptive) | E3 |
|---|---|---|---|---|
| A_r2_eval accuracy in [55, 63] | 54.80 | miss | 55.46 | hit |
| TR − A_r2_eval in [+5, +10] | +13.97 | miss | +18.78 | miss |
| TR_wrong_eval − A_r2_eval ≤ −3 | -2.18 | miss | -1.75 | miss |
| TR_L1_eval − A_r2_eval in [+4, +9] | +9.83 | miss | +15.50 | miss |
| gain on gold-no rows > gain on gold-yes rows (TR vs A_r2_eval) | gold-no +27.85, gold-yes -0.90 | hit | gold-no +33.76, gold-yes +2.71 | hit |
