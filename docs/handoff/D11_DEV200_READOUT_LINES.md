# D11 dev200 readout lines (Amendment 5) — SELF-TEST, not a decision

Written by `eval/error_analysis_tests/efbpt/d11_eval458_readout.py --set dev200` (MD5 `8fb00396fc676a1066d924dfdb3ed874`); Amendment 5 `45c118de9804390d12853f231e7632a0`; E3 from `d11_scorer_audit.py` `246cf6c9687b3fa67324722cb5cd16b2` (audit commit e3fa2ca); E0 = the frozen extractor. n = 200 (gold yes 87 / no 113). b = the first-named arm right and the second wrong; c = the reverse; p = exact two-sided McNemar; unparsed = wrong.

| arm | file | MD5 | seal |
|---|---|---|---|
| A_r2 | `outputs/efbpt/d10/student/d10_armA_r2.jsonl` | `7ccec942c8f0eee08156d258baaf806d` | MD5 pinned (readout §1) |
| TR | `outputs/efbpt/d11/student/dev200/d11_armTR.jsonl` | `bab987cc492012f4e32dbcd348be9668` | MD5 pinned (readout §1) |
| TR_wrong | `outputs/efbpt/d11/student/dev200/d11_armTR_wrong.jsonl` | `9519b9adf215ac90eb6de743ec1dca57` | MD5 pinned (readout §1) |
| TR_L1 | `outputs/efbpt/d11/student/dev200/d11_armTR_L1.jsonl` | `b8db599f8c4b427ddcb2d772fd8f09aa` | MD5 pinned (readout §1) |

Prompts: 800 sealed prompts equal the prompts rebuilt from the sealed teacher facts.

## Per arm, under E0 and E3 (A5.3)

| arm | scorer | acc % (correct) | unparsed (%) | predicted yes % | acc on gold-yes % | acc on gold-no % | balanced acc % |
|---|---|---|---|---|---|---|---|
| A_r2 | E0 | 60.00 (120) | 8 (4.00) | 41.00 | 54.02 | 64.60 | 59.31 |
| A_r2 | E3 | 55.00 (110) | 10 (5.00) | 54.50 | 64.37 | 47.79 | 56.08 |
| TR | E0 | 68.50 (137) | 3 (1.50) | 33.50 | 52.87 | 80.53 | 66.70 |
| TR | E3 | 71.50 (143) | 4 (2.00) | 44.00 | 68.97 | 73.45 | 71.21 |
| TR_wrong | E0 | 50.50 (101) | 11 (5.50) | 24.00 | 22.99 | 71.68 | 47.33 |
| TR_wrong | E3 | 50.50 (101) | 11 (5.50) | 29.00 | 28.74 | 67.26 | 48.00 |
| TR_L1 | E0 | 67.00 (134) | 2 (1.00) | 30.00 | 47.13 | 82.30 | 64.71 |
| TR_L1 | E3 | 70.00 (140) | 2 (1.00) | 44.00 | 66.67 | 72.57 | 69.62 |

Balanced accuracy = mean of the gold-yes and gold-no accuracies (descriptive; separates knowledge from a shift to "no"). Rows where E0 and E3 differ: A_r2 36; TR 27; TR_wrong 14; TR_L1 36.

## Comparisons, four lines each (A5.3)

**TR vs A_r2 — PRIMARY (gate)**

| line | n | acc TR | acc A_r2 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  E0, all rows (pre-registered gate) | 200 | 68.50 | 60.00 | +8.50 | 39 | 22 | 0.03962 |
| L2  E3, all rows (validated scorer) | 200 | 71.50 | 55.00 | +16.50 | 57 | 24 | 0.0003173 |
| L3  E0, rows parsed in both arms (A3.1) | 189 | 69.84 | 63.49 | +6.35 | 34 | 22 | 0.1409 |
| L4  rows where E0 = E3 in both arms (descriptive) | 143 | 74.83 | 64.34 | +10.49 | 30 | 15 | 0.0357 |

**TR vs TR_wrong — CONTENT CONTROL (A5.5)**

| line | n | acc TR | acc TR_wrong | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  E0, all rows (pre-registered gate) | 200 | 68.50 | 50.50 | +18.00 | 53 | 17 | 1.92e-05 |
| L2  E3, all rows (validated scorer) | 200 | 71.50 | 50.50 | +21.00 | 62 | 20 | 3.715e-06 |
| L3  E0, rows parsed in both arms (A3.1) | 187 | 68.45 | 53.48 | +14.97 | 44 | 16 | 0.000394 |
| L4  rows where E0 = E3 in both arms (descriptive) | 161 | 73.91 | 53.42 | +20.50 | 45 | 12 | 1.313e-05 |

**TR_wrong vs A_r2 — wrong facts vs floor (reported, A5.5)**

| line | n | acc TR_wrong | acc A_r2 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  E0, all rows (pre-registered gate) | 200 | 50.50 | 60.00 | -9.50 | 30 | 49 | 0.04217 |
| L2  E3, all rows (validated scorer) | 200 | 50.50 | 55.00 | -4.50 | 37 | 46 | 0.38 |
| L3  E0, rows parsed in both arms (A3.1) | 181 | 54.14 | 61.88 | -7.73 | 27 | 41 | 0.1143 |
| L4  rows where E0 = E3 in both arms (descriptive) | 157 | 50.96 | 59.24 | -8.28 | 25 | 38 | 0.1299 |

**TR_L1 vs A_r2 — SECONDARY (A2.2, A3.2)**

| line | n | acc TR_L1 | acc A_r2 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  E0, all rows (pre-registered gate) | 200 | 67.00 | 60.00 | +7.00 | 45 | 31 | 0.1354 |
| L2  E3, all rows (validated scorer) | 200 | 70.00 | 55.00 | +15.00 | 60 | 30 | 0.00206 |
| L3  E0, rows parsed in both arms (A3.1) | 190 | 68.42 | 62.63 | +5.79 | 41 | 30 | 0.2351 |
| L4  rows where E0 = E3 in both arms (descriptive) | 136 | 72.79 | 62.50 | +10.29 | 34 | 20 | 0.0759 |

## A3.1 and verdict tier (A5.4) — SELF-TEST on DEV200: the tier logic is exercised; the DEV200 decision is the freeze §7 GO

- A3.1 primary: |unparsed E0 TR − A_r2| = |1.50 − 4.00| = 2.50 pp ≤ 5.0 → the pre-registered gate is read on **L1**.
- A3.1 control: unparsed E0 TR_wrong 5.50 % vs A_r2 4.00 % (difference 1.50 pp); it never changes the primary.
- Gate (≥ +5.0 pp and p < 0.05) on L1: Δ +8.50 pp, b 39 / c 22, p = 0.03962 → **PASS**.
- Validated scorer, L2: Δ +16.50 pp, b 57 / c 24, p = 0.0003173 → passes.

**VERDICT (self-test label only): CONFIRMED**

- Secondary TR_L1 vs A_r2 (A3.2: p < 0.05 with a positive delta; tested in tier 1 on L1 (A3.1: unparsed difference 3.00 pp)): Δ +7.00 pp, b 45 / c 31, p = 0.1354 → **not significant**.
- Content control TR vs TR_wrong (A5.5): L1 Δ +18.00 pp (b 53 / c 17, p = 1.92e-05); L2 Δ +21.00 pp (b 62 / c 20, p = 3.715e-06).
- TR_wrong vs A_r2 (reported): L1 Δ -9.50 pp (p = 0.04217); L2 Δ -4.50 pp (p = 0.38). A5.5: under E3 on DEV200 this is −4.5 pp (p 0.38), so "wrong facts harm the baseline" is not claimed under E3.

## A2.6 predictions (written for eval458; shown on DEV200 only to exercise the table)

| prediction | observed, E0 | E0 | observed, E3 (descriptive) | E3 |
|---|---|---|---|---|
| A_r2 accuracy in [55, 63] | 60.00 | hit | 55.00 | hit |
| TR − A_r2 in [+5, +10] | +8.50 | hit | +16.50 | miss |
| TR_wrong − A_r2 ≤ −3 | -9.50 | hit | -4.50 | hit |
| TR_L1 − A_r2 in [+4, +9] | +7.00 | hit | +15.00 | miss |
| gain on gold-no rows > gain on gold-yes rows (TR vs A_r2) | gold-no +15.93, gold-yes -1.15 | hit | gold-no +25.66, gold-yes +4.60 | hit |

DEV200 self-test PASS: L1 TR vs A_r2 = 68.5 / 60.0, b 39 / c 22, p 0.0396 (sealed); L2 = 71.5 / 55.0, b 57 / c 24 (A4.4); TR_L1 and TR_wrong L1 / L2 pairs, TR vs TR_wrong L1, L3 n / b / c (supplement S2) and the unparsed counts all as sealed.
