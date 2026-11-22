# D15 readout — reading effect and teacher facts on native Urdu questions (UrduMMLU)

Written by `eval/error_analysis_tests/efbpt/d15_readout.py` (MD5 `766b1408c05cc8ebf9968f67c89f6180`, git HEAD `e4a4c8b8360e6c01f6ec3dc820c28d7ad9e208f4`), one run, 2026-11-22 20:56:18. Freeze: `docs/EFBPT_D15_NATIVE_URDU_FREEZE.md` (MD5 `86cfd8ff6f994def4f801d68543a16ef`). Sample: `eval/error_analysis_tests/efbpt/d15_sample_ids.txt` (MD5 `b430f4ee354ddfa3cc5fd8df71f96c69`), 2600 rows. Every arm, translation and facts file was sealed before this script opened it; no D15 arm was scored before this run. No arm is run again after this readout.

Data: UrduMMLU (MBZUAI/UrduMMLU, CC BY 4.0), revision `aae060a8564c261393de206c94a4557c850e7f2f`, `data/urdummlu/urdummlu.json` (MD5 `a25bbf61360af54747d2b1b8dbd96041`): native Urdu questions, never translated from English. **The sample is equal per subject (100 rows from each of the 26 subdomains), so every overall accuracy here is the mean over subjects, not the dataset-weighted accuracy of UrduMMLU.**

Scoring (freeze N5, 11.10): the CSQA fair `extract_answer` with the letter class `[A-D]`, on the answer text (after `</think>` for Qwen3-14B; the whole output otherwise). Empty prediction = unparsed = wrong; a reading_failed, refused or too_long row is unparsed = wrong. Truncated = reached max_tokens.

Arms: S = Qwen3-14B (student), T = Llama-3.1-70B-AWQ (teacher), M = qwen3.8-max (commercial). `_ur` = the Urdu prompt (CSQA fair prompt for four options); `_ue` = the English instruction with the Urdu item; T_r1 / M_r1 = the English prompt with the R1 translation (qwen3.8-max); S_ro = the Urdu prompt plus the R1 translation; S_self = the Urdu prompt plus Qwen3-14B's own translation (SL); T_self = the English prompt with Llama's own translation (TL); S_tr = the Urdu prompt plus teacher facts (from the R1 item). M_r1 is self-translation for M (R1 is the same model).

## 0. Self-test (run before any D15 arm was opened; every check fatal)

- S1 PASS: with the unrestricted class [A-E], this script's row evaluation and the scorer reproduce 190 / 300 on the sealed CSQA fair output; the recomputed prediction equals the stored one on every row.
- S2 PASS: 10 hand-made scorer cases for the class [A-D] (d15_score.self_check).
- S3 PASS: Holm step-down, 8 hand-made cases (m = 3, 2 and 1; ties; a missing test; the cap at 1; input order kept).
- S4 PASS: exact two-sided McNemar, 5 hand-made cases.
- S5 PASS: hand-made rows: LA counts, delta and p; the deciding-line rule at and above 5.0 pp (LA, LP, LT, LPT); a comparison restricted to a row set.
- S6 PASS: hand-made rows: the answer after </think>, a truncated thinking row, E outside the class, a reading_failed and a refused row (unparsed, wrong); set A drops a row without a valid R1 reading and a row refused in any M arm.
- S7 PASS: wording rules, prediction marks, and the N9 reading (H1: at least 2 of 3 'helps' = holds, none = not shown, otherwise mixed; H2: both = holds, none = not shown, otherwise mixed).

## 1. Sealed files

| unit | file | MD5 | rows | row status | seal |
|---|---|---|---|---|---|
| translation R1 | `outputs/efbpt/d15/tr/d15_tr_R1.jsonl` | `016e9e0c32467e97f8c310312795e54e` | 2600 | ok 2600 | outputs/efbpt/d15/tr/SEAL_tr_R1.json (md5 4b3131a03c1eb0ab783151c622838293; login node) |
| translation TL | `outputs/efbpt/d15/tr/d15_tr_TL.jsonl` | `9872a64c63c302201cd9b9d9951c6f33` | 2600 | ok 1947, reading_failed 653 | outputs/efbpt/d15/tr/SEAL_tr_TL.json (md5 2ec53a51614c0560ff66f46c575d28f2; slurm 121454) |
| translation SL | `outputs/efbpt/d15/tr/d15_tr_SL.jsonl` | `8accc7cf0b54bff9ab7d7fdb778b7d4b` | 2600 | ok 2598, reading_failed 2 | outputs/efbpt/d15/tr/SEAL_tr_SL.json (md5 cb05dc2793ff28af1ae670971901678e; slurm 121455) |
| teacher facts | `outputs/efbpt/d15/facts/d15_facts.jsonl` | `9dc6dd0eae7ce7437784181545e20389` | 2600 | ok 2600 | slurm 121463 |
| S_ur | `outputs/efbpt/d15/student/d15_S_ur.jsonl` | `7353b38050111d06ff6736afb9f5d1d1` | 2600 | ok 2600 | outputs/efbpt/d15/student/SEAL_S_ur.json (md5 ec168707b0f16aaaf82074f0dee4e333; slurm 121464; 2026-11-22 18:28:33) |
| S_ro | `outputs/efbpt/d15/student/d15_S_ro.jsonl` | `6b156eea4539c0577a872c951eb9192f` | 2600 | ok 2600 | outputs/efbpt/d15/student/SEAL_S_ro.json (md5 da532d283567e45d0cb6bb3c6f4aa2eb; slurm 121464; 2026-11-22 19:04:11) |
| S_self | `outputs/efbpt/d15/student/d15_S_self.jsonl` | `bdeb0e1bd3d187f6c46083f47c5f665e` | 2600 | ok 2598, reading_failed 2 | outputs/efbpt/d15/student/SEAL_S_self.json (md5 9a6803c860ef91ce9d833dca64b61e12; slurm 121464; 2026-11-22 19:40:35) |
| S_tr | `outputs/efbpt/d15/student/d15_S_tr.jsonl` | `270852b3aa882c63714436d0bfd95ac7` | 2600 | ok 2600 | outputs/efbpt/d15/student/SEAL_S_tr.json (md5 f4b372da97b37f74729725019dcba89d; slurm 121464; 2026-11-22 20:18:49) |
| T_ur | `outputs/efbpt/d15/teacher/d15_T_ur.jsonl` | `061ed58ecc923b381c57f54df09cc3cd` | 2600 | ok 2600 | outputs/efbpt/d15/teacher/SEAL_T_ur.json (md5 1c3981460fb4f2266515a52d9f7904ce; slurm 121465; 2026-11-22 19:22:45) |
| T_ue | `outputs/efbpt/d15/teacher/d15_T_ue.jsonl` | `d1d5e8b8e66ee023ff141dbdf9477ee2` | 2600 | ok 2600 | outputs/efbpt/d15/teacher/SEAL_T_ue.json (md5 2a95602257655ad9bd3e06ded5dc4464; slurm 121465; 2026-11-22 19:53:13) |
| T_r1 | `outputs/efbpt/d15/teacher/d15_T_r1.jsonl` | `4e8681c3e8872d04969df217b3438137` | 2600 | ok 2600 | outputs/efbpt/d15/teacher/SEAL_T_r1.json (md5 43b979606cf9a5aa6576242d50363f81; slurm 121465; 2026-11-22 20:40:05) |
| T_self | `outputs/efbpt/d15/teacher/d15_T_self.jsonl` | `4bdd8563b6d47d305f5ebe02b0a9506d` | 2600 | ok 1947, reading_failed 653 | outputs/efbpt/d15/teacher/SEAL_T_self.json (md5 da4173b96f02882e0fa102ebf6ee7387; slurm 121466; 2026-11-22 20:55:31) |
| M_ur | `outputs/efbpt/d15/api/d15_M_ur.jsonl` | `50b7a7907aba2db53568a5952333243a` | 2600 | ok 2600 | outputs/efbpt/d15/api/SEAL_M_ur.json (md5 bdab304f4f98ae060bca4759737a3532; login node; 2026-11-22 20:51:24) |
| M_ue | `outputs/efbpt/d15/api/d15_M_ue.jsonl` | `bf7df1c657c18f25ec7c44e08bd1c41c` | 2600 | ok 2600 | outputs/efbpt/d15/api/SEAL_M_ue.json (md5 cd0d53e0128a5131cfd1d18ac99b236c; login node; 2026-11-22 20:37:27) |
| M_r1 | `outputs/efbpt/d15/api/d15_M_r1.jsonl` | `c779a2adf7ffe170da33d364a53c086a` | 2600 | ok 2600 | outputs/efbpt/d15/api/SEAL_M_r1.json (md5 ecf8620f16da4ad84f40ae26c80e6dbc; login node; 2026-11-22 20:32:29) |

Missing arms: none; all 11 arm files, the 3 translation files and the facts file are sealed.

## 2. Sample, set A, dropped and failed rows

Set A (freeze N6) = sampled rows with a valid R1 translation and an answered (not refused) call in M_ur, M_ue, M_r1. n(sampled) = 2600, n(set A) = 2600, dropped = 0.

| rows | sampled | set A | dropped | R1 reading_failed | M_ur refused | M_ue refused | M_r1 refused | TL not valid | SL not valid | rows without facts |
|---|---|---|---|---|---|---|---|---|---|---|
| all | 2600 | 2600 | 0 | 0 | 0 | 0 | 0 | 653 | 2 | 7 |
| group URDU | 300 | 300 | 0 | 0 | 0 | 0 | 0 | 44 | 1 | 2 |
| group REGIONAL | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 37 | 0 | 2 |
| group OTHER | 2100 | 2100 | 0 | 0 | 0 | 0 | 0 | 572 | 1 | 3 |
| domain Humanities | 600 | 600 | 0 | 0 | 0 | 0 | 0 | 100 | 2 | 3 |
| domain Other | 100 | 100 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 |
| domain Profession | 200 | 200 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 |
| domain STEM | 600 | 600 | 0 | 0 | 0 | 0 | 0 | 239 | 0 | 1 |
| domain Social Sciences | 1100 | 1100 | 0 | 0 | 0 | 0 | 0 | 286 | 0 | 3 |

A row is counted once under 'dropped' even if it has more than one cause; 'not valid' = reading_failed or too_long.

## 3. Per arm

### set A (n = 2600)

| arm | model | acc % (correct) | unparsed (%) | truncated (%) | reading_failed | refused | too_long |
|---|---|---|---|---|---|---|---|
| S_ur | Qwen3-14B | 72.85 (1894) | 5 (0.19) | 101 (3.88) | 0 | 0 | 0 |
| S_ro | Qwen3-14B | 77.38 (2012) | 2 (0.08) | 82 (3.15) | 0 | 0 | 0 |
| S_self | Qwen3-14B | 71.96 (1871) | 2 (0.08) | 111 (4.27) | 2 | 0 | 0 |
| S_tr | Qwen3-14B | 75.77 (1970) | 0 (0.00) | 158 (6.08) | 0 | 0 | 0 |
| T_ur | Llama-3.1-70B-AWQ | 65.15 (1694) | 377 (14.50) | 34 (1.31) | 0 | 0 | 0 |
| T_ue | Llama-3.1-70B-AWQ | 67.46 (1754) | 153 (5.88) | 15 (0.58) | 0 | 0 | 0 |
| T_r1 | Llama-3.1-70B-AWQ | 80.69 (2098) | 23 (0.88) | 19 (0.73) | 0 | 0 | 0 |
| T_self | Llama-3.1-70B-AWQ | 56.46 (1468) | 682 (26.23) | 12 (0.46) | 653 | 0 | 0 |
| M_ur | qwen3.8-max | 88.77 (2308) | 0 (0.00) | 0 (0.00) | 0 | 0 | 0 |
| M_ue | qwen3.8-max | 89.15 (2318) | 0 (0.00) | 1 (0.04) | 0 | 0 | 0 |
| M_r1 | qwen3.8-max | 87.46 (2274) | 0 (0.00) | 0 (0.00) | 0 | 0 | 0 |

### all sampled rows (n = 2600)

| arm | model | acc % (correct) | unparsed (%) | truncated (%) | reading_failed | refused | too_long |
|---|---|---|---|---|---|---|---|
| S_ur | Qwen3-14B | 72.85 (1894) | 5 (0.19) | 101 (3.88) | 0 | 0 | 0 |
| S_ro | Qwen3-14B | 77.38 (2012) | 2 (0.08) | 82 (3.15) | 0 | 0 | 0 |
| S_self | Qwen3-14B | 71.96 (1871) | 2 (0.08) | 111 (4.27) | 2 | 0 | 0 |
| S_tr | Qwen3-14B | 75.77 (1970) | 0 (0.00) | 158 (6.08) | 0 | 0 | 0 |
| T_ur | Llama-3.1-70B-AWQ | 65.15 (1694) | 377 (14.50) | 34 (1.31) | 0 | 0 | 0 |
| T_ue | Llama-3.1-70B-AWQ | 67.46 (1754) | 153 (5.88) | 15 (0.58) | 0 | 0 | 0 |
| T_r1 | Llama-3.1-70B-AWQ | 80.69 (2098) | 23 (0.88) | 19 (0.73) | 0 | 0 | 0 |
| T_self | Llama-3.1-70B-AWQ | 56.46 (1468) | 682 (26.23) | 12 (0.46) | 653 | 0 | 0 |
| M_ur | qwen3.8-max | 88.77 (2308) | 0 (0.00) | 0 (0.00) | 0 | 0 | 0 |
| M_ue | qwen3.8-max | 89.15 (2318) | 0 (0.00) | 1 (0.04) | 0 | 0 | 0 |
| M_r1 | qwen3.8-max | 87.46 (2274) | 0 (0.00) | 0 (0.00) | 0 | 0 | 0 |

## 4. Family H1 — reading effect on native Urdu (3 tests, Holm at 0.05)

First arm = the model with the R1 English translation; second arm = the same model on the Urdu item (S: Urdu prompt; T and M: English instruction with the Urdu item). Wording (D14 F6): "helps" only if delta > 0 and Holm p < 0.05; "hurts" only if delta < 0 and Holm p < 0.05; otherwise "no clear effect". Delta = first arm minus second arm on the deciding line of set A.

| test | deciding line (set A) | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p | Holm p | wording |
|---|---|---|---|---|---|---|---|---|---|---|
| S_ro vs S_ur | LA | 2600 | 77.38 (2012) | 72.85 (1894) | +4.54 | 256 | 138 | 2.89e-09 | 5.78e-09 | helps |
| T_r1 vs T_ue | LA | 2600 | 80.69 (2098) | 67.46 (1754) | +13.23 | 504 | 160 | 2.38e-42 | 7.13e-42 | helps |
| M_r1 vs M_ue | LA | 2600 | 87.46 (2274) | 89.15 (2318) | -1.69 | 83 | 127 | 0.0029 | 0.0029 | hurts |

Printed beside each test (not deciding):

| test | row set | deciding line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| S_ro vs S_ur | all sampled rows | LA | 2600 | 77.38 (2012) | 72.85 (1894) | +4.54 | 256 | 138 | 2.89e-09 |
| T_r1 vs T_ue | all sampled rows | LA | 2600 | 80.69 (2098) | 67.46 (1754) | +13.23 | 504 | 160 | 2.38e-42 |
| M_r1 vs M_ue | all sampled rows | LA | 2600 | 87.46 (2274) | 89.15 (2318) | -1.69 | 83 | 127 | 0.0029 |

Every line (H1):

| row set | test | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| set A | S_ro vs S_ur | LA (deciding) | 2600 | 77.38 (2012) | 72.85 (1894) | +4.54 | 256 | 138 | 2.89e-09 |
| set A | S_ro vs S_ur | LP | 2593 | 77.48 (2009) | 73.00 (1893) | +4.47 | 253 | 137 | 4.48e-09 |
| set A | S_ro vs S_ur | LT | 2454 | 79.46 (1950) | 75.26 (1847) | +4.20 | 215 | 112 | 1.30e-08 |
| set A | S_ro vs S_ur | LPT | 2453 | 79.49 (1950) | 75.25 (1846) | +4.24 | 215 | 111 | 8.80e-09 |
| set A | T_r1 vs T_ue | LA (deciding) | 2600 | 80.69 (2098) | 67.46 (1754) | +13.23 | 504 | 160 | 2.38e-42 |
| set A | T_r1 vs T_ue | LP | 2425 | 81.36 (1973) | 72.00 (1746) | +9.36 | 379 | 152 | 2.18e-23 |
| set A | T_r1 vs T_ue | LT | 2567 | 81.22 (2085) | 68.02 (1746) | +13.21 | 492 | 153 | 2.12e-42 |
| set A | T_r1 vs T_ue | LPT | 2411 | 81.75 (1971) | 72.17 (1740) | +9.58 | 378 | 147 | 1.80e-24 |
| set A | M_r1 vs M_ue | LA (deciding) | 2600 | 87.46 (2274) | 89.15 (2318) | -1.69 | 83 | 127 | 0.0029 |
| set A | M_r1 vs M_ue | LP | 2600 | 87.46 (2274) | 89.15 (2318) | -1.69 | 83 | 127 | 0.0029 |
| set A | M_r1 vs M_ue | LT | 2599 | 87.46 (2273) | 89.19 (2318) | -1.73 | 82 | 127 | 0.0023 |
| set A | M_r1 vs M_ue | LPT | 2599 | 87.46 (2273) | 89.19 (2318) | -1.73 | 82 | 127 | 0.0023 |
| all sampled rows | S_ro vs S_ur | LA (deciding) | 2600 | 77.38 (2012) | 72.85 (1894) | +4.54 | 256 | 138 | 2.89e-09 |
| all sampled rows | S_ro vs S_ur | LP | 2593 | 77.48 (2009) | 73.00 (1893) | +4.47 | 253 | 137 | 4.48e-09 |
| all sampled rows | S_ro vs S_ur | LT | 2454 | 79.46 (1950) | 75.26 (1847) | +4.20 | 215 | 112 | 1.30e-08 |
| all sampled rows | S_ro vs S_ur | LPT | 2453 | 79.49 (1950) | 75.25 (1846) | +4.24 | 215 | 111 | 8.80e-09 |
| all sampled rows | T_r1 vs T_ue | LA (deciding) | 2600 | 80.69 (2098) | 67.46 (1754) | +13.23 | 504 | 160 | 2.38e-42 |
| all sampled rows | T_r1 vs T_ue | LP | 2425 | 81.36 (1973) | 72.00 (1746) | +9.36 | 379 | 152 | 2.18e-23 |
| all sampled rows | T_r1 vs T_ue | LT | 2567 | 81.22 (2085) | 68.02 (1746) | +13.21 | 492 | 153 | 2.12e-42 |
| all sampled rows | T_r1 vs T_ue | LPT | 2411 | 81.75 (1971) | 72.17 (1740) | +9.58 | 378 | 147 | 1.80e-24 |
| all sampled rows | M_r1 vs M_ue | LA (deciding) | 2600 | 87.46 (2274) | 89.15 (2318) | -1.69 | 83 | 127 | 0.0029 |
| all sampled rows | M_r1 vs M_ue | LP | 2600 | 87.46 (2274) | 89.15 (2318) | -1.69 | 83 | 127 | 0.0029 |
| all sampled rows | M_r1 vs M_ue | LT | 2599 | 87.46 (2273) | 89.19 (2318) | -1.73 | 82 | 127 | 0.0023 |
| all sampled rows | M_r1 vs M_ue | LPT | 2599 | 87.46 (2273) | 89.19 (2318) | -1.73 | 82 | 127 | 0.0023 |

- S_ro vs S_ur on set A: unparsed 2 / 5 (0.08% / 0.19%), truncated 82 / 101 (3.15% / 3.88%); deciding line LA (all rows).
- T_r1 vs T_ue on set A: unparsed 23 / 153 (0.88% / 5.88%), truncated 19 / 15 (0.73% / 0.58%); deciding line LA (all rows).
- M_r1 vs M_ue on set A: unparsed 0 / 0 (0.00% / 0.00%), truncated 0 / 1 (0.00% / 0.04%); deciding line LA (all rows).

H1 wording:

- S_ro vs S_ur: helps (delta +4.54 pp, Holm p 5.78e-09).
- T_r1 vs T_ue: helps (delta +13.23 pp, Holm p 7.13e-42).
- M_r1 vs M_ue: hurts (delta -1.69 pp, Holm p 0.0029).

## 5. Family H2 — self-translation, no outside model (2 tests, Holm at 0.05)

First arm = the model with its own English translation (S_self: SL in the Urdu prompt; T_self: TL in the English prompt); second arm = S_ur / T_ue. A row without a valid own translation counts as wrong in the first arm (N6); the line on the rows with a valid own translation is printed beside it. Wording (D14 F6): "helps" only if delta > 0 and Holm p < 0.05; "hurts" only if delta < 0 and Holm p < 0.05; otherwise "no clear effect". Delta = first arm minus second arm on the deciding line of set A.

| test | deciding line (set A) | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p | Holm p | wording |
|---|---|---|---|---|---|---|---|---|---|---|
| S_self vs S_ur | LA | 2600 | 71.96 (1871) | 72.85 (1894) | -0.88 | 173 | 196 | 0.2521 | 0.2521 | no clear effect |
| T_self vs T_ue | LP | 1810 | 76.69 (1388) | 71.22 (1289) | +5.47 | 246 | 147 | 6.74e-07 | 1.35e-06 | helps |

Printed beside each test (not deciding):

| test | row set | deciding line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| S_self vs S_ur | all sampled rows | LA | 2600 | 71.96 (1871) | 72.85 (1894) | -0.88 | 173 | 196 | 0.2521 |
| S_self vs S_ur | set A rows with a valid SL translation | LA | 2598 | 72.02 (1871) | 72.83 (1892) | -0.81 | 173 | 194 | 0.2965 |
| T_self vs T_ue | all sampled rows | LP | 1810 | 76.69 (1388) | 71.22 (1289) | +5.47 | 246 | 147 | 6.74e-07 |
| T_self vs T_ue | set A rows with a valid TL translation | LA | 1947 | 75.40 (1468) | 66.87 (1302) | +8.53 | 326 | 160 | 4.08e-14 |

Every line (H2):

| row set | test | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| set A | S_self vs S_ur | LA (deciding) | 2600 | 71.96 (1871) | 72.85 (1894) | -0.88 | 173 | 196 | 0.2521 |
| set A | S_self vs S_ur | LP | 2593 | 72.04 (1868) | 72.97 (1892) | -0.93 | 170 | 194 | 0.2280 |
| set A | S_self vs S_ur | LT | 2419 | 74.29 (1797) | 75.49 (1826) | -1.20 | 128 | 157 | 0.0970 |
| set A | S_self vs S_ur | LPT | 2417 | 74.35 (1797) | 75.47 (1824) | -1.12 | 128 | 155 | 0.1221 |
| set A | T_self vs T_ue | LA | 2600 | 56.46 (1468) | 67.46 (1754) | -11.00 | 326 | 612 | 6.67e-21 |
| set A | T_self vs T_ue | LP (deciding) | 1810 | 76.69 (1388) | 71.22 (1289) | +5.47 | 246 | 147 | 6.74e-07 |
| set A | T_self vs T_ue | LT | 2573 | 56.78 (1461) | 67.90 (1747) | -11.12 | 320 | 606 | 3.68e-21 |
| set A | T_self vs T_ue | LPT | 1805 | 76.79 (1386) | 71.30 (1287) | +5.48 | 245 | 146 | 6.31e-07 |
| all sampled rows | S_self vs S_ur | LA (deciding) | 2600 | 71.96 (1871) | 72.85 (1894) | -0.88 | 173 | 196 | 0.2521 |
| all sampled rows | S_self vs S_ur | LP | 2593 | 72.04 (1868) | 72.97 (1892) | -0.93 | 170 | 194 | 0.2280 |
| all sampled rows | S_self vs S_ur | LT | 2419 | 74.29 (1797) | 75.49 (1826) | -1.20 | 128 | 157 | 0.0970 |
| all sampled rows | S_self vs S_ur | LPT | 2417 | 74.35 (1797) | 75.47 (1824) | -1.12 | 128 | 155 | 0.1221 |
| set A, valid SL | S_self vs S_ur | LA (deciding) | 2598 | 72.02 (1871) | 72.83 (1892) | -0.81 | 173 | 194 | 0.2965 |
| set A, valid SL | S_self vs S_ur | LP | 2593 | 72.04 (1868) | 72.97 (1892) | -0.93 | 170 | 194 | 0.2280 |
| set A, valid SL | S_self vs S_ur | LT | 2417 | 74.35 (1797) | 75.47 (1824) | -1.12 | 128 | 155 | 0.1221 |
| set A, valid SL | S_self vs S_ur | LPT | 2417 | 74.35 (1797) | 75.47 (1824) | -1.12 | 128 | 155 | 0.1221 |
| all sampled rows | T_self vs T_ue | LA | 2600 | 56.46 (1468) | 67.46 (1754) | -11.00 | 326 | 612 | 6.67e-21 |
| all sampled rows | T_self vs T_ue | LP (deciding) | 1810 | 76.69 (1388) | 71.22 (1289) | +5.47 | 246 | 147 | 6.74e-07 |
| all sampled rows | T_self vs T_ue | LT | 2573 | 56.78 (1461) | 67.90 (1747) | -11.12 | 320 | 606 | 3.68e-21 |
| all sampled rows | T_self vs T_ue | LPT | 1805 | 76.79 (1386) | 71.30 (1287) | +5.48 | 245 | 146 | 6.31e-07 |
| set A, valid TL | T_self vs T_ue | LA (deciding) | 1947 | 75.40 (1468) | 66.87 (1302) | +8.53 | 326 | 160 | 4.08e-14 |
| set A, valid TL | T_self vs T_ue | LP | 1810 | 76.69 (1388) | 71.22 (1289) | +5.47 | 246 | 147 | 6.74e-07 |
| set A, valid TL | T_self vs T_ue | LT | 1928 | 75.78 (1461) | 67.17 (1295) | +8.61 | 320 | 154 | 1.90e-14 |
| set A, valid TL | T_self vs T_ue | LPT | 1805 | 76.79 (1386) | 71.30 (1287) | +5.48 | 245 | 146 | 6.31e-07 |

- S_self vs S_ur on set A: unparsed 2 / 5 (0.08% / 0.19%), truncated 111 / 101 (4.27% / 3.88%); deciding line LA (all rows).
- T_self vs T_ue on set A: unparsed 682 / 153 (26.23% / 5.88%), truncated 12 / 15 (0.46% / 0.58%); deciding line LP (rows parsed in both arms).

H2 wording:

- S_self vs S_ur: no clear effect (delta -0.88 pp, Holm p 0.2521).
- T_self vs T_ue: helps (delta +5.47 pp, Holm p 1.35e-06).

## 6. Family H3 — method (1 test)

First arm = S_tr (Qwen3-14B with teacher facts from the R1 item); second arm = S_ur. Wording (D14 F6): "helps" only if delta > 0 and Holm p < 0.05; "hurts" only if delta < 0 and Holm p < 0.05; otherwise "no clear effect". Delta = first arm minus second arm on the deciding line of set A.

| test | deciding line (set A) | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p | Holm p | wording |
|---|---|---|---|---|---|---|---|---|---|---|
| S_tr vs S_ur | LA | 2600 | 75.77 (1970) | 72.85 (1894) | +2.92 | 273 | 197 | 0.0005 | 0.0005 | helps |

Printed beside each test (not deciding):

| test | row set | deciding line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| S_tr vs S_ur | all sampled rows | LA | 2600 | 75.77 (1970) | 72.85 (1894) | +2.92 | 273 | 197 | 0.0005 |

Every line (H3):

| row set | test | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| set A | S_tr vs S_ur | LA (deciding) | 2600 | 75.77 (1970) | 72.85 (1894) | +2.92 | 273 | 197 | 0.0005 |
| set A | S_tr vs S_ur | LP | 2595 | 75.76 (1966) | 72.99 (1894) | +2.77 | 269 | 197 | 0.0010 |
| set A | S_tr vs S_ur | LT | 2375 | 78.78 (1871) | 75.96 (1804) | +2.82 | 215 | 148 | 0.0005 |
| set A | S_tr vs S_ur | LPT | 2375 | 78.78 (1871) | 75.96 (1804) | +2.82 | 215 | 148 | 0.0005 |
| all sampled rows | S_tr vs S_ur | LA (deciding) | 2600 | 75.77 (1970) | 72.85 (1894) | +2.92 | 273 | 197 | 0.0005 |
| all sampled rows | S_tr vs S_ur | LP | 2595 | 75.76 (1966) | 72.99 (1894) | +2.77 | 269 | 197 | 0.0010 |
| all sampled rows | S_tr vs S_ur | LT | 2375 | 78.78 (1871) | 75.96 (1804) | +2.82 | 215 | 148 | 0.0005 |
| all sampled rows | S_tr vs S_ur | LPT | 2375 | 78.78 (1871) | 75.96 (1804) | +2.82 | 215 | 148 | 0.0005 |

- S_tr vs S_ur on set A: unparsed 0 / 5 (0.00% / 0.19%), truncated 158 / 101 (6.08% / 3.88%); deciding line LA (all rows).

H3 wording:

- S_tr vs S_ur: helps (delta +2.92 pp, Holm p 0.0005).

## 7. Reading of the outcome (freeze N9, fixed before any run)

- H1: 2 of 3 tests say "helps". **The reading effect holds on native Urdu text.** (S_ro vs S_ur: helps; T_r1 vs T_ue: helps; M_r1 vs M_ue: hurts)
- H2: **mixed** (1 of 2 say "helps"). One by one: S_self vs S_ur: no clear effect; T_self vs T_ue: helps.
- H3: S_tr vs S_ur: helps.
- Whatever the outcome, D11, D13 and D14 stand as they are.

## 8. Descriptive comparisons on set A (no wording, raw p only)

| row set | comparison | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| set A | T_ue vs T_ur | LA | 2600 | 67.46 (1754) | 65.15 (1694) | +2.31 | 334 | 274 | 0.0167 |
| set A | T_ue vs T_ur | LP (deciding) | 2134 | 72.12 (1539) | 76.29 (1628) | -4.17 | 119 | 208 | 9.82e-07 |
| set A | T_ue vs T_ur | LT | 2554 | 67.82 (1732) | 66.01 (1686) | +1.80 | 313 | 267 | 0.0616 |
| set A | T_ue vs T_ur | LPT | 2128 | 72.13 (1535) | 76.36 (1625) | -4.23 | 116 | 206 | 6.01e-07 |
| set A | M_ue vs M_ur | LA (deciding) | 2600 | 89.15 (2318) | 88.77 (2308) | +0.38 | 73 | 63 | 0.4404 |
| set A | M_ue vs M_ur | LP | 2600 | 89.15 (2318) | 88.77 (2308) | +0.38 | 73 | 63 | 0.4404 |
| set A | M_ue vs M_ur | LT | 2599 | 89.19 (2318) | 88.80 (2308) | +0.38 | 73 | 63 | 0.4404 |
| set A | M_ue vs M_ur | LPT | 2599 | 89.19 (2318) | 88.80 (2308) | +0.38 | 73 | 63 | 0.4404 |
| set A | T_r1 vs T_ur | LA | 2600 | 80.69 (2098) | 65.15 (1694) | +15.54 | 555 | 151 | 4.19e-55 |
| set A | T_r1 vs T_ur | LP (deciding) | 2200 | 81.50 (1793) | 76.45 (1682) | +5.05 | 250 | 139 | 1.97e-08 |
| set A | T_r1 vs T_ur | LT | 2547 | 81.27 (2070) | 66.08 (1683) | +15.19 | 530 | 143 | 3.92e-53 |
| set A | T_r1 vs T_ur | LPT | 2182 | 81.94 (1788) | 76.72 (1674) | +5.22 | 248 | 134 | 5.75e-09 |
| set A | M_r1 vs M_ur | LA (deciding) | 2600 | 87.46 (2274) | 88.77 (2308) | -1.31 | 91 | 125 | 0.0245 |
| set A | M_r1 vs M_ur | LP | 2600 | 87.46 (2274) | 88.77 (2308) | -1.31 | 91 | 125 | 0.0245 |
| set A | M_r1 vs M_ur | LT | 2600 | 87.46 (2274) | 88.77 (2308) | -1.31 | 91 | 125 | 0.0245 |
| set A | M_r1 vs M_ur | LPT | 2600 | 87.46 (2274) | 88.77 (2308) | -1.31 | 91 | 125 | 0.0245 |
| set A | S_tr vs S_ro | LA (deciding) | 2600 | 75.77 (1970) | 77.38 (2012) | -1.62 | 184 | 226 | 0.0428 |
| set A | S_tr vs S_ro | LP | 2598 | 75.75 (1968) | 77.44 (2012) | -1.69 | 182 | 226 | 0.0331 |
| set A | S_tr vs S_ro | LT | 2393 | 79.02 (1891) | 80.07 (1916) | -1.04 | 143 | 168 | 0.1734 |
| set A | S_tr vs S_ro | LPT | 2392 | 79.01 (1890) | 80.10 (1916) | -1.09 | 142 | 168 | 0.1555 |
| set A | S_self vs S_ro | LA (deciding) | 2600 | 71.96 (1871) | 77.38 (2012) | -5.42 | 113 | 254 | 1.36e-13 |
| set A | S_self vs S_ro | LP | 2596 | 72.03 (1870) | 77.47 (2011) | -5.43 | 112 | 253 | 1.16e-13 |
| set A | S_self vs S_ro | LT | 2441 | 74.27 (1813) | 79.43 (1939) | -5.16 | 82 | 208 | 8.65e-14 |
| set A | S_self vs S_ro | LPT | 2438 | 74.32 (1812) | 79.49 (1938) | -5.17 | 81 | 207 | 7.01e-14 |
| set A | T_self vs T_r1 | LA | 2600 | 56.46 (1468) | 80.69 (2098) | -24.23 | 82 | 712 | 3.65e-126 |
| set A | T_self vs T_r1 | LP (deciding) | 1908 | 76.68 (1463) | 80.66 (1539) | -3.98 | 77 | 153 | 6.07e-07 |
| set A | T_self vs T_r1 | LT | 2572 | 56.84 (1462) | 81.26 (2090) | -24.42 | 78 | 706 | 2.12e-127 |
| set A | T_self vs T_r1 | LPT | 1894 | 76.98 (1458) | 81.05 (1535) | -4.07 | 74 | 151 | 3.14e-07 |

## 9. Group and domain splits (set A; descriptive, raw p only)

Groups from the subdomain names only (freeze 11.2): URDU = urdu literature, urdu language, urdu grammar; REGIONAL = pakistan studies, islamic studies; OTHER = the other 21. Inside a split the deciding-line rule is applied to the rows of the split.

| split | test | deciding line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| group URDU | S_ro vs S_ur | LA | 300 | 45.67 (137) | 43.00 (129) | +2.67 | 40 | 32 | 0.4096 |
| group URDU | T_r1 vs T_ue | LA | 300 | 56.33 (169) | 53.00 (159) | +3.33 | 50 | 40 | 0.3428 |
| group URDU | M_r1 vs M_ue | LA | 300 | 67.67 (203) | 76.33 (229) | -8.67 | 12 | 38 | 0.0003 |
| group URDU | S_self vs S_ur | LA | 300 | 34.67 (104) | 43.00 (129) | -8.33 | 24 | 49 | 0.0046 |
| group URDU | T_self vs T_ue | LP | 233 | 48.93 (114) | 55.36 (129) | -6.44 | 25 | 40 | 0.0817 |
| group REGIONAL | S_ro vs S_ur | LA | 200 | 73.50 (147) | 77.00 (154) | -3.50 | 16 | 23 | 0.3368 |
| group REGIONAL | T_r1 vs T_ue | LP | 186 | 81.72 (152) | 77.42 (144) | +4.30 | 20 | 12 | 0.2153 |
| group REGIONAL | M_r1 vs M_ue | LA | 200 | 88.00 (176) | 93.00 (186) | -5.00 | 5 | 15 | 0.0414 |
| group REGIONAL | S_self vs S_ur | LA | 200 | 72.00 (144) | 77.00 (154) | -5.00 | 15 | 25 | 0.1539 |
| group REGIONAL | T_self vs T_ue | LP | 150 | 78.00 (117) | 76.67 (115) | +1.33 | 14 | 12 | 0.8450 |
| group OTHER | S_ro vs S_ur | LA | 2100 | 82.29 (1728) | 76.71 (1611) | +5.57 | 200 | 83 | 2.63e-12 |
| group OTHER | T_r1 vs T_ue | LP | 1959 | 84.79 (1661) | 73.66 (1443) | +11.13 | 318 | 100 | 1.49e-27 |
| group OTHER | M_r1 vs M_ue | LA | 2100 | 90.24 (1895) | 90.62 (1903) | -0.38 | 66 | 74 | 0.5543 |
| group OTHER | S_self vs S_ur | LA | 2100 | 77.29 (1623) | 76.71 (1611) | +0.57 | 134 | 122 | 0.4918 |
| group OTHER | T_self vs T_ue | LP | 1427 | 81.08 (1157) | 73.23 (1045) | +7.85 | 207 | 95 | 1.04e-10 |
| domain Humanities | S_ro vs S_ur | LA | 600 | 60.50 (363) | 56.00 (336) | +4.50 | 79 | 52 | 0.0227 |
| domain Humanities | T_r1 vs T_ue | LA | 600 | 68.83 (413) | 61.17 (367) | +7.67 | 104 | 58 | 0.0004 |
| domain Humanities | M_r1 vs M_ue | LA | 600 | 76.50 (459) | 82.00 (492) | -5.50 | 29 | 62 | 0.0007 |
| domain Humanities | S_self vs S_ur | LA | 600 | 50.33 (302) | 56.00 (336) | -5.67 | 47 | 81 | 0.0034 |
| domain Humanities | T_self vs T_ue | LP | 463 | 63.93 (296) | 64.79 (300) | -0.86 | 57 | 61 | 0.7826 |
| domain Other | S_ro vs S_ur | LA | 100 | 75.00 (75) | 69.00 (69) | +6.00 | 11 | 5 | 0.2101 |
| domain Other | T_r1 vs T_ue | LA | 100 | 78.00 (78) | 74.00 (74) | +4.00 | 12 | 8 | 0.5034 |
| domain Other | M_r1 vs M_ue | LA | 100 | 87.00 (87) | 88.00 (88) | -1.00 | 2 | 3 | 1.0000 |
| domain Other | S_self vs S_ur | LA | 100 | 73.00 (73) | 69.00 (69) | +4.00 | 10 | 6 | 0.4545 |
| domain Other | T_self vs T_ue | LP | 85 | 84.71 (72) | 77.65 (66) | +7.06 | 11 | 5 | 0.2101 |
| domain Profession | S_ro vs S_ur | LA | 200 | 85.00 (170) | 78.50 (157) | +6.50 | 21 | 8 | 0.0241 |
| domain Profession | T_r1 vs T_ue | LA | 200 | 86.00 (172) | 70.50 (141) | +15.50 | 40 | 9 | 9.26e-06 |
| domain Profession | M_r1 vs M_ue | LA | 200 | 90.00 (180) | 92.50 (185) | -2.50 | 2 | 7 | 0.1797 |
| domain Profession | S_self vs S_ur | LA | 200 | 78.00 (156) | 78.50 (157) | -0.50 | 14 | 15 | 1.0000 |
| domain Profession | T_self vs T_ue | LP | 169 | 83.43 (141) | 75.74 (128) | +7.69 | 25 | 12 | 0.0470 |
| domain STEM | S_ro vs S_ur | LA | 600 | 96.17 (577) | 90.83 (545) | +5.33 | 42 | 10 | 9.06e-06 |
| domain STEM | T_r1 vs T_ue | LP | 560 | 95.18 (533) | 81.07 (454) | +14.11 | 89 | 10 | 5.52e-17 |
| domain STEM | M_r1 vs M_ue | LA | 600 | 97.67 (586) | 98.33 (590) | -0.67 | 3 | 7 | 0.3438 |
| domain STEM | S_self vs S_ur | LA | 600 | 91.83 (551) | 90.83 (545) | +1.00 | 24 | 18 | 0.4408 |
| domain STEM | T_self vs T_ue | LP | 338 | 93.49 (316) | 81.95 (277) | +11.54 | 46 | 7 | 4.00e-08 |
| domain Social Sciences | S_ro vs S_ur | LA | 1100 | 75.18 (827) | 71.55 (787) | +3.64 | 103 | 63 | 0.0024 |
| domain Social Sciences | T_r1 vs T_ue | LA | 1100 | 78.91 (868) | 65.18 (717) | +13.73 | 225 | 74 | 7.12e-19 |
| domain Social Sciences | M_r1 vs M_ue | LA | 1100 | 87.45 (962) | 87.55 (963) | -0.09 | 47 | 48 | 1.0000 |
| domain Social Sciences | S_self vs S_ur | LA | 1100 | 71.73 (789) | 71.55 (787) | +0.18 | 78 | 76 | 0.9358 |
| domain Social Sciences | T_self vs T_ue | LP | 755 | 74.57 (563) | 68.61 (518) | +5.96 | 107 | 62 | 0.0007 |

## 10. Per subdomain (set A; deltas in pp on all rows of the subdomain in set A, no p)

| subdomain | group | domain | n (set A) | S_ur % | S_ro − S_ur | T_r1 − T_ue | M_r1 − M_ue | S_self − S_ur | T_self − T_ue | S_tr − S_ur |
|---|---|---|---|---|---|---|---|---|---|---|
| biology | OTHER | STEM | 100 | 91.00 | +5.00 | +22.00 | -1.00 | -5.00 | -5.00 | +6.00 |
| chemistry | OTHER | STEM | 100 | 86.00 | +10.00 | +26.00 | -2.00 | +3.00 | -3.00 | +8.00 |
| civics | OTHER | Social Sciences | 100 | 75.00 | +3.00 | +18.00 | -3.00 | +3.00 | -20.00 | +8.00 |
| commerce | OTHER | Social Sciences | 100 | 79.00 | +3.00 | +12.00 | +0.00 | -4.00 | -11.00 | -3.00 |
| computer science | OTHER | STEM | 100 | 89.00 | +7.00 | +17.00 | +0.00 | +6.00 | -22.00 | +3.00 |
| current and international affairs | OTHER | Social Sciences | 100 | 46.00 | -4.00 | -3.00 | +2.00 | -5.00 | -6.00 | -10.00 |
| economics | OTHER | Social Sciences | 100 | 75.00 | +13.00 | +21.00 | +1.00 | +5.00 | -18.00 | +11.00 |
| education | OTHER | Social Sciences | 100 | 72.00 | +2.00 | +12.00 | +0.00 | +0.00 | -8.00 | +5.00 |
| ethics | OTHER | Humanities | 100 | 72.00 | +7.00 | +11.00 | -1.00 | +0.00 | -5.00 | +12.00 |
| fine arts | OTHER | Humanities | 100 | 62.00 | +11.00 | +16.00 | +2.00 | -5.00 | -6.00 | -1.00 |
| general knowledge | OTHER | Other | 100 | 69.00 | +6.00 | +4.00 | -1.00 | +4.00 | +1.00 | +4.00 |
| general science | OTHER | STEM | 100 | 91.00 | +6.00 | +14.00 | -2.00 | -2.00 | -12.00 | +2.00 |
| geography | OTHER | Social Sciences | 100 | 74.00 | +12.00 | +14.00 | +1.00 | +6.00 | -10.00 | +10.00 |
| health and physical education | OTHER | Social Sciences | 100 | 69.00 | +8.00 | +19.00 | -5.00 | +1.00 | -16.00 | +4.00 |
| islamic studies | REGIONAL | Humanities | 100 | 73.00 | +1.00 | +9.00 | -8.00 | -4.00 | -8.00 | -2.00 |
| mathematics | OTHER | STEM | 100 | 95.00 | +1.00 | +15.00 | +0.00 | +0.00 | -55.00 | +1.00 |
| pakistan studies | REGIONAL | Social Sciences | 100 | 81.00 | -8.00 | +9.00 | -2.00 | -6.00 | -12.00 | -1.00 |
| physics | OTHER | STEM | 100 | 93.00 | +3.00 | +18.00 | +1.00 | +4.00 | -22.00 | +1.00 |
| professional development | OTHER | Profession | 100 | 75.00 | +4.00 | +21.00 | +0.00 | -2.00 | +9.00 | +0.00 |
| professional studies | OTHER | Profession | 100 | 82.00 | +9.00 | +10.00 | -5.00 | +1.00 | -1.00 | +5.00 |
| psychology | OTHER | Social Sciences | 100 | 85.00 | +4.00 | +22.00 | +1.00 | -4.00 | -12.00 | +8.00 |
| psychometrics | OTHER | Social Sciences | 100 | 42.00 | +1.00 | +10.00 | +3.00 | +4.00 | +0.00 | -5.00 |
| sociology | OTHER | Social Sciences | 100 | 89.00 | +6.00 | +17.00 | +1.00 | +2.00 | -4.00 | -1.00 |
| urdu grammar | URDU | Humanities | 100 | 47.00 | +9.00 | +7.00 | -8.00 | -10.00 | -16.00 | -1.00 |
| urdu language | URDU | Humanities | 100 | 50.00 | -4.00 | +8.00 | -6.00 | -3.00 | -12.00 | +9.00 |
| urdu literature | URDU | Humanities | 100 | 32.00 | +3.00 | -5.00 | -12.00 | -12.00 | -12.00 | +3.00 |

## 11. Predictions (freeze N8, written before any run; read on LA of set A)

| prediction | first % | second % | delta pp | result |
|---|---|---|---|---|
| S_ro - S_ur in [-2, +6] | 77.38 | 72.85 | +4.54 | hit |
| T_r1 - T_ue in [-2, +8] | 80.69 | 67.46 | +13.23 | miss |
| M_r1 - M_ue in [-4, +3] | 87.46 | 89.15 | -1.69 | hit |
| S_self - S_ur in [-4, +4] | 71.96 | 72.85 | -0.88 | hit |
| T_self - T_ue in [-3, +6] | 56.46 | 67.46 | -11.00 | miss |
| S_tr - S_ur in [-2, +6] | 75.77 | 72.85 | +2.92 | hit |
| S_ro − S_ur: the delta in URDU is not above the delta in OTHER | URDU +2.67 | OTHER +5.57 | -2.90 | hit |
| T_r1 − T_ue: the delta in URDU is not above the delta in OTHER | URDU +3.33 | OTHER +15.05 | -11.71 | hit |
| M_r1 − M_ue: the delta in URDU is not above the delta in OTHER | URDU -8.67 | OTHER -0.38 | -8.29 | hit |

Predictions hit: 7 of 9.

## 12. Translations and teacher facts

| translation | model | rows | valid | reading_failed | too_long | requests | accepted readings that reached max_tokens | contains "correct" | contains "answer is" | mark on one option | any flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | qwen3.8-max (API) | 2600 | 2600 | 0 | 0 | 2600 | 0 | 30 | 0 | 2 | 32 |
| TL | Llama-3.1-70B-AWQ, D10 engine | 2600 | 1947 | 653 | 0 | 2600 | 0 | 29 | 0 | 3 | 32 |
| SL | Qwen3-14B, thinking off | 2600 | 2598 | 2 | 0 | 2600 | 0 | 35 | 0 | 2 | 37 |

Flags are reported, not a check (freeze 11.5). "correct" also matches questions such as "Which of the following is correct?".

| teacher facts | rows | replies parsed | facts kept | lines removed by the D10 verdict filter | lines removed by the D14 filter | rows with a D14-removed line | rows without facts |
|---|---|---|---|---|---|---|---|
| `9dc6dd0eae7ce7437784181545e20389` | 2600 | 2593 | 15347 | 4 | 34 | 33 | 7 |

A row without facts has no facts block: its S_tr prompt equals its S_ur prompt.

## 13. Seconds and tokens

Seconds of a vLLM row = the seconds of its chunk divided by the rows generated in the chunk (up to 64 prompts per call); seconds of an API row = the latency of its answered request (R1: of all its requests).

| unit | rows generated | seconds total | seconds per row | prompt tokens per row | new tokens per row |
|---|---|---|---|---|---|
| translation R1 | 2600 | 5444 | 2.09 | 155.8 | 38.1 |
| translation TL | 2600 | 8210 | 3.16 | 187.9 | 35.3 |
| translation SL | 2600 | 186 | 0.07 | 174.4 | 38.0 |
| S_ur | 2600 | 1961 | 0.75 | 212.4 | 378.9 |
| S_ro | 2600 | 1941 | 0.75 | 261.7 | 345.8 |
| S_self | 2598 | 1989 | 0.77 | 261.4 | 372.1 |
| S_tr | 2600 | 2075 | 0.80 | 332.4 | 358.5 |
| T_ur | 2600 | 2561 | 0.99 | 229.9 | 60.0 |
| T_ue | 2600 | 1721 | 0.66 | 157.9 | 28.1 |
| T_r1 | 2600 | 2690 | 1.03 | 129.3 | 129.7 |
| T_self | 1947 | 2055 | 1.06 | 128.1 | 126.2 |
| M_ur | 2600 | 10516 | 4.04 | 155.8 | 125.8 |
| M_ue | 2600 | 8847 | 3.40 | 124.8 | 101.9 |
| M_r1 | 2600 | 8261 | 3.18 | 112.1 | 91.5 |

API ledger at the time of the readout (TEST-5 and retries included): 10421 calls of at most 11500; 2360888 tokens of at most 5000000 (prompt 1429371, completion 931517); stopped: no.

Amount used by the key (gateway billing query; the key has no limit, so no balance is given): before D15 (no D15 request sent yet): 13.5241; after D15: 21.0341.

