# D14 readout — reading effect and teacher facts on CSQA, PIQA and GSM8K

Written by `eval/error_analysis_tests/efbpt/d14_readout.py` (MD5 `1734cee1b977395ef6cb175875321380`, git HEAD `8d2e1429defb73ae7d71f1774d2e2ea7fc222ab4`), one run, 2026-11-16 04:18:39. Freeze: `docs/EFBPT_D14_MORE_TASKS_FREEZE.md` (MD5 `31bc15881ad8102a2b35a676abf1c268`). Every arm file was sealed before this script opened it; no D14 arm was scored before this run. No arm is run again after this readout.

Scoring (freeze F5, 9.11): the task's `extract_answer` and gold rule from its fair script, unchanged, on the answer text (after `</think>` for Qwen3-14B; the whole output otherwise); for T_r1, T_og, M_r1, M_og on PIQA and GSM8K the same function with "Final answer" in place of the Urdu marker. Empty prediction = unparsed = wrong. A reading_failed, refused or too_long row has no answer: unparsed = wrong. Truncated = reached max_tokens.

Arms: S = Qwen3-14B (student), T = Llama-3.1-70B-AWQ (teacher), M = qwen3.8-max (commercial). `_ur` = Urdu prompt; S_ro = Urdu prompt plus the R1 English translation; S_og = Urdu prompt plus the original English item; S_tr = Urdu prompt plus teacher facts; T_r1, M_r1 = English prompt with the R1 translation; T_og, M_og = English prompt with the original English item.

## 0. Self-test (run before any D14 arm was opened; every check fatal)

- S1 PASS: the task scorers and this script's row evaluation reproduce the sealed fair outputs (CSQA 190/300, PIQA 108/150, GSM8K 621/700); the recomputed prediction equals the stored one on every row.
- S2 PASS: 9 hand-made scorer cases (the EN marker for PIQA and GSM8K, the unchanged functions, CSQA).
- S3 PASS: Holm step-down, 7 hand-made cases (running maximum, the cap at 1, ties, a family larger than the tests present, a missing test, input order kept).
- S4 PASS: exact two-sided McNemar, 5 hand-made cases.
- S5 PASS: hand-made rows: LA / LP / LT / LPT counts and p, the deciding-line rule at and above 5.0 pp for unparsed and for truncated, a reading_failed row and a refused row (unparsed, wrong), the answer after </think>, the EN marker on an API arm, the wording rules, the prediction marks.

## 1. Sealed files

| task | arm | file | MD5 | rows | row status | seal |
|---|---|---|---|---|---|---|
| CSQA | S_ur | `outputs/efbpt/d14/student/csqa/d14_S_ur.jsonl` | `4640d9e7bf8b95683e0599882953bc0c` | 1500 | ok 1500 | outputs/efbpt/d14/student/csqa/SEAL_S_ur.json (md5 86e9539fc0862ce32a8d7032d5d7378c; slurm 118377; 2026-11-16 00:32:53) |
| CSQA | S_ro | `outputs/efbpt/d14/student/csqa/d14_S_ro.jsonl` | `25718fcfd376db17a10f8e61bebc4dea` | 1500 | ok 1499, reading_failed 1 | outputs/efbpt/d14/student/csqa/SEAL_S_ro.json (md5 6a039cd9dbbdb8f4d52435e3fe0dafd3; slurm 118377; 2026-11-16 00:54:56) |
| CSQA | S_og | `outputs/efbpt/d14/student/csqa/d14_S_og.jsonl` | `10c786079484dffaf3df6ab2a92e0022` | 1500 | ok 1500 | outputs/efbpt/d14/student/csqa/SEAL_S_og.json (md5 f51805c6efdb70aedb7c00554336fb28; slurm 118377; 2026-11-16 01:16:57) |
| CSQA | S_tr | `outputs/efbpt/d14/student/csqa/d14_S_tr.jsonl` | `b2dc72c029a0281e8db0bc582d4ce6b0` | 1500 | ok 1499, reading_failed 1 | outputs/efbpt/d14/student/csqa/SEAL_S_tr.json (md5 142d4844d3e1088825552bb134b7373b; slurm 118377; 2026-11-16 01:41:00) |
| CSQA | T_ur | `outputs/efbpt/d14/teacher/csqa/d14_T_ur.jsonl` | `b4a6ed67517a6f39f2cb3cfd00aa8ec5` | 1500 | ok 1500 | outputs/efbpt/d14/teacher/csqa/SEAL_T_ur.json (md5 a1ba1e5a2fd624c5986689188bdce9c3; slurm 118379; 2026-11-16 02:06:54) |
| CSQA | T_r1 | `outputs/efbpt/d14/teacher/csqa/d14_T_r1.jsonl` | `7017c24b963c465bcf33744c00461a54` | 1500 | ok 1499, reading_failed 1 | outputs/efbpt/d14/teacher/csqa/SEAL_T_r1.json (md5 63ea01d1da168986873c7f0f39c60270; slurm 118379; 2026-11-16 02:40:34) |
| CSQA | T_og | `outputs/efbpt/d14/teacher/csqa/d14_T_og.jsonl` | `97ba368c9fcdaba19d4ae81189c80bec` | 1500 | ok 1500 | outputs/efbpt/d14/teacher/csqa/SEAL_T_og.json (md5 096d49f0c9600739e339c22151c5ed8e; slurm 118379; 2026-11-16 03:11:40) |
| CSQA | M_ur | `outputs/efbpt/d14/api/csqa/d14_M_ur.jsonl` | `e496d65c227924c9efda19aa160ff248` | 1500 | ok 1500 | outputs/efbpt/d14/api/csqa/SEAL_M_ur.json (md5 491c5f0a7f328b6ad6ad94a903f181de; login node, no slurm job; 2026-11-16 00:34:19) |
| CSQA | M_r1 | `outputs/efbpt/d14/api/csqa/d14_M_r1.jsonl` | `c05e44892fafbf450eba1032cb60349a` | 1500 | ok 1499, reading_failed 1 | outputs/efbpt/d14/api/csqa/SEAL_M_r1.json (md5 b2d3ba56aeb1ed20439791612279de75; login node, no slurm job; 2026-11-16 01:19:21) |
| CSQA | M_og | `outputs/efbpt/d14/api/csqa/d14_M_og.jsonl` | `33fd7c49faed891e0405065d8194c11f` | 1500 | ok 1500 | outputs/efbpt/d14/api/csqa/SEAL_M_og.json (md5 c29ab362d93dc51e1f9124f51065037d; login node, no slurm job; 2026-11-16 02:04:01) |
| PIQA | S_ur | `outputs/efbpt/d14/student/piqa/d14_S_ur.jsonl` | `6da874976b35979224a05b2b0fef25f4` | 750 | ok 750 | outputs/efbpt/d14/student/piqa/SEAL_S_ur.json (md5 467849e7ea87980ac97a38e81eecaa9a; slurm 118378; 2026-11-16 00:44:06) |
| PIQA | S_ro | `outputs/efbpt/d14/student/piqa/d14_S_ro.jsonl` | `f4186945da5ccc502c09928019bf57f1` | 750 | ok 750 | outputs/efbpt/d14/student/piqa/SEAL_S_ro.json (md5 a79c09b6a3b2d8963b53099d1181e790; slurm 118378; 2026-11-16 00:58:40) |
| PIQA | S_og | `outputs/efbpt/d14/student/piqa/d14_S_og.jsonl` | `5adca62927b1c4830c9e030640046ae4` | 750 | ok 750 | outputs/efbpt/d14/student/piqa/SEAL_S_og.json (md5 483aa4d7aca35eb2d53bb54075946b8f; slurm 118378; 2026-11-16 01:13:43) |
| PIQA | S_tr | `outputs/efbpt/d14/student/piqa/d14_S_tr.jsonl` | `85c020514a9c296eb67d8a53fe033b77` | 750 | ok 750 | outputs/efbpt/d14/student/piqa/SEAL_S_tr.json (md5 a825c359125ccf6fe761d71d9da60a2c; slurm 118378; 2026-11-16 01:32:59) |
| PIQA | T_ur | `outputs/efbpt/d14/teacher/piqa/d14_T_ur.jsonl` | `3f22a0f73b073854b986f5cc96080bab` | 750 | ok 750 | outputs/efbpt/d14/teacher/piqa/SEAL_T_ur.json (md5 536db6751c0fd5bee53c2fe08fc03c44; slurm 118380; 2026-11-16 02:35:49) |
| PIQA | T_r1 | `outputs/efbpt/d14/teacher/piqa/d14_T_r1.jsonl` | `4a4adef5a31b5009e78403c4beaa261c` | 750 | ok 750 | outputs/efbpt/d14/teacher/piqa/SEAL_T_r1.json (md5 9194f821b48b0a89264b628756a8aff4; slurm 118380; 2026-11-16 02:56:20) |
| PIQA | T_og | `outputs/efbpt/d14/teacher/piqa/d14_T_og.jsonl` | `99e88b1d4e208fde2a3fc4339ae701bd` | 750 | ok 750 | outputs/efbpt/d14/teacher/piqa/SEAL_T_og.json (md5 d7256496a89cc1dabef5e02e87c44c9e; slurm 118380; 2026-11-16 03:16:50) |
| PIQA | M_ur | `outputs/efbpt/d14/api/piqa/d14_M_ur.jsonl` | `94d43a37ad4eb49d6a346fcf24806bce` | 750 | ok 750 | outputs/efbpt/d14/api/piqa/SEAL_M_ur.json (md5 6ff969a03cf60ed8b42baccde8eeffe5; login node, no slurm job; 2026-11-16 00:35:40) |
| PIQA | M_r1 | `outputs/efbpt/d14/api/piqa/d14_M_r1.jsonl` | `753202736877d9ef657eff1fdf6c9d22` | 750 | ok 750 | outputs/efbpt/d14/api/piqa/SEAL_M_r1.json (md5 1398dc1d2178a3ee3441e75050c26bc3; login node, no slurm job; 2026-11-16 01:10:12) |
| PIQA | M_og | `outputs/efbpt/d14/api/piqa/d14_M_og.jsonl` | `997618ad80f628b5e2b2e33ab995c8a1` | 750 | ok 750 | outputs/efbpt/d14/api/piqa/SEAL_M_og.json (md5 b9ec567949bebe0d6a2c68b61021fcb1; login node, no slurm job; 2026-11-16 01:44:56) |
| GSM8K | S_ur | `outputs/efbpt/d14/student/gsm8k/d14_S_ur.jsonl` | `b64750f37a2a46505b62c160f43bbc17` | 700 | ok 700 | outputs/efbpt/d14/student/gsm8k/SEAL_S_ur.json (md5 46dbb77b937e3d8d59c3a6c9d0300380; slurm 118376; 2026-11-15 23:48:19) |
| GSM8K | S_ro | `outputs/efbpt/d14/student/gsm8k/d14_S_ro.jsonl` | `443e62d57306bed4e59dc9ceb10179af` | 700 | ok 700 | outputs/efbpt/d14/student/gsm8k/SEAL_S_ro.json (md5 2cde1cad4f95a0332a851d874ecb1981; slurm 118376; 2026-11-16 00:07:09) |
| GSM8K | S_og | `outputs/efbpt/d14/student/gsm8k/d14_S_og.jsonl` | `37556d0e2943e59cc30f8bb9434ea6db` | 700 | ok 700 | outputs/efbpt/d14/student/gsm8k/SEAL_S_og.json (md5 76900d042b9557ae0b82691b2562625a; slurm 118376; 2026-11-16 00:25:51) |
| GSM8K | T_ur | `outputs/efbpt/d14/teacher/gsm8k/d14_T_ur.jsonl` | `5d7632877dff9f2ef9438b482e701f46` | 700 | ok 700 | outputs/efbpt/d14/teacher/gsm8k/SEAL_T_ur.json (md5 6f6b26d70e75e0ff4a2b1b572a3042b7; slurm 118381; 2026-11-16 03:47:57) |
| GSM8K | T_r1 | `outputs/efbpt/d14/teacher/gsm8k/d14_T_r1.jsonl` | `829721a5c84ea015fc5be19431bfd696` | 700 | ok 700 | outputs/efbpt/d14/teacher/gsm8k/SEAL_T_r1.json (md5 7f2559dd2dcd665a40cd33385f971f99; slurm 118381; 2026-11-16 04:02:46) |
| GSM8K | T_og | `outputs/efbpt/d14/teacher/gsm8k/d14_T_og.jsonl` | `4f0ac0aa7da959a351fec4f302c42dcf` | 700 | ok 700 | outputs/efbpt/d14/teacher/gsm8k/SEAL_T_og.json (md5 1d2e88db924a9caf46f793f99698609a; slurm 118381; 2026-11-16 04:17:07) |
| GSM8K | M_ur | `outputs/efbpt/d14/api/gsm8k/d14_M_ur.jsonl` | `236b11af3fd6111f8432d97660b192f5` | 700 | ok 700 | outputs/efbpt/d14/api/gsm8k/SEAL_M_ur.json (md5 640b92cecc5bcbf70e089bc5fb1d048d; login node, no slurm job; 2026-11-15 23:57:36) |
| GSM8K | M_r1 | `outputs/efbpt/d14/api/gsm8k/d14_M_r1.jsonl` | `607a5f8d0ef13ab91b9d81fcedf5f8d7` | 700 | ok 700 | outputs/efbpt/d14/api/gsm8k/SEAL_M_r1.json (md5 5bb89bc57dc26669cbfa3b67600a1092; login node, no slurm job; 2026-11-16 00:22:27) |
| GSM8K | M_og | `outputs/efbpt/d14/api/gsm8k/d14_M_og.jsonl` | `211da49c427013dc9ef0da845f36d2b1` | 700 | ok 700 | outputs/efbpt/d14/api/gsm8k/SEAL_M_og.json (md5 13e397318e9264f0c1da04c3ea337b9b; login node, no slurm job; 2026-11-16 00:45:07) |

Missing arms: none; all 29 arm files are sealed.

## 2. Per arm

### CSQA (n = 1500)

| arm | acc % (correct) | unparsed (%) | truncated (%) | reading_failed | refused | too_long |
|---|---|---|---|---|---|---|
| S_ur | 68.53 (1028) | 0 (0.00) | 83 (5.53) | 0 | 0 | 0 |
| S_ro | 74.20 (1113) | 2 (0.13) | 54 (3.60) | 1 | 0 | 0 |
| S_og | 78.93 (1184) | 2 (0.13) | 62 (4.13) | 0 | 0 | 0 |
| S_tr | 70.00 (1050) | 2 (0.13) | 154 (10.27) | 1 | 0 | 0 |
| T_ur | 56.93 (854) | 228 (15.20) | 9 (0.60) | 0 | 0 | 0 |
| T_r1 | 74.20 (1113) | 1 (0.07) | 4 (0.27) | 1 | 0 | 0 |
| T_og | 81.67 (1225) | 0 (0.00) | 1 (0.07) | 0 | 0 | 0 |
| M_ur | 75.80 (1137) | 0 (0.00) | 0 (0.00) | 0 | 0 | 0 |
| M_r1 | 80.33 (1205) | 1 (0.07) | 1 (0.07) | 1 | 0 | 0 |
| M_og | 88.40 (1326) | 0 (0.00) | 0 (0.00) | 0 | 0 | 0 |

### PIQA (n = 750)

| arm | acc % (correct) | unparsed (%) | truncated (%) | reading_failed | refused | too_long |
|---|---|---|---|---|---|---|
| S_ur | 71.60 (537) | 0 (0.00) | 1 (0.13) | 0 | 0 | 0 |
| S_ro | 77.33 (580) | 0 (0.00) | 2 (0.27) | 0 | 0 | 0 |
| S_og | 78.40 (588) | 0 (0.00) | 1 (0.13) | 0 | 0 | 0 |
| S_tr | 72.40 (543) | 0 (0.00) | 12 (1.60) | 0 | 0 | 0 |
| T_ur | 78.93 (592) | 7 (0.93) | 12 (1.60) | 0 | 0 | 0 |
| T_r1 | 86.80 (651) | 1 (0.13) | 0 (0.00) | 0 | 0 | 0 |
| T_og | 88.13 (661) | 1 (0.13) | 0 (0.00) | 0 | 0 | 0 |
| M_ur | 92.40 (693) | 0 (0.00) | 0 (0.00) | 0 | 0 | 0 |
| M_r1 | 93.07 (698) | 0 (0.00) | 2 (0.27) | 0 | 0 | 0 |
| M_og | 95.73 (718) | 0 (0.00) | 0 (0.00) | 0 | 0 | 0 |

### GSM8K (n = 700)

| arm | acc % (correct) | unparsed (%) | truncated (%) | reading_failed | refused | too_long |
|---|---|---|---|---|---|---|
| S_ur | 87.43 (612) | 22 (3.14) | 42 (6.00) | 0 | 0 | 0 |
| S_ro | 91.14 (638) | 11 (1.57) | 29 (4.14) | 0 | 0 | 0 |
| S_og | 93.29 (653) | 12 (1.71) | 24 (3.43) | 0 | 0 | 0 |
| T_ur | 86.14 (603) | 1 (0.14) | 16 (2.29) | 0 | 0 | 0 |
| T_r1 | 94.43 (661) | 1 (0.14) | 4 (0.57) | 0 | 0 | 0 |
| T_og | 97.86 (685) | 0 (0.00) | 1 (0.14) | 0 | 0 | 0 |
| M_ur | 94.57 (662) | 0 (0.00) | 3 (0.43) | 0 | 0 | 0 |
| M_r1 | 94.43 (661) | 2 (0.29) | 7 (1.00) | 0 | 0 | 0 |
| M_og | 97.57 (683) | 0 (0.00) | 2 (0.29) | 0 | 0 | 0 |

## 3. Family G1 — reading effect (9 tests, Holm at 0.05)

First arm = the model with the English translation R1; second arm = the same model on the Urdu item. Delta = first minus second on the deciding line. Wording (freeze F6): "helps" only if delta > 0 and Holm p < 0.05; "hurts" only if delta < 0 and Holm p < 0.05; otherwise "no clear effect".

| task | test | deciding line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p | Holm p | wording |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CSQA | S_ro vs S_ur | LA | 1500 | 74.20 (1113) | 68.53 (1028) | +5.67 | 172 | 87 | 1.40e-07 | 1.06e-06 | helps |
| CSQA | T_r1 vs T_ur | LP | 1271 | 74.35 (945) | 67.11 (853) | +7.24 | 197 | 105 | 1.32e-07 | 1.06e-06 | helps |
| CSQA | M_r1 vs M_ur | LA | 1500 | 80.33 (1205) | 75.80 (1137) | +4.53 | 137 | 69 | 2.49e-06 | 1.46e-05 | helps |
| PIQA | S_ro vs S_ur | LA | 750 | 77.33 (580) | 71.60 (537) | +5.73 | 114 | 71 | 0.0019 | 0.0058 | helps |
| PIQA | T_r1 vs T_ur | LA | 750 | 86.80 (651) | 78.93 (592) | +7.87 | 107 | 48 | 2.44e-06 | 1.46e-05 | helps |
| PIQA | M_r1 vs M_ur | LA | 750 | 93.07 (698) | 92.40 (693) | +0.67 | 31 | 26 | 0.5966 | 1.0000 | no clear effect |
| GSM8K | S_ro vs S_ur | LA | 700 | 91.14 (638) | 87.43 (612) | +3.71 | 44 | 18 | 0.0013 | 0.0052 | helps |
| GSM8K | T_r1 vs T_ur | LA | 700 | 94.43 (661) | 86.14 (603) | +8.29 | 77 | 19 | 1.86e-09 | 1.68e-08 | helps |
| GSM8K | M_r1 vs M_ur | LA | 700 | 94.43 (661) | 94.57 (662) | -0.14 | 8 | 9 | 1.0000 | 1.0000 | no clear effect |

Every line of every G1 test (unparsed and truncated counts are on all rows, first arm / second arm):

| task | test | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| CSQA | S_ro vs S_ur | LA (deciding) | 1500 | 74.20 (1113) | 68.53 (1028) | +5.67 | 172 | 87 | 1.40e-07 |
| CSQA | S_ro vs S_ur | LP | 1498 | 74.30 (1113) | 68.56 (1027) | +5.74 | 172 | 86 | 9.30e-08 |
| CSQA | S_ro vs S_ur | LT | 1376 | 76.31 (1050) | 72.02 (991) | +4.29 | 126 | 67 | 2.60e-05 |
| CSQA | S_ro vs S_ur | LPT | 1375 | 76.36 (1050) | 72.07 (991) | +4.29 | 126 | 67 | 2.60e-05 |
| CSQA | T_r1 vs T_ur | LA | 1500 | 74.20 (1113) | 56.93 (854) | +17.27 | 365 | 106 | 2.40e-34 |
| CSQA | T_r1 vs T_ur | LP (deciding) | 1271 | 74.35 (945) | 67.11 (853) | +7.24 | 197 | 105 | 1.32e-07 |
| CSQA | T_r1 vs T_ur | LT | 1487 | 74.45 (1107) | 57.36 (853) | +17.08 | 359 | 105 | 1.51e-33 |
| CSQA | T_r1 vs T_ur | LPT | 1268 | 74.53 (945) | 67.19 (852) | +7.33 | 197 | 104 | 9.09e-08 |
| CSQA | M_r1 vs M_ur | LA (deciding) | 1500 | 80.33 (1205) | 75.80 (1137) | +4.53 | 137 | 69 | 2.49e-06 |
| CSQA | M_r1 vs M_ur | LP | 1499 | 80.39 (1205) | 75.78 (1136) | +4.60 | 137 | 68 | 1.65e-06 |
| CSQA | M_r1 vs M_ur | LT | 1499 | 80.39 (1205) | 75.78 (1136) | +4.60 | 137 | 68 | 1.65e-06 |
| CSQA | M_r1 vs M_ur | LPT | 1498 | 80.44 (1205) | 75.77 (1135) | +4.67 | 137 | 67 | 1.08e-06 |
| PIQA | S_ro vs S_ur | LA (deciding) | 750 | 77.33 (580) | 71.60 (537) | +5.73 | 114 | 71 | 0.0019 |
| PIQA | S_ro vs S_ur | LP | 750 | 77.33 (580) | 71.60 (537) | +5.73 | 114 | 71 | 0.0019 |
| PIQA | S_ro vs S_ur | LT | 748 | 77.27 (578) | 71.52 (535) | +5.75 | 114 | 71 | 0.0019 |
| PIQA | S_ro vs S_ur | LPT | 748 | 77.27 (578) | 71.52 (535) | +5.75 | 114 | 71 | 0.0019 |
| PIQA | T_r1 vs T_ur | LA (deciding) | 750 | 86.80 (651) | 78.93 (592) | +7.87 | 107 | 48 | 2.44e-06 |
| PIQA | T_r1 vs T_ur | LP | 742 | 86.93 (645) | 79.65 (591) | +7.28 | 101 | 47 | 1.07e-05 |
| PIQA | T_r1 vs T_ur | LT | 738 | 86.72 (640) | 79.95 (590) | +6.78 | 98 | 48 | 4.27e-05 |
| PIQA | T_r1 vs T_ur | LPT | 737 | 86.84 (640) | 79.92 (589) | +6.92 | 98 | 47 | 2.76e-05 |
| PIQA | M_r1 vs M_ur | LA (deciding) | 750 | 93.07 (698) | 92.40 (693) | +0.67 | 31 | 26 | 0.5966 |
| PIQA | M_r1 vs M_ur | LP | 750 | 93.07 (698) | 92.40 (693) | +0.67 | 31 | 26 | 0.5966 |
| PIQA | M_r1 vs M_ur | LT | 748 | 93.05 (696) | 92.38 (691) | +0.67 | 31 | 26 | 0.5966 |
| PIQA | M_r1 vs M_ur | LPT | 748 | 93.05 (696) | 92.38 (691) | +0.67 | 31 | 26 | 0.5966 |
| GSM8K | S_ro vs S_ur | LA (deciding) | 700 | 91.14 (638) | 87.43 (612) | +3.71 | 44 | 18 | 0.0013 |
| GSM8K | S_ro vs S_ur | LP | 668 | 93.41 (624) | 91.17 (609) | +2.25 | 30 | 15 | 0.0357 |
| GSM8K | S_ro vs S_ur | LT | 643 | 95.65 (615) | 93.78 (603) | +1.87 | 21 | 9 | 0.0428 |
| GSM8K | S_ro vs S_ur | LPT | 643 | 95.65 (615) | 93.78 (603) | +1.87 | 21 | 9 | 0.0428 |
| GSM8K | T_r1 vs T_ur | LA (deciding) | 700 | 94.43 (661) | 86.14 (603) | +8.29 | 77 | 19 | 1.86e-09 |
| GSM8K | T_r1 vs T_ur | LP | 698 | 94.56 (660) | 86.39 (603) | +8.17 | 76 | 19 | 3.00e-09 |
| GSM8K | T_r1 vs T_ur | LT | 681 | 95.01 (647) | 88.25 (601) | +6.75 | 64 | 18 | 3.32e-07 |
| GSM8K | T_r1 vs T_ur | LPT | 681 | 95.01 (647) | 88.25 (601) | +6.75 | 64 | 18 | 3.32e-07 |
| GSM8K | M_r1 vs M_ur | LA (deciding) | 700 | 94.43 (661) | 94.57 (662) | -0.14 | 8 | 9 | 1.0000 |
| GSM8K | M_r1 vs M_ur | LP | 698 | 94.70 (661) | 94.84 (662) | -0.14 | 8 | 9 | 1.0000 |
| GSM8K | M_r1 vs M_ur | LT | 692 | 95.52 (661) | 95.38 (660) | +0.14 | 8 | 7 | 1.0000 |
| GSM8K | M_r1 vs M_ur | LPT | 692 | 95.52 (661) | 95.38 (660) | +0.14 | 8 | 7 | 1.0000 |

- CSQA S_ro vs S_ur: unparsed 2 / 0 (0.13% / 0.00%), truncated 54 / 83 (3.60% / 5.53%); deciding line LA (all rows).
- CSQA T_r1 vs T_ur: unparsed 1 / 228 (0.07% / 15.20%), truncated 4 / 9 (0.27% / 0.60%); deciding line LP (rows parsed in both arms).
- CSQA M_r1 vs M_ur: unparsed 1 / 0 (0.07% / 0.00%), truncated 1 / 0 (0.07% / 0.00%); deciding line LA (all rows).
- PIQA S_ro vs S_ur: unparsed 0 / 0 (0.00% / 0.00%), truncated 2 / 1 (0.27% / 0.13%); deciding line LA (all rows).
- PIQA T_r1 vs T_ur: unparsed 1 / 7 (0.13% / 0.93%), truncated 0 / 12 (0.00% / 1.60%); deciding line LA (all rows).
- PIQA M_r1 vs M_ur: unparsed 0 / 0 (0.00% / 0.00%), truncated 2 / 0 (0.27% / 0.00%); deciding line LA (all rows).
- GSM8K S_ro vs S_ur: unparsed 11 / 22 (1.57% / 3.14%), truncated 29 / 42 (4.14% / 6.00%); deciding line LA (all rows).
- GSM8K T_r1 vs T_ur: unparsed 1 / 1 (0.14% / 0.14%), truncated 4 / 16 (0.57% / 2.29%); deciding line LA (all rows).
- GSM8K M_r1 vs M_ur: unparsed 2 / 0 (0.29% / 0.00%), truncated 7 / 3 (1.00% / 0.43%); deciding line LA (all rows).

G1 wording:

- CSQA, S_ro vs S_ur: the English translation: helps.
- CSQA, T_r1 vs T_ur: the English translation: helps.
- CSQA, M_r1 vs M_ur: the English translation: helps.
- PIQA, S_ro vs S_ur: the English translation: helps.
- PIQA, T_r1 vs T_ur: the English translation: helps.
- PIQA, M_r1 vs M_ur: the English translation: no clear effect.
- GSM8K, S_ro vs S_ur: the English translation: helps.
- GSM8K, T_r1 vs T_ur: the English translation: helps.
- GSM8K, M_r1 vs M_ur: the English translation: no clear effect.

## 4. Family G2 — method (2 tests, Holm at 0.05)

First arm = S_tr (Qwen3-14B with teacher facts); second arm = S_ur (Qwen3-14B on the Urdu item alone). Same wording rule.

| task | test | deciding line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p | Holm p | wording |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CSQA | S_tr vs S_ur | LA | 1500 | 70.00 (1050) | 68.53 (1028) | +1.47 | 154 | 132 | 0.2143 | 0.4285 | no clear effect |
| PIQA | S_tr vs S_ur | LA | 750 | 72.40 (543) | 71.60 (537) | +0.80 | 106 | 100 | 0.7277 | 0.7277 | no clear effect |

Every line of every G2 test (unparsed and truncated counts are on all rows, first arm / second arm):

| task | test | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| CSQA | S_tr vs S_ur | LA (deciding) | 1500 | 70.00 (1050) | 68.53 (1028) | +1.47 | 154 | 132 | 0.2143 |
| CSQA | S_tr vs S_ur | LP | 1498 | 70.09 (1050) | 68.62 (1028) | +1.47 | 154 | 132 | 0.2143 |
| CSQA | S_tr vs S_ur | LT | 1282 | 76.76 (984) | 73.40 (941) | +3.35 | 115 | 72 | 0.0020 |
| CSQA | S_tr vs S_ur | LPT | 1281 | 76.81 (984) | 73.46 (941) | +3.36 | 115 | 72 | 0.0020 |
| PIQA | S_tr vs S_ur | LA (deciding) | 750 | 72.40 (543) | 71.60 (537) | +0.80 | 106 | 100 | 0.7277 |
| PIQA | S_tr vs S_ur | LP | 750 | 72.40 (543) | 71.60 (537) | +0.80 | 106 | 100 | 0.7277 |
| PIQA | S_tr vs S_ur | LT | 738 | 72.63 (536) | 71.68 (529) | +0.95 | 104 | 97 | 0.6722 |
| PIQA | S_tr vs S_ur | LPT | 738 | 72.63 (536) | 71.68 (529) | +0.95 | 104 | 97 | 0.6722 |

- CSQA S_tr vs S_ur: unparsed 2 / 0 (0.13% / 0.00%), truncated 154 / 83 (10.27% / 5.53%); deciding line LA (all rows).
- PIQA S_tr vs S_ur: unparsed 0 / 0 (0.00% / 0.00%), truncated 12 / 1 (1.60% / 0.13%); deciding line LA (all rows).

G2 wording:

- CSQA, S_tr vs S_ur: the teacher facts: no clear effect.
- PIQA, S_tr vs S_ur: the teacher facts: no clear effect.

## 5. Descriptive comparisons (no claim, raw p only)

| task | comparison | line | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|---|
| CSQA | S_og vs S_ur | LA (deciding) | 1500 | 78.93 (1184) | 68.53 (1028) | +10.40 | 227 | 71 | 3.53e-20 |
| CSQA | S_og vs S_ur | LP | 1498 | 79.04 (1184) | 68.56 (1027) | +10.48 | 227 | 70 | 1.67e-20 |
| CSQA | S_og vs S_ur | LT | 1367 | 81.42 (1113) | 72.28 (988) | +9.14 | 177 | 52 | 3.89e-17 |
| CSQA | S_og vs S_ur | LPT | 1367 | 81.42 (1113) | 72.28 (988) | +9.14 | 177 | 52 | 3.89e-17 |
| CSQA | T_og vs T_ur | LA | 1500 | 81.67 (1225) | 56.93 (854) | +24.73 | 451 | 80 | 9.27e-64 |
| CSQA | T_og vs T_ur | LP (deciding) | 1272 | 81.68 (1039) | 67.14 (854) | +14.54 | 265 | 80 | 2.79e-24 |
| CSQA | T_og vs T_ur | LT | 1490 | 81.68 (1217) | 57.32 (854) | +24.36 | 443 | 80 | 6.39e-62 |
| CSQA | T_og vs T_ur | LPT | 1271 | 81.75 (1039) | 67.19 (854) | +14.56 | 265 | 80 | 2.79e-24 |
| CSQA | M_og vs M_ur | LA (deciding) | 1500 | 88.40 (1326) | 75.80 (1137) | +12.60 | 239 | 50 | 1.02e-30 |
| CSQA | M_og vs M_ur | LP | 1500 | 88.40 (1326) | 75.80 (1137) | +12.60 | 239 | 50 | 1.02e-30 |
| CSQA | M_og vs M_ur | LT | 1500 | 88.40 (1326) | 75.80 (1137) | +12.60 | 239 | 50 | 1.02e-30 |
| CSQA | M_og vs M_ur | LPT | 1500 | 88.40 (1326) | 75.80 (1137) | +12.60 | 239 | 50 | 1.02e-30 |
| CSQA | S_og vs S_ro | LA (deciding) | 1500 | 78.93 (1184) | 74.20 (1113) | +4.73 | 143 | 72 | 1.47e-06 |
| CSQA | S_og vs S_ro | LP | 1496 | 79.08 (1183) | 74.33 (1112) | +4.75 | 142 | 71 | 1.30e-06 |
| CSQA | S_og vs S_ro | LT | 1396 | 81.73 (1141) | 77.15 (1077) | +4.58 | 116 | 52 | 8.74e-07 |
| CSQA | S_og vs S_ro | LPT | 1395 | 81.72 (1140) | 77.20 (1077) | +4.52 | 115 | 52 | 1.21e-06 |
| CSQA | T_og vs T_r1 | LA (deciding) | 1500 | 81.67 (1225) | 74.20 (1113) | +7.47 | 186 | 74 | 2.68e-12 |
| CSQA | T_og vs T_r1 | LP | 1499 | 81.65 (1224) | 74.25 (1113) | +7.40 | 185 | 74 | 3.84e-12 |
| CSQA | T_og vs T_r1 | LT | 1496 | 81.75 (1223) | 74.40 (1113) | +7.35 | 184 | 74 | 5.51e-12 |
| CSQA | T_og vs T_r1 | LPT | 1495 | 81.74 (1222) | 74.45 (1113) | +7.29 | 183 | 74 | 7.89e-12 |
| CSQA | M_og vs M_r1 | LA (deciding) | 1500 | 88.40 (1326) | 80.33 (1205) | +8.07 | 155 | 34 | 1.17e-19 |
| CSQA | M_og vs M_r1 | LP | 1499 | 88.39 (1325) | 80.39 (1205) | +8.01 | 154 | 34 | 1.92e-19 |
| CSQA | M_og vs M_r1 | LT | 1499 | 88.39 (1325) | 80.39 (1205) | +8.01 | 154 | 34 | 1.92e-19 |
| CSQA | M_og vs M_r1 | LPT | 1498 | 88.38 (1324) | 80.44 (1205) | +7.94 | 153 | 34 | 3.16e-19 |
| CSQA | S_tr vs S_ro | LA | 1500 | 70.00 (1050) | 74.20 (1113) | -4.20 | 98 | 161 | 0.0001 |
| CSQA | S_tr vs S_ro | LP | 1497 | 70.07 (1049) | 74.35 (1113) | -4.28 | 97 | 161 | 8.12e-05 |
| CSQA | S_tr vs S_ro | LT (deciding) | 1308 | 76.83 (1005) | 78.75 (1030) | -1.91 | 73 | 98 | 0.0661 |
| CSQA | S_tr vs S_ro | LPT | 1307 | 76.89 (1005) | 78.81 (1030) | -1.91 | 73 | 98 | 0.0661 |
| CSQA | S_tr vs T_r1 | LA | 1500 | 70.00 (1050) | 74.20 (1113) | -4.20 | 122 | 185 | 0.0004 |
| CSQA | S_tr vs T_r1 | LP | 1498 | 70.09 (1050) | 74.30 (1113) | -4.21 | 122 | 185 | 0.0004 |
| CSQA | S_tr vs T_r1 | LT (deciding) | 1343 | 75.80 (1018) | 77.44 (1040) | -1.64 | 108 | 130 | 0.1733 |
| CSQA | S_tr vs T_r1 | LPT | 1342 | 75.86 (1018) | 77.50 (1040) | -1.64 | 108 | 130 | 0.1733 |
| PIQA | S_og vs S_ur | LA (deciding) | 750 | 78.40 (588) | 71.60 (537) | +6.80 | 128 | 77 | 0.0005 |
| PIQA | S_og vs S_ur | LP | 750 | 78.40 (588) | 71.60 (537) | +6.80 | 128 | 77 | 0.0005 |
| PIQA | S_og vs S_ur | LT | 748 | 78.34 (586) | 71.52 (535) | +6.82 | 128 | 77 | 0.0005 |
| PIQA | S_og vs S_ur | LPT | 748 | 78.34 (586) | 71.52 (535) | +6.82 | 128 | 77 | 0.0005 |
| PIQA | T_og vs T_ur | LA (deciding) | 750 | 88.13 (661) | 78.93 (592) | +9.20 | 113 | 44 | 3.49e-08 |
| PIQA | T_og vs T_ur | LP | 742 | 88.27 (655) | 79.65 (591) | +8.63 | 107 | 43 | 1.77e-07 |
| PIQA | T_og vs T_ur | LT | 738 | 88.08 (650) | 79.95 (590) | +8.13 | 104 | 44 | 8.87e-07 |
| PIQA | T_og vs T_ur | LPT | 737 | 88.20 (650) | 79.92 (589) | +8.28 | 104 | 43 | 5.20e-07 |
| PIQA | M_og vs M_ur | LA (deciding) | 750 | 95.73 (718) | 92.40 (693) | +3.33 | 41 | 16 | 0.0013 |
| PIQA | M_og vs M_ur | LP | 750 | 95.73 (718) | 92.40 (693) | +3.33 | 41 | 16 | 0.0013 |
| PIQA | M_og vs M_ur | LT | 750 | 95.73 (718) | 92.40 (693) | +3.33 | 41 | 16 | 0.0013 |
| PIQA | M_og vs M_ur | LPT | 750 | 95.73 (718) | 92.40 (693) | +3.33 | 41 | 16 | 0.0013 |
| PIQA | S_og vs S_ro | LA (deciding) | 750 | 78.40 (588) | 77.33 (580) | +1.07 | 84 | 76 | 0.5801 |
| PIQA | S_og vs S_ro | LP | 750 | 78.40 (588) | 77.33 (580) | +1.07 | 84 | 76 | 0.5801 |
| PIQA | S_og vs S_ro | LT | 747 | 78.31 (585) | 77.24 (577) | +1.07 | 84 | 76 | 0.5801 |
| PIQA | S_og vs S_ro | LPT | 747 | 78.31 (585) | 77.24 (577) | +1.07 | 84 | 76 | 0.5801 |
| PIQA | T_og vs T_r1 | LA (deciding) | 750 | 88.13 (661) | 86.80 (651) | +1.33 | 45 | 35 | 0.3143 |
| PIQA | T_og vs T_r1 | LP | 749 | 88.25 (661) | 86.92 (651) | +1.34 | 45 | 35 | 0.3143 |
| PIQA | T_og vs T_r1 | LT | 750 | 88.13 (661) | 86.80 (651) | +1.33 | 45 | 35 | 0.3143 |
| PIQA | T_og vs T_r1 | LPT | 749 | 88.25 (661) | 86.92 (651) | +1.34 | 45 | 35 | 0.3143 |
| PIQA | M_og vs M_r1 | LA (deciding) | 750 | 95.73 (718) | 93.07 (698) | +2.67 | 34 | 14 | 0.0055 |
| PIQA | M_og vs M_r1 | LP | 750 | 95.73 (718) | 93.07 (698) | +2.67 | 34 | 14 | 0.0055 |
| PIQA | M_og vs M_r1 | LT | 748 | 95.72 (716) | 93.05 (696) | +2.67 | 34 | 14 | 0.0055 |
| PIQA | M_og vs M_r1 | LPT | 748 | 95.72 (716) | 93.05 (696) | +2.67 | 34 | 14 | 0.0055 |
| PIQA | S_tr vs S_ro | LA (deciding) | 750 | 72.40 (543) | 77.33 (580) | -4.93 | 76 | 113 | 0.0087 |
| PIQA | S_tr vs S_ro | LP | 750 | 72.40 (543) | 77.33 (580) | -4.93 | 76 | 113 | 0.0087 |
| PIQA | S_tr vs S_ro | LT | 737 | 72.59 (535) | 77.34 (570) | -4.75 | 75 | 110 | 0.0122 |
| PIQA | S_tr vs S_ro | LPT | 737 | 72.59 (535) | 77.34 (570) | -4.75 | 75 | 110 | 0.0122 |
| PIQA | S_tr vs T_r1 | LA (deciding) | 750 | 72.40 (543) | 86.80 (651) | -14.40 | 45 | 153 | 5.78e-15 |
| PIQA | S_tr vs T_r1 | LP | 749 | 72.36 (542) | 86.92 (651) | -14.55 | 44 | 153 | 2.60e-15 |
| PIQA | S_tr vs T_r1 | LT | 738 | 72.63 (536) | 86.86 (641) | -14.23 | 45 | 150 | 2.14e-14 |
| PIQA | S_tr vs T_r1 | LPT | 737 | 72.59 (535) | 86.97 (641) | -14.38 | 44 | 150 | 9.79e-15 |
| GSM8K | S_og vs S_ur | LA (deciding) | 700 | 93.29 (653) | 87.43 (612) | +5.86 | 49 | 8 | 2.72e-08 |
| GSM8K | S_og vs S_ur | LP | 669 | 95.22 (637) | 91.18 (610) | +4.04 | 33 | 6 | 1.43e-05 |
| GSM8K | S_og vs S_ur | LT | 647 | 96.91 (627) | 93.82 (607) | +3.09 | 23 | 3 | 8.80e-05 |
| GSM8K | S_og vs S_ur | LPT | 647 | 96.91 (627) | 93.82 (607) | +3.09 | 23 | 3 | 8.80e-05 |
| GSM8K | T_og vs T_ur | LA (deciding) | 700 | 97.86 (685) | 86.14 (603) | +11.71 | 87 | 5 | 2.10e-20 |
| GSM8K | T_og vs T_ur | LP | 699 | 97.85 (684) | 86.27 (603) | +11.59 | 86 | 5 | 3.98e-20 |
| GSM8K | T_og vs T_ur | LT | 683 | 97.95 (669) | 88.14 (602) | +9.81 | 72 | 5 | 2.80e-16 |
| GSM8K | T_og vs T_ur | LPT | 683 | 97.95 (669) | 88.14 (602) | +9.81 | 72 | 5 | 2.80e-16 |
| GSM8K | M_og vs M_ur | LA (deciding) | 700 | 97.57 (683) | 94.57 (662) | +3.00 | 23 | 2 | 1.94e-05 |
| GSM8K | M_og vs M_ur | LP | 700 | 97.57 (683) | 94.57 (662) | +3.00 | 23 | 2 | 1.94e-05 |
| GSM8K | M_og vs M_ur | LT | 695 | 97.99 (681) | 95.25 (662) | +2.73 | 21 | 2 | 6.60e-05 |
| GSM8K | M_og vs M_ur | LPT | 695 | 97.99 (681) | 95.25 (662) | +2.73 | 21 | 2 | 6.60e-05 |
| GSM8K | S_og vs S_ro | LA (deciding) | 700 | 93.29 (653) | 91.14 (638) | +2.14 | 29 | 14 | 0.0315 |
| GSM8K | S_og vs S_ro | LP | 681 | 95.15 (648) | 93.39 (636) | +1.76 | 24 | 12 | 0.0652 |
| GSM8K | S_og vs S_ro | LT | 661 | 96.52 (638) | 95.31 (630) | +1.21 | 14 | 6 | 0.1153 |
| GSM8K | S_og vs S_ro | LPT | 661 | 96.52 (638) | 95.31 (630) | +1.21 | 14 | 6 | 0.1153 |
| GSM8K | T_og vs T_r1 | LA (deciding) | 700 | 97.86 (685) | 94.43 (661) | +3.43 | 28 | 4 | 1.93e-05 |
| GSM8K | T_og vs T_r1 | LP | 699 | 97.85 (684) | 94.56 (661) | +3.29 | 27 | 4 | 3.40e-05 |
| GSM8K | T_og vs T_r1 | LT | 696 | 97.99 (682) | 94.97 (661) | +3.02 | 25 | 4 | 0.0001 |
| GSM8K | T_og vs T_r1 | LPT | 696 | 97.99 (682) | 94.97 (661) | +3.02 | 25 | 4 | 0.0001 |
| GSM8K | M_og vs M_r1 | LA (deciding) | 700 | 97.57 (683) | 94.43 (661) | +3.14 | 25 | 3 | 2.74e-05 |
| GSM8K | M_og vs M_r1 | LP | 698 | 97.71 (682) | 94.70 (661) | +3.01 | 24 | 3 | 4.92e-05 |
| GSM8K | M_og vs M_r1 | LT | 693 | 97.84 (678) | 95.38 (661) | +2.45 | 20 | 3 | 0.0005 |
| GSM8K | M_og vs M_r1 | LPT | 693 | 97.84 (678) | 95.38 (661) | +2.45 | 20 | 3 | 0.0005 |

## 6. Predictions (freeze F7, read on LA = all rows)

| prediction | task | first % | second % | delta pp | result |
|---|---|---|---|---|---|
| S_ro - S_ur in [0, +8] | CSQA | 74.20 | 68.53 | +5.67 | hit |
| S_ro - S_ur in [-2, +8] | PIQA | 77.33 | 71.60 | +5.73 | hit |
| S_ro - S_ur in [-2, +4] | GSM8K | 91.14 | 87.43 | +3.71 | hit |
| T_r1 - T_ur in [0, +10] | CSQA | 74.20 | 56.93 | +17.27 | miss |
| T_r1 - T_ur in [0, +10] | PIQA | 86.80 | 78.93 | +7.87 | hit |
| T_r1 - T_ur in [0, +10] | GSM8K | 94.43 | 86.14 | +8.29 | hit |
| M_r1 - M_ur in [0, +10] | CSQA | 80.33 | 75.80 | +4.53 | hit |
| M_r1 - M_ur in [0, +10] | PIQA | 93.07 | 92.40 | +0.67 | hit |
| M_r1 - M_ur in [0, +10] | GSM8K | 94.43 | 94.57 | -0.14 | miss |
| S_tr - S_ur in [-3, +6] | CSQA | 70.00 | 68.53 | +1.47 | hit |
| S_tr - S_ur in [-3, +6] | PIQA | 72.40 | 71.60 | +0.80 | hit |
| S_og is at least S_ro minus 2 | CSQA | 78.93 | 74.20 | +4.73 | hit |
| S_og is at least S_ro minus 2 | PIQA | 78.40 | 77.33 | +1.07 | hit |
| S_og is at least S_ro minus 2 | GSM8K | 93.29 | 91.14 | +2.14 | hit |
| T_og is at least T_r1 minus 2 | CSQA | 81.67 | 74.20 | +7.47 | hit |
| T_og is at least T_r1 minus 2 | PIQA | 88.13 | 86.80 | +1.33 | hit |
| T_og is at least T_r1 minus 2 | GSM8K | 97.86 | 94.43 | +3.43 | hit |
| M_og is at least M_r1 minus 2 | CSQA | 88.40 | 80.33 | +8.07 | hit |
| M_og is at least M_r1 minus 2 | PIQA | 95.73 | 93.07 | +2.67 | hit |
| M_og is at least M_r1 minus 2 | GSM8K | 97.57 | 94.43 | +3.14 | hit |

Predictions hit: 18 of 20.

## 7. S_ur against the sealed fair baseline on the shared rows (agreement, reported only)

The sealed fair outputs are the evaluation splits: the last rows of the CSQA and PIQA source files and all GSM8K rows. S_ur uses the same prompt, engine arguments and sampling; the rows were generated in other batches, so the outputs need not be identical.

| task | shared rows | sealed fair acc % (correct) | S_ur acc % (correct) on these rows | both right | fair only | S_ur only | both wrong | same prediction |
|---|---|---|---|---|---|---|---|---|
| CSQA | 300 | 63.33 (190) | 64.67 (194) | 174 | 16 | 20 | 90 | 246 |
| PIQA | 150 | 72.00 (108) | 70.67 (106) | 93 | 15 | 13 | 29 | 122 |
| GSM8K | 700 | 88.71 (621) | 87.43 (612) | 602 | 19 | 10 | 69 | 647 |

## 8. Readings and teacher facts

| task | rows | R1 file MD5 | requests | reading_failed | accepted readings that reached max_tokens |
|---|---|---|---|---|---|
| CSQA | 1500 | `fc1f6663d7a8e009dd00dcd7920b8239` | 1501 | 1 | 0 |
| PIQA | 750 | `308ba4b28e73cc032674a1f2c35d8724` | 750 | 0 | 2 |
| GSM8K | 700 | `3dcea6dd06cdc6fdef410f17e525f470` | 700 | 0 | 0 |

A reading_failed row is not generated in S_ro, S_tr, T_r1 and M_r1 and counts as unparsed = wrong there.

| task | rows | facts file MD5 | replies parsed | facts kept | lines removed by the D10 verdict filter | lines removed by the D14 filter | rows with a line removed by the D14 filter | rows without facts |
|---|---|---|---|---|---|---|---|---|
| CSQA | 1500 | `77ab2a53c4526c28c1b82394d666775d` | 1498 | 8834 | 5 | 34 | 32 | 2 |
| PIQA | 750 | `e0a0f4d07fcf3744edca995b17201b8d` | 749 | 4470 | 0 | 17 | 13 | 1 |

A row without facts has no facts block: its S_tr prompt equals its S_ur prompt (freeze 9.6).

Gold: taken from the Urdu source row with the fair script's rule. Positions where the English source row gives another gold: CSQA 0; PIQA 5; GSM8K 0.

## 9. Seconds and tokens

Seconds of a vLLM row = the seconds of its chunk divided by the rows generated in the chunk (up to 64 prompts per call); seconds of an API row = the latency of its answered request.

| task | arm | rows generated | seconds total | seconds per row | prompt tokens per row | new tokens per row |
|---|---|---|---|---|---|---|
| CSQA | S_ur | 1500 | 1208 | 0.81 | 244.6 | 452.3 |
| CSQA | S_ro | 1499 | 1211 | 0.81 | 297.6 | 416.7 |
| CSQA | S_og | 1500 | 1208 | 0.81 | 296.1 | 402.2 |
| CSQA | S_tr | 1499 | 1289 | 0.86 | 357.1 | 472.4 |
| CSQA | T_ur | 1500 | 1696 | 1.13 | 261.8 | 75.0 |
| CSQA | T_r1 | 1499 | 1931 | 1.29 | 138.0 | 202.6 |
| CSQA | T_og | 1500 | 1780 | 1.19 | 136.5 | 194.6 |
| CSQA | M_ur | 1500 | 7832 | 5.22 | 177.5 | 172.3 |
| CSQA | M_r1 | 1499 | 5220 | 3.48 | 119.9 | 99.2 |
| CSQA | M_og | 1500 | 5059 | 3.37 | 118.5 | 92.2 |
| PIQA | S_ur | 750 | 838 | 1.12 | 364.0 | 477.7 |
| PIQA | S_ro | 750 | 816 | 1.09 | 443.5 | 435.6 |
| PIQA | S_og | 750 | 838 | 1.12 | 444.8 | 422.1 |
| PIQA | S_tr | 750 | 1082 | 1.44 | 478.0 | 537.3 |
| PIQA | T_ur | 750 | 3178 | 4.24 | 396.3 | 407.8 |
| PIQA | T_r1 | 750 | 1177 | 1.57 | 171.3 | 262.7 |
| PIQA | T_og | 750 | 1184 | 1.58 | 172.4 | 259.6 |
| PIQA | M_ur | 750 | 8092 | 10.79 | 248.8 | 421.5 |
| PIQA | M_r1 | 750 | 4045 | 5.39 | 155.6 | 192.7 |
| PIQA | M_og | 750 | 4006 | 5.34 | 156.8 | 187.0 |
| GSM8K | S_ur | 700 | 1083 | 1.55 | 316.7 | 608.7 |
| GSM8K | S_ro | 700 | 1072 | 1.53 | 389.9 | 540.9 |
| GSM8K | S_og | 700 | 1069 | 1.53 | 389.1 | 520.6 |
| GSM8K | T_ur | 700 | 2098 | 3.00 | 338.9 | 280.9 |
| GSM8K | T_r1 | 700 | 852 | 1.22 | 148.9 | 183.4 |
| GSM8K | T_og | 700 | 821 | 1.17 | 148.1 | 175.2 |
| GSM8K | M_ur | 700 | 3543 | 5.06 | 218.7 | 207.8 |
| GSM8K | M_r1 | 700 | 2891 | 4.13 | 133.1 | 154.8 |
| GSM8K | M_og | 700 | 2638 | 3.77 | 132.3 | 142.5 |

API ledger at the time of the readout (TEST-5 and retries included): 11860 calls of at most 13000; 3639718 tokens of at most 7000000 (prompt 1972524, completion 1667194); stopped: no.

