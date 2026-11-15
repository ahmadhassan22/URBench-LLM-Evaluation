# D13 readout — baselines and transfer

Written by `eval/error_analysis_tests/efbpt/d13_readout.py` (MD5 `c8c882c72258e9bd01ee8e162a9ce3eb`, git HEAD `889d12bc948bdc517f2041af10e6f3d8588adb27`), one run, stage 1 (Groups 1-3) and stage 2 (Group 4) together: all 19 D13 arm files were sealed before it started. Contract: freeze `2ee24716b0e0ca2db0cc5f8fb986ac08`; Amendments 1 `26bf37ea7e17146d44916b83cfa6e5d5`, 2 `12809c115ae8b6fba6bf6afe7fb6c51c`, 3 `344590e857e9f3ac075519120892e5d6`. E0 = the frozen extractor; E3 = `d11_scorer_audit.e3` (script MD5 `246cf6c9687b3fa67324722cb5cd16b2`); E0fa (freeze §7.1) for T_tr_R1 and T_tr_L1. "label" = E0, or E0fa for the two T_tr arms. b = the first-named arm right and the second wrong; c = the reverse; p = exact two-sided McNemar; unparsed = wrong; Δ = first-named minus second, in pp. eval458 decides (n = 458, gold yes 221 / no 237); DEV200 is descriptive (n = 200, 87 / 113).

The D11 verdict CONFIRMED stays as it is whatever this readout shows (freeze, first page). No arm is run again after this readout.

## 0. Self-test (run before any D13 arm was opened; every check fatal)

- S1 PASS: the 6 sealed DEV200 reference arms give the E0 numbers of freeze §1 (A_r2 60.00 (120), TR 68.50 (137), TR_L1 67.00 (134), T_direct_F 55.50 (111), API_direct_ur 69.00 (138), RO 57.50 (115)); TR vs A_r2 and TR_L1 vs A_r2 give the L1-L4 rows and the per-arm E0 / E3 rows of `docs/handoff/D11_DEV200_READOUT_LINES.md` (MD5 `888b477fd42c9d2886e384655bab2443`), 92 cells compared as printed.
- S2 PASS: the 3 sealed D11 eval458 arms give the E0 numbers of freeze §1 (A_r2_eval 54.80 (251), TR 68.78 (315), TR_L1_eval 64.63 (296)); TR vs A_r2_eval and TR_L1_eval vs A_r2_eval give the L1-L4 rows and the per-arm E0 / E3 rows of `docs/handoff/D11_EVAL458_READOUT_LINES.md` (MD5 `9f95a95ecd0a92fe6362b438c3f900a4`), 92 cells compared as printed.
- S3 PASS: Holm step-down, 7 hand-made cases (ties, the cap at 1, the running maximum, input order kept).
- S4 PASS: E0fa, 11 hand-made cases (last marker, first yes/no after it, case, word bounds, marker without an answer = unparsed, no marker = E0).
- S5 PASS: hand-made rows: L1, L3, L5, L6, L7 counts and p, the deciding-line rule at and above 5.0 pp, a refused row (wrong, unparsed, not truncated, left out of L3 and of the A3.3 line), the wording rules, the title share.

## 1. Sealed files

| set | arm | file | MD5 | seal |
|---|---|---|---|---|
| eval458 | A_r2_eval | `outputs/efbpt/d11/student/eval458/d11_armA_r2_eval.jsonl` | `f139ba9247cdd9fef191f00d404d8212` | sealed reference, MD5 pinned (freeze §1) |
| eval458 | TR | `outputs/efbpt/d11/student/eval458/d11_armTR.jsonl` | `e38d64114089a88b2deb202de0240ca9` | sealed reference, MD5 pinned (freeze §1) |
| eval458 | TR_L1_eval | `outputs/efbpt/d11/student/eval458/d11_armTR_L1_eval.jsonl` | `4a1b2118bbcaa9348713a600b9281733` | sealed reference, MD5 pinned (freeze §1) |
| eval458 | T_direct_F_eval | `outputs/efbpt/d13/large_alone/eval458/d13_T_direct_F_eval.jsonl` | `3bfec18ccddffdcec20b185ff5434d00` | outputs/efbpt/d13/large_alone/eval458/SEAL_T_direct_F_eval.json (md5 d9b7bc1e46a5b9a7b87183d077f6d7bd; slurm 116802; 2026-11-13 13:40:44; 2 files verified) |
| eval458 | T_tr_R1 | `outputs/efbpt/d13/large_alone/eval458/d13_T_tr_R1.jsonl` | `e68ea3058fd168cd7e6304c18515d880` | outputs/efbpt/d13/large_alone/eval458/SEAL_T_tr_R1.json (md5 6f05834b63294c40318d99c92a7525b6; slurm 116802; 2026-11-13 14:57:30; 2 files verified) |
| eval458 | T_tr_L1 | `outputs/efbpt/d13/large_alone/eval458/d13_T_tr_L1.jsonl` | `bf3543d1bb673772bd79158a2c689408` | outputs/efbpt/d13/large_alone/eval458/SEAL_T_tr_L1.json (md5 f80dad56faac519363e064e000ef967a; slurm 116802; 2026-11-13 16:13:39; 2 files verified) |
| eval458 | API_direct_ur_eval | `outputs/efbpt/d13/large_alone/eval458/d13_API_direct_ur_eval.jsonl` | `9531a4a66ff937958883f99d5db3434b` | outputs/efbpt/d13/large_alone/eval458/SEAL_API_direct_ur_eval.json (md5 00de2a2f2f5453753ed97fa28da36b91; login node, no slurm job; 2026-11-14 14:29:21; 2 files verified) |
| eval458 | API_tr_R1 | `outputs/efbpt/d13/large_alone/eval458/d13_API_tr_R1.jsonl` | `3519334ee98de0192a43fd75ee9f3be7` | outputs/efbpt/d13/large_alone/eval458/SEAL_API_tr_R1.json (md5 f7260df3f5221be9c4027769753bd823; login node, no slurm job; 2026-11-15 14:10:07; 2 files verified) |
| eval458 | RAG_ur | `outputs/efbpt/d13/rag/eval458/d13_RAG_ur.jsonl` | `e2ac1572a65644b5626985352e2792dc` | outputs/efbpt/d13/rag/eval458/SEAL_RAG_ur.json (md5 5b95192f0fd20a61d51a5fd5aacae604; slurm 117477; 2026-11-14 20:13:21; 2 files verified) |
| eval458 | RAG_en | `outputs/efbpt/d13/rag/eval458/d13_RAG_en.jsonl` | `57334d669068f4a713826d42498270c0` | outputs/efbpt/d13/rag/eval458/SEAL_RAG_en.json (md5 e1062497473191c000a77c8a6fa2ecc0; slurm 117478; 2026-11-14 23:34:19; 2 files verified) |
| eval458 | RO_eval | `outputs/efbpt/d13/ablation/eval458/d13_RO_eval.jsonl` | `163cbfeb6c741ba48c3faffc92ed4ae2` | outputs/efbpt/d13/ablation/eval458/SEAL_RO_eval.json (md5 f827ccf12b43baf8b73dac1d8f51c123; slurm 117480; 2026-11-14 18:23:25; 2 files verified) |
| eval458 | Llama8B_A | `outputs/efbpt/d13/transfer/eval458/d13_Llama8B_A.jsonl` | `742ebc8f56d060e7a053459006c53b9e` | outputs/efbpt/d13/transfer/eval458/SEAL_Llama8B_A.json (md5 cc9025afbf766291f6bac7783a9fdf6a; slurm 117481; 2026-11-15 02:43:32; 2 files verified) |
| eval458 | Llama8B_TR | `outputs/efbpt/d13/transfer/eval458/d13_Llama8B_TR.jsonl` | `25c3d30630bc64547398aab105f7cc81` | outputs/efbpt/d13/transfer/eval458/SEAL_Llama8B_TR.json (md5 dff344ca0489f81a643a345858077db6; slurm 117481; 2026-11-15 04:28:06; 2 files verified) |
| eval458 | Gemma9B_A | `outputs/efbpt/d13/transfer/eval458/d13_Gemma9B_A.jsonl` | `bafddee3cdac3bc3b27459b0257f1e0a` | outputs/efbpt/d13/transfer/eval458/SEAL_Gemma9B_A.json (md5 2e809c3076959d34a8f8e457441bbfed; slurm 117482; 2026-11-15 02:35:55; 2 files verified) |
| eval458 | Gemma9B_TR | `outputs/efbpt/d13/transfer/eval458/d13_Gemma9B_TR.jsonl` | `4e223aed0fc73888c1596af133b1370e` | outputs/efbpt/d13/transfer/eval458/SEAL_Gemma9B_TR.json (md5 55be463bea949855105c0b662e694fee; slurm 117482; 2026-11-15 03:40:24; 2 files verified) |
| eval458 | Qwen8B_A | `outputs/efbpt/d13/transfer/eval458/d13_Qwen8B_A.jsonl` | `1ead3c47fbff6dbe2c02996871fb62fa` | outputs/efbpt/d13/transfer/eval458/SEAL_Qwen8B_A.json (md5 ab5337f0372979aafc199618e7b50668; slurm 117483; 2026-11-15 08:06:49; 2 files verified) |
| eval458 | Qwen8B_TR | `outputs/efbpt/d13/transfer/eval458/d13_Qwen8B_TR.jsonl` | `62325ec5c189d9f9ce24cc99a543bae3` | outputs/efbpt/d13/transfer/eval458/SEAL_Qwen8B_TR.json (md5 e1cc5b9312f8ffb446ddf9e5a88256b9; slurm 117483; 2026-11-15 11:33:23; 2 files verified) |
| dev200 | A_r2 | `outputs/efbpt/d10/student/d10_armA_r2.jsonl` | `7ccec942c8f0eee08156d258baaf806d` | sealed reference, MD5 pinned (freeze §1) |
| dev200 | TR | `outputs/efbpt/d11/student/dev200/d11_armTR.jsonl` | `bab987cc492012f4e32dbcd348be9668` | sealed reference, MD5 pinned (freeze §1) |
| dev200 | TR_L1 | `outputs/efbpt/d11/student/dev200/d11_armTR_L1.jsonl` | `b8db599f8c4b427ddcb2d772fd8f09aa` | sealed reference, MD5 pinned (freeze §1) |
| dev200 | T_direct_F | `outputs/efbpt/d10/teacher/t_direct_f.jsonl` | `40ebfe643e244c5a1a3fff729f23462c` | sealed reference, MD5 pinned (freeze §1) |
| dev200 | API_direct_ur | `outputs/efbpt/d11/api_direct/dev200/api_direct_ur.jsonl` | `31c82fa180a2677ce9df85db2ac3e0d7` | sealed reference, MD5 pinned (freeze §1) |
| dev200 | RO | `outputs/efbpt/d11/ablate_ro/dev200/d11_armRO.jsonl` | `d3a8850836389d89eac5369601de2f98` | sealed reference, MD5 pinned (freeze §1) |
| dev200 | T_tr_R1 | `outputs/efbpt/d13/large_alone/dev200/d13_T_tr_R1.jsonl` | `c11282e230887f23f5aeb4b41ee1eecc` | outputs/efbpt/d13/large_alone/dev200/SEAL_T_tr_R1.json (md5 e48980d52dd2bde8b941744d526ad26f; slurm 116803; 2026-11-13 12:58:30; 2 files verified) |
| dev200 | T_tr_L1 | `outputs/efbpt/d13/large_alone/dev200/d13_T_tr_L1.jsonl` | `62477705fa801a9d157788b021173892` | outputs/efbpt/d13/large_alone/dev200/SEAL_T_tr_L1.json (md5 652252e8c7e178ee4badd8ca69c1395c; slurm 116803; 2026-11-13 13:35:03; 2 files verified) |
| dev200 | API_tr_R1 | `outputs/efbpt/d13/large_alone/dev200/d13_API_tr_R1.jsonl` | `df0ea78cdf605a1e6788b8127b8c8490` | outputs/efbpt/d13/large_alone/dev200/SEAL_API_tr_R1.json (md5 2f6f048cf27f23faa14fca5ad63ef108; login node, no slurm job; 2026-11-14 14:31:16; 2 files verified) |
| dev200 | RAG_ur | `outputs/efbpt/d13/rag/dev200/d13_RAG_ur.jsonl` | `e6d62c7a2a0fc32a69e6970a0c08ac25` | outputs/efbpt/d13/rag/dev200/SEAL_RAG_ur.json (md5 10ecc7ac88ef120fd37c1e780f57f73a; slurm 117479; 2026-11-14 22:43:32; 2 files verified) |
| dev200 | RAG_en | `outputs/efbpt/d13/rag/dev200/d13_RAG_en.jsonl` | `e7107092541458e20393c2e0fa5284c2` | outputs/efbpt/d13/rag/dev200/SEAL_RAG_en.json (md5 0336d3ea6f549b9ffb5e93c28a275cf9; slurm 117479; 2026-11-15 01:08:49; 2 files verified) |

API_tr_R1 on eval458 (Amendment 3): 456 ok rows and 2 rows refused by the provider's input inspection (qids 90538ec7aa1100a34829, 78f2f99d04b9acd8bada). The refused rows have no answer: unparsed under every scorer, wrong on every line, not truncated.

## 2. Per arm

Balanced accuracy = mean of the gold-yes and gold-no accuracies. The sealed reference arms (A_r2_eval, TR, TR_L1_eval; on DEV200 A_r2, TR, TR_L1, T_direct_F, API_direct_ur, RO) are shown beside the D13 arms; their numbers are the committed D10 / D11 numbers.

### eval458 (n = 458)

| arm | scorer | acc % (correct) | unparsed (%) | predicted yes % | acc on gold-yes % | acc on gold-no % | balanced acc % |
|---|---|---|---|---|---|---|---|
| A_r2_eval | E0 (label) | 54.80 (251) | 15 (3.28) | 45.63 | 52.04 | 57.38 | 54.71 |
| A_r2_eval | E3 | 55.46 (254) | 19 (4.15) | 59.61 | 67.42 | 44.30 | 55.86 |
| TR | E0 (label) | 68.78 (315) | 6 (1.31) | 31.66 | 51.13 | 85.23 | 68.18 |
| TR | E3 | 74.24 (340) | 8 (1.75) | 44.32 | 70.14 | 78.06 | 74.10 |
| TR_L1_eval | E0 (label) | 64.63 (296) | 7 (1.53) | 33.84 | 49.77 | 78.48 | 64.13 |
| TR_L1_eval | E3 | 70.96 (325) | 10 (2.18) | 46.29 | 69.68 | 72.15 | 70.92 |
| T_direct_F_eval | E0 (label) | 63.32 (290) | 9 (1.97) | 67.25 | 82.81 | 45.15 | 63.98 |
| T_direct_F_eval | E3 | 63.10 (289) | 10 (2.18) | 67.03 | 82.35 | 45.15 | 63.75 |
| T_tr_R1 | E0fa (label) | 75.11 (344) | 18 (3.93) | 45.20 | 72.85 | 77.22 | 75.03 |
| T_tr_R1 | E0 (plain) | 75.11 (344) | 18 (3.93) | 45.20 | 72.85 | 77.22 | 75.03 |
| T_tr_R1 | E3 | 75.11 (344) | 18 (3.93) | 45.20 | 72.85 | 77.22 | 75.03 |
| T_tr_L1 | E0fa (label) | 70.96 (325) | 25 (5.46) | 45.63 | 69.68 | 72.15 | 70.92 |
| T_tr_L1 | E0 (plain) | 70.96 (325) | 25 (5.46) | 45.63 | 69.68 | 72.15 | 70.92 |
| T_tr_L1 | E3 | 70.96 (325) | 25 (5.46) | 45.63 | 69.68 | 72.15 | 70.92 |
| API_direct_ur_eval | E0 (label) | 67.47 (309) | 1 (0.22) | 25.11 | 42.53 | 90.72 | 66.63 |
| API_direct_ur_eval | E3 | 67.47 (309) | 1 (0.22) | 25.11 | 42.53 | 90.72 | 66.63 |
| API_tr_R1 | E0 (label) | 80.13 (367) | 2 (0.44) | 41.05 | 72.40 | 87.34 | 79.87 |
| API_tr_R1 | E3 | 80.13 (367) | 2 (0.44) | 41.05 | 72.40 | 87.34 | 79.87 |
| RAG_ur | E0 (label) | 53.93 (247) | 21 (4.59) | 39.52 | 45.25 | 62.03 | 53.64 |
| RAG_ur | E3 | 51.97 (238) | 23 (5.02) | 51.75 | 56.11 | 48.10 | 52.10 |
| RAG_en | E0 (label) | 53.93 (247) | 23 (5.02) | 38.86 | 43.44 | 63.71 | 53.58 |
| RAG_en | E3 | 54.37 (249) | 25 (5.46) | 53.93 | 59.73 | 49.37 | 54.55 |
| RO_eval | E0 (label) | 64.63 (296) | 6 (1.31) | 37.34 | 52.49 | 75.95 | 64.22 |
| RO_eval | E3 | 67.90 (311) | 6 (1.31) | 48.03 | 66.97 | 68.78 | 67.87 |
| Llama8B_A | E0 (label) | 30.79 (141) | 188 (41.05) | 20.52 | 19.91 | 40.93 | 30.42 |
| Llama8B_A | E3 | 24.02 (110) | 263 (57.42) | 2.40 | 2.26 | 44.30 | 23.28 |
| Llama8B_TR | E0 (label) | 49.34 (226) | 118 (25.76) | 14.41 | 20.36 | 76.37 | 48.37 |
| Llama8B_TR | E3 | 42.79 (196) | 154 (33.62) | 4.37 | 5.43 | 77.64 | 41.53 |
| Gemma9B_A | E0 (label) | 47.60 (218) | 81 (17.69) | 31.22 | 36.20 | 58.23 | 47.21 |
| Gemma9B_A | E3 | 45.41 (208) | 95 (20.74) | 33.84 | 38.46 | 51.90 | 45.18 |
| Gemma9B_TR | E0 (label) | 51.97 (238) | 108 (23.58) | 15.50 | 25.34 | 76.79 | 51.07 |
| Gemma9B_TR | E3 | 50.66 (232) | 125 (27.29) | 14.41 | 23.53 | 75.95 | 49.74 |
| Qwen8B_A | E0 (label) | 52.62 (241) | 40 (8.73) | 41.70 | 48.42 | 56.54 | 52.48 |
| Qwen8B_A | E3 | 51.53 (236) | 43 (9.39) | 44.76 | 50.68 | 52.32 | 51.50 |
| Qwen8B_TR | E0 (label) | 64.63 (296) | 20 (4.37) | 32.97 | 50.68 | 77.64 | 64.16 |
| Qwen8B_TR | E3 | 64.85 (297) | 23 (5.02) | 36.90 | 55.20 | 73.84 | 64.52 |

Rows where plain E0 and E3 differ: A_r2_eval 81; TR 69; TR_L1_eval 70; T_direct_F_eval 1; T_tr_R1 0; T_tr_L1 0; API_direct_ur_eval 0; API_tr_R1 0; RAG_ur 66; RAG_en 79; RO_eval 65; Llama8B_A 102; Llama8B_TR 65; Gemma9B_A 49; Gemma9B_TR 34; Qwen8B_A 42; Qwen8B_TR 34.

### DEV200 (n = 200, descriptive)

| arm | scorer | acc % (correct) | unparsed (%) | predicted yes % | acc on gold-yes % | acc on gold-no % | balanced acc % |
|---|---|---|---|---|---|---|---|
| A_r2 | E0 (label) | 60.00 (120) | 8 (4.00) | 41.00 | 54.02 | 64.60 | 59.31 |
| A_r2 | E3 | 55.00 (110) | 10 (5.00) | 54.50 | 64.37 | 47.79 | 56.08 |
| TR | E0 (label) | 68.50 (137) | 3 (1.50) | 33.50 | 52.87 | 80.53 | 66.70 |
| TR | E3 | 71.50 (143) | 4 (2.00) | 44.00 | 68.97 | 73.45 | 71.21 |
| TR_L1 | E0 (label) | 67.00 (134) | 2 (1.00) | 30.00 | 47.13 | 82.30 | 64.71 |
| TR_L1 | E3 | 70.00 (140) | 2 (1.00) | 44.00 | 66.67 | 72.57 | 69.62 |
| T_direct_F | E0 (label) | 55.50 (111) | 1 (0.50) | 74.50 | 85.06 | 32.74 | 58.90 |
| T_direct_F | E3 | 55.00 (110) | 2 (1.00) | 74.00 | 83.91 | 32.74 | 58.33 |
| API_direct_ur | E0 (label) | 69.00 (138) | 0 (0.00) | 27.50 | 45.98 | 86.73 | 66.35 |
| API_direct_ur | E3 | 69.00 (138) | 0 (0.00) | 27.50 | 45.98 | 86.73 | 66.35 |
| RO | E0 (label) | 57.50 (115) | 9 (4.50) | 35.50 | 44.83 | 67.26 | 56.04 |
| RO | E3 | 63.00 (126) | 10 (5.00) | 47.50 | 65.52 | 61.06 | 63.29 |
| T_tr_R1 | E0fa (label) | 75.50 (151) | 7 (3.50) | 46.50 | 75.86 | 75.22 | 75.54 |
| T_tr_R1 | E0 (plain) | 75.50 (151) | 7 (3.50) | 46.50 | 75.86 | 75.22 | 75.54 |
| T_tr_R1 | E3 | 75.50 (151) | 7 (3.50) | 46.50 | 75.86 | 75.22 | 75.54 |
| T_tr_L1 | E0fa (label) | 71.00 (142) | 9 (4.50) | 47.00 | 73.56 | 69.03 | 71.29 |
| T_tr_L1 | E0 (plain) | 71.00 (142) | 9 (4.50) | 47.00 | 73.56 | 69.03 | 71.29 |
| T_tr_L1 | E3 | 71.00 (142) | 9 (4.50) | 47.00 | 73.56 | 69.03 | 71.29 |
| API_tr_R1 | E0 (label) | 79.00 (158) | 0 (0.00) | 39.50 | 71.26 | 84.96 | 78.11 |
| API_tr_R1 | E3 | 79.00 (158) | 0 (0.00) | 39.50 | 71.26 | 84.96 | 78.11 |
| RAG_ur | E0 (label) | 58.50 (117) | 9 (4.50) | 37.50 | 48.28 | 66.37 | 57.32 |
| RAG_ur | E3 | 60.00 (120) | 11 (5.50) | 42.50 | 56.32 | 62.83 | 59.58 |
| RAG_en | E0 (label) | 52.00 (104) | 10 (5.00) | 39.50 | 44.83 | 57.52 | 51.17 |
| RAG_en | E3 | 54.50 (109) | 11 (5.50) | 52.50 | 63.22 | 47.79 | 55.50 |

Rows where plain E0 and E3 differ: A_r2 36; TR 27; TR_L1 36; T_direct_F 1; API_direct_ur 0; RO 32; T_tr_R1 0; T_tr_L1 0; API_tr_R1 0; RAG_ur 24; RAG_en 34.

## 3. E0fa for T_tr_R1 and T_tr_L1 (freeze §7.1; Amendment 2 A2.1)

E0fa replaces E0 for these two arms on L1, L3, L5, L6 and L7. L2 is E3 and L4 uses plain E0 and E3, as for every arm.

| set | arm | E0fa acc % (correct) | plain E0 acc % (correct) | E3 acc % (correct) | rows without the marker | marker present, no yes/no after it (unparsed) | rows where E0fa and E0 differ |
|---|---|---|---|---|---|---|---|
| eval458 | T_tr_R1 | 75.11 (344) | 75.11 (344) | 75.11 (344) | 29 | 0 | 0 |
| eval458 | T_tr_L1 | 70.96 (325) | 70.96 (325) | 70.96 (325) | 38 | 0 | 0 |
| dev200 | T_tr_R1 | 75.50 (151) | 75.50 (151) | 75.50 (151) | 9 | 0 | 0 |
| dev200 | T_tr_L1 | 71.00 (142) | 71.00 (142) | 71.00 (142) | 14 | 0 | 0 |

## 4. eval458 — family F1: TR vs the large model alone and vs traditional RAG (7 tests, Holm at 0.05)

**TR vs T_direct_F_eval — F1 test 1 of 7**

| line | n | acc TR | acc T_direct_F_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 68.78 | 63.32 | +5.46 | 120 | 95 | 0.1015 |
| L2  E3, all rows | 458 | 74.24 | 63.10 | +11.14 | 117 | 66 | 0.0002008 |
| L3  label, rows parsed in both arms | 443 | 69.30 | 64.56 | +4.74 | 112 | 91 | 0.1602 |
| L4  rows where E0 = E3 in both arms (descriptive) | 388 | 75.52 | 61.60 | +13.92 | 109 | 55 | 2.995e-05 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 69.36 | 63.73 | +5.64 | 106 | 83 | 0.1093 |
| L6  label, rows truncated in neither arm | 433 | 68.82 | 63.74 | +5.08 | 112 | 90 | 0.1393 |
| L7  label, parsed in both and truncated in neither | 424 | 69.34 | 64.39 | +4.95 | 108 | 87 | 0.1519 |

Unparsed (label): TR 6 (1.31 %), T_direct_F_eval 9 (1.97 %), difference 0.66 pp ≤ 5.0. Truncated: TR 20 (4.37 %), T_direct_F_eval 5 (1.09 %), difference 3.28 pp ≤ 5.0. Deciding line (Amendment 2 A2.2): **L1**.

**TR vs T_tr_R1 — F1 test 2 of 7**

| line | n | acc TR | acc T_tr_R1 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 68.78 | 75.11 | -6.33 | 55 | 84 | 0.01724 |
| L2  E3, all rows | 458 | 74.24 | 75.11 | -0.87 | 54 | 58 | 0.777 |
| L3  label, rows parsed in both arms | 434 | 70.05 | 78.34 | -8.29 | 44 | 80 | 0.001565 |
| L4  rows where E0 = E3 in both arms (descriptive) | 389 | 75.58 | 76.86 | -1.29 | 43 | 48 | 0.6752 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 69.36 | 74.75 | -5.39 | 52 | 74 | 0.06094 |
| L6  label, rows truncated in neither arm | 433 | 69.28 | 75.75 | -6.47 | 51 | 79 | 0.01754 |
| L7  label, parsed in both and truncated in neither | 415 | 70.12 | 78.31 | -8.19 | 42 | 76 | 0.002243 |

Unparsed (label): TR 6 (1.31 %), T_tr_R1 18 (3.93 %), difference 2.62 pp ≤ 5.0. Truncated: TR 20 (4.37 %), T_tr_R1 7 (1.53 %), difference 2.84 pp ≤ 5.0. Deciding line (Amendment 2 A2.2): **L1**.

**TR vs T_tr_L1 — F1 test 3 of 7**

| line | n | acc TR | acc T_tr_L1 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 68.78 | 70.96 | -2.18 | 70 | 80 | 0.4625 |
| L2  E3, all rows | 458 | 74.24 | 70.96 | +3.28 | 74 | 59 | 0.2246 |
| L3  label, rows parsed in both arms | 427 | 70.73 | 75.41 | -4.68 | 57 | 77 | 0.1004 |
| L4  rows where E0 = E3 in both arms (descriptive) | 389 | 75.58 | 72.24 | +3.34 | 60 | 47 | 0.2459 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 69.36 | 71.08 | -1.72 | 63 | 70 | 0.603 |
| L6  label, rows truncated in neither arm | 434 | 69.35 | 71.20 | -1.84 | 67 | 75 | 0.5571 |
| L7  label, parsed in both and truncated in neither | 410 | 70.73 | 74.63 | -3.90 | 56 | 72 | 0.1847 |

Unparsed (label): TR 6 (1.31 %), T_tr_L1 25 (5.46 %), difference 4.15 pp ≤ 5.0. Truncated: TR 20 (4.37 %), T_tr_L1 6 (1.31 %), difference 3.06 pp ≤ 5.0. Deciding line (Amendment 2 A2.2): **L1**.

**TR vs API_direct_ur_eval — F1 test 4 of 7**

| line | n | acc TR | acc API_direct_ur_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 68.78 | 67.47 | +1.31 | 59 | 53 | 0.6368 |
| L2  E3, all rows | 458 | 74.24 | 67.47 | +6.77 | 73 | 42 | 0.004928 |
| L3  label, rows parsed in both arms | 451 | 69.62 | 67.63 | +2.00 | 58 | 49 | 0.4394 |
| L4  rows where E0 = E3 in both arms (descriptive) | 389 | 75.58 | 69.92 | +5.66 | 50 | 28 | 0.01685 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 69.36 | 67.40 | +1.96 | 53 | 45 | 0.4797 |
| L6  label, rows truncated in neither arm | 438 | 68.95 | 67.12 | +1.83 | 57 | 49 | 0.4968 |
| L7  label, parsed in both and truncated in neither | 432 | 69.68 | 67.36 | +2.31 | 56 | 46 | 0.3729 |

Unparsed (label): TR 6 (1.31 %), API_direct_ur_eval 1 (0.22 %), difference 1.09 pp ≤ 5.0. Truncated: TR 20 (4.37 %), API_direct_ur_eval 0 (0.00 %), difference 4.37 pp ≤ 5.0. Deciding line (Amendment 2 A2.2): **L1**.

**TR vs API_tr_R1 — F1 test 5 of 7**

| line | n | acc TR | acc API_tr_R1 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 68.78 | 80.13 | -11.35 | 31 | 83 | 1.18e-06 |
| L2  E3, all rows | 458 | 74.24 | 80.13 | -5.90 | 35 | 62 | 0.007959 |
| L3  label, rows parsed in both arms | 450 | 69.56 | 80.22 | -10.67 | 29 | 77 | 3.453e-06 |
| L4  rows where E0 = E3 in both arms (descriptive) | 389 | 75.58 | 82.78 | -7.20 | 21 | 49 | 0.001093 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 69.36 | 80.15 | -10.78 | 30 | 74 | 1.894e-05 |
| L6  label, rows truncated in neither arm | 438 | 68.95 | 79.68 | -10.73 | 31 | 78 | 7.734e-06 |
| L7  label, parsed in both and truncated in neither | 431 | 69.61 | 79.81 | -10.21 | 29 | 73 | 1.55e-05 |
| A3.3  label, the 456 rows the provider answered | 456 | 68.64 | 80.48 | -11.84 | 29 | 83 | 3.291e-07 |

Unparsed (label): TR 6 (1.31 %), API_tr_R1 2 (0.44 %), difference 0.87 pp ≤ 5.0. Truncated: TR 20 (4.37 %), API_tr_R1 0 (0.00 %), difference 4.37 pp ≤ 5.0. Deciding line (Amendment 2 A2.2): **L1**.

**TR vs RAG_ur — F1 test 6 of 7**

| line | n | acc TR | acc RAG_ur | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 68.78 | 53.93 | +14.85 | 118 | 50 | 1.612e-07 |
| L2  E3, all rows | 458 | 74.24 | 51.97 | +22.27 | 142 | 40 | 1.357e-14 |
| L3  label, rows parsed in both arms | 432 | 69.68 | 56.25 | +13.43 | 104 | 46 | 2.472e-06 |
| L4  rows where E0 = E3 in both arms (descriptive) | 341 | 75.37 | 54.25 | +21.11 | 99 | 27 | 7.472e-11 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 69.36 | 53.92 | +15.44 | 108 | 45 | 3.724e-07 |
| L6  label, rows truncated in neither arm | 391 | 69.31 | 53.20 | +16.11 | 101 | 38 | 8.818e-08 |
| L7  label, parsed in both and truncated in neither | 373 | 70.24 | 54.69 | +15.55 | 92 | 34 | 2.365e-07 |

Unparsed (label): TR 6 (1.31 %), RAG_ur 21 (4.59 %), difference 3.28 pp ≤ 5.0. Truncated: TR 20 (4.37 %), RAG_ur 50 (10.92 %), difference 6.55 pp > 5.0. Deciding line (Amendment 2 A2.2): **L6**; L1 is reported beside it.

**TR vs RAG_en — F1 test 7 of 7**

| line | n | acc TR | acc RAG_en | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 68.78 | 53.93 | +14.85 | 107 | 39 | 1.631e-08 |
| L2  E3, all rows | 458 | 74.24 | 54.37 | +19.87 | 131 | 40 | 1.711e-12 |
| L3  label, rows parsed in both arms | 430 | 70.47 | 56.51 | +13.95 | 95 | 35 | 1.385e-07 |
| L4  rows where E0 = E3 in both arms (descriptive) | 327 | 76.45 | 55.96 | +20.49 | 85 | 18 | 1.363e-11 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 69.36 | 55.39 | +13.97 | 94 | 37 | 6.718e-07 |
| L6  label, rows truncated in neither arm | 383 | 69.45 | 56.66 | +12.79 | 85 | 36 | 9.815e-06 |
| L7  label, parsed in both and truncated in neither | 366 | 71.04 | 58.20 | +12.84 | 79 | 32 | 9.451e-06 |

Unparsed (label): TR 6 (1.31 %), RAG_en 23 (5.02 %), difference 3.71 pp ≤ 5.0. Truncated: TR 20 (4.37 %), RAG_en 61 (13.32 %), difference 8.95 pp > 5.0. Deciding line (Amendment 2 A2.2): **L6**; L1 is reported beside it.

### F1 summary (freeze §3 Holm, §4 wording)

| test | deciding line | n | Δ pp | b | c | raw p | Holm p | L1 beside it: Δ pp, b / c, raw p | reading |
|---|---|---|---|---|---|---|---|---|---|
| TR vs T_direct_F_eval | L1 | 458 | +5.46 | 120 | 95 | 0.1015 | 0.3044 | +5.46, 120 / 95, 0.1015 | **not distinguishable** |
| TR vs T_tr_R1 | L1 | 458 | -6.33 | 55 | 84 | 0.01724 | 0.06895 | -6.33, 55 / 84, 0.01724 | **not distinguishable** |
| TR vs T_tr_L1 | L1 | 458 | -2.18 | 70 | 80 | 0.4625 | 0.9251 | -2.18, 70 / 80, 0.4625 | **not distinguishable** |
| TR vs API_direct_ur_eval | L1 | 458 | +1.31 | 59 | 53 | 0.6368 | 0.9251 | +1.31, 59 / 53, 0.6368 | **not distinguishable** |
| TR vs API_tr_R1 | L1 | 458 | -11.35 | 31 | 83 | 1.18e-06 | 7.082e-06 | -11.35, 31 / 83, 1.18e-06 | **API_tr_R1 beats TR** |
| TR vs RAG_ur | L6 | 391 | +16.11 | 101 | 38 | 8.818e-08 | 6.172e-07 | +14.85, 118 / 50, 1.612e-07 | **TR beats RAG_ur** |
| TR vs RAG_en | L6 | 383 | +12.79 | 85 | 36 | 9.815e-06 | 4.907e-05 | +14.85, 107 / 39, 1.631e-08 | **TR beats RAG_en** |

Wording allowed by freeze §4: "TR beats X" only if Δ > 0 and Holm p < 0.05; "X beats TR" only if Δ < 0 and Holm p < 0.05; everything else "not distinguishable".

Amendment 3 A3.3, TR vs API_tr_R1 on the 456 rows the provider answered: acc 68.64 vs 80.48, Δ -11.84 pp, b 29 / c 83, raw p = 3.291e-07 (reported beside the test; the test counts the 2 refused rows as wrong).

## 5. eval458 — family F2: transfer to other students, S_TR vs S_A (3 tests, Holm at 0.05)

**Llama8B_TR vs Llama8B_A — F2 test 1 of 3**

| line | n | acc Llama8B_TR | acc Llama8B_A | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 49.34 | 30.79 | +18.56 | 136 | 51 | 4.067e-10 |
| L2  E3, all rows | 458 | 42.79 | 24.02 | +18.78 | 109 | 23 | 1.387e-14 |
| L3  label, rows parsed in both arms | 216 | 66.67 | 51.85 | +14.81 | 54 | 22 | 0.000313 |
| L4  rows where E0 = E3 in both arms (descriptive) | 310 | 47.10 | 29.03 | +18.06 | 71 | 15 | 7.076e-10 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 50.74 | 30.39 | +20.34 | 127 | 44 | 1.565e-10 |
| L6  label, rows truncated in neither arm | 372 | 50.54 | 30.65 | +19.89 | 109 | 35 | 4.856e-10 |
| L7  label, parsed in both and truncated in neither | 174 | 66.67 | 54.02 | +12.64 | 37 | 15 | 0.003185 |

Unparsed (label): Llama8B_TR 118 (25.76 %), Llama8B_A 188 (41.05 %), difference 15.28 pp > 5.0. Truncated: Llama8B_TR 24 (5.24 %), Llama8B_A 70 (15.28 %), difference 10.04 pp > 5.0. Deciding line (Amendment 2 A2.2): **L7**; L1 is reported beside it.

**Gemma9B_TR vs Gemma9B_A — F2 test 2 of 3**

| line | n | acc Gemma9B_TR | acc Gemma9B_A | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 51.97 | 47.60 | +4.37 | 100 | 80 | 0.1565 |
| L2  E3, all rows | 458 | 50.66 | 45.41 | +5.24 | 110 | 86 | 0.1002 |
| L3  label, rows parsed in both arms | 291 | 69.07 | 58.76 | +10.31 | 63 | 33 | 0.002879 |
| L4  rows where E0 = E3 in both arms (descriptive) | 382 | 51.31 | 47.91 | +3.40 | 83 | 70 | 0.332 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 50.74 | 46.81 | +3.92 | 89 | 73 | 0.2385 |
| L6  label, rows truncated in neither arm | 451 | 51.88 | 48.12 | +3.77 | 97 | 80 | 0.229 |
| L7  label, parsed in both and truncated in neither | 289 | 68.86 | 58.82 | +10.03 | 62 | 33 | 0.003833 |

Unparsed (label): Gemma9B_TR 108 (23.58 %), Gemma9B_A 81 (17.69 %), difference 5.90 pp > 5.0. Truncated: Gemma9B_TR 0 (0.00 %), Gemma9B_A 7 (1.53 %), difference 1.53 pp ≤ 5.0. Deciding line (Amendment 2 A2.2): **L3**; L1 is reported beside it.

**Qwen8B_TR vs Qwen8B_A — F2 test 3 of 3**

| line | n | acc Qwen8B_TR | acc Qwen8B_A | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 64.63 | 52.62 | +12.01 | 121 | 66 | 7.027e-05 |
| L2  E3, all rows | 458 | 64.85 | 51.53 | +13.32 | 121 | 60 | 6.79e-06 |
| L3  label, rows parsed in both arms | 403 | 67.25 | 57.07 | +10.17 | 96 | 55 | 0.001065 |
| L4  rows where E0 = E3 in both arms (descriptive) | 389 | 66.32 | 52.96 | +13.37 | 102 | 50 | 2.983e-05 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 64.71 | 53.68 | +11.03 | 104 | 59 | 0.0005282 |
| L6  label, rows truncated in neither arm | 359 | 67.69 | 55.99 | +11.70 | 90 | 48 | 0.0004406 |
| L7  label, parsed in both and truncated in neither | 330 | 69.39 | 58.79 | +10.61 | 76 | 41 | 0.001557 |

Unparsed (label): Qwen8B_TR 20 (4.37 %), Qwen8B_A 40 (8.73 %), difference 4.37 pp ≤ 5.0. Truncated: Qwen8B_TR 29 (6.33 %), Qwen8B_A 79 (17.25 %), difference 10.92 pp > 5.0. Deciding line (Amendment 2 A2.2): **L6**; L1 is reported beside it.

### F2 summary

| test | deciding line | n | Δ pp | b | c | raw p | Holm p | L1 beside it: Δ pp, b / c, raw p | reading |
|---|---|---|---|---|---|---|---|---|---|
| Llama8B_TR vs Llama8B_A | L7 | 174 | +12.64 | 37 | 15 | 0.003185 | 0.005757 | +18.56, 136 / 51, 4.067e-10 | **Llama8B_TR beats Llama8B_A**; Δ > 0 and Holm p < 0.05: yes |
| Gemma9B_TR vs Gemma9B_A | L3 | 291 | +10.31 | 63 | 33 | 0.002879 | 0.005757 | +4.37, 100 / 80, 0.1565 | **Gemma9B_TR beats Gemma9B_A**; Δ > 0 and Holm p < 0.05: yes |
| Qwen8B_TR vs Qwen8B_A | L6 | 359 | +11.70 | 90 | 48 | 0.0004406 | 0.001322 | +12.01, 121 / 66, 7.027e-05 | **Qwen8B_TR beats Qwen8B_A**; Δ > 0 and Holm p < 0.05: yes |

Students with Δ > 0 and Holm p < 0.05: 3 of 3. Freeze §4 ("holds" if at least 2 of 3, "partial" if 1, "not shown" if 0): transfer **holds**.

## 6. eval458 — descriptive comparisons (freeze §3: no claim, no Holm, no wording)

**TR_L1_eval vs T_direct_F_eval — descriptive**

| line | n | acc TR_L1_eval | acc T_direct_F_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 64.63 | 63.32 | +1.31 | 109 | 103 | 0.7314 |
| L2  E3, all rows | 458 | 70.96 | 63.10 | +7.86 | 105 | 69 | 0.007788 |
| L3  label, rows parsed in both arms | 442 | 65.38 | 64.71 | +0.68 | 102 | 99 | 0.8879 |
| L4  rows where E0 = E3 in both arms (descriptive) | 388 | 71.39 | 61.34 | +10.05 | 98 | 59 | 0.002317 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 65.44 | 63.73 | +1.72 | 97 | 90 | 0.6609 |
| L6  label, rows truncated in neither arm | 434 | 65.67 | 63.36 | +2.30 | 104 | 94 | 0.5225 |
| L7  label, parsed in both and truncated in neither | 424 | 66.27 | 64.15 | +2.12 | 100 | 91 | 0.5628 |

Unparsed (label): TR_L1_eval 7 (1.53 %), T_direct_F_eval 9 (1.97 %), difference 0.44 pp ≤ 5.0. Truncated: TR_L1_eval 19 (4.15 %), T_direct_F_eval 5 (1.09 %), difference 3.06 pp ≤ 5.0. Descriptive, no claim; the A2.2 rule would read it on L1.

**TR_L1_eval vs T_tr_L1 — descriptive**

| line | n | acc TR_L1_eval | acc T_tr_L1 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 64.63 | 70.96 | -6.33 | 54 | 83 | 0.01643 |
| L2  E3, all rows | 458 | 70.96 | 70.96 | +0.00 | 59 | 59 | 1 |
| L3  label, rows parsed in both arms | 426 | 66.67 | 75.59 | -8.92 | 42 | 80 | 0.0007408 |
| L4  rows where E0 = E3 in both arms (descriptive) | 388 | 71.39 | 72.42 | -1.03 | 45 | 49 | 0.7572 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 65.44 | 71.08 | -5.64 | 48 | 71 | 0.04327 |
| L6  label, rows truncated in neither arm | 434 | 65.90 | 72.58 | -6.68 | 49 | 78 | 0.01266 |
| L7  label, parsed in both and truncated in neither | 409 | 67.48 | 76.28 | -8.80 | 39 | 75 | 0.0009592 |

Unparsed (label): TR_L1_eval 7 (1.53 %), T_tr_L1 25 (5.46 %), difference 3.93 pp ≤ 5.0. Truncated: TR_L1_eval 19 (4.15 %), T_tr_L1 6 (1.31 %), difference 2.84 pp ≤ 5.0. Descriptive, no claim; the A2.2 rule would read it on L1.

**TR vs RO_eval — descriptive**

| line | n | acc TR | acc RO_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 68.78 | 64.63 | +4.15 | 75 | 56 | 0.1155 |
| L2  E3, all rows | 458 | 74.24 | 67.90 | +6.33 | 77 | 48 | 0.01196 |
| L3  label, rows parsed in both arms | 446 | 70.18 | 65.25 | +4.93 | 73 | 51 | 0.05888 |
| L4  rows where E0 = E3 in both arms (descriptive) | 341 | 75.07 | 70.09 | +4.99 | 46 | 29 | 0.06395 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 69.36 | 65.44 | +3.92 | 67 | 51 | 0.1671 |
| L6  label, rows truncated in neither arm | 421 | 68.88 | 63.18 | +5.70 | 70 | 46 | 0.03227 |
| L7  label, parsed in both and truncated in neither | 410 | 70.24 | 63.90 | +6.34 | 68 | 42 | 0.01675 |

Unparsed (label): TR 6 (1.31 %), RO_eval 6 (1.31 %), difference 0.00 pp ≤ 5.0. Truncated: TR 20 (4.37 %), RO_eval 21 (4.59 %), difference 0.22 pp ≤ 5.0. Descriptive, no claim; the A2.2 rule would read it on L1.

**RO_eval vs A_r2_eval — descriptive**

| line | n | acc RO_eval | acc A_r2_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 64.63 | 54.80 | +9.83 | 106 | 61 | 0.0006187 |
| L2  E3, all rows | 458 | 67.90 | 55.46 | +12.45 | 105 | 48 | 4.697e-06 |
| L3  label, rows parsed in both arms | 438 | 65.53 | 56.85 | +8.68 | 97 | 59 | 0.002931 |
| L4  rows where E0 = E3 in both arms (descriptive) | 328 | 69.51 | 55.49 | +14.02 | 74 | 28 | 5.908e-06 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 65.44 | 56.13 | +9.31 | 93 | 55 | 0.002244 |
| L6  label, rows truncated in neither arm | 415 | 64.58 | 55.42 | +9.16 | 93 | 55 | 0.002244 |
| L7  label, parsed in both and truncated in neither | 397 | 65.74 | 57.43 | +8.31 | 86 | 53 | 0.006435 |

Unparsed (label): RO_eval 6 (1.31 %), A_r2_eval 15 (3.28 %), difference 1.97 pp ≤ 5.0. Truncated: RO_eval 21 (4.59 %), A_r2_eval 26 (5.68 %), difference 1.09 pp ≤ 5.0. Descriptive, no claim; the A2.2 rule would read it on L1.

**RAG_ur vs A_r2_eval — descriptive**

| line | n | acc RAG_ur | acc A_r2_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 53.93 | 54.80 | -0.87 | 79 | 83 | 0.8138 |
| L2  E3, all rows | 458 | 51.97 | 55.46 | -3.49 | 62 | 78 | 0.2047 |
| L3  label, rows parsed in both arms | 425 | 57.18 | 56.47 | +0.71 | 75 | 72 | 0.869 |
| L4  rows where E0 = E3 in both arms (descriptive) | 334 | 52.69 | 56.59 | -3.89 | 43 | 56 | 0.2276 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 53.92 | 56.13 | -2.21 | 70 | 79 | 0.5124 |
| L6  label, rows truncated in neither arm | 387 | 52.71 | 54.52 | -1.81 | 65 | 72 | 0.6084 |
| L7  label, parsed in both and truncated in neither | 364 | 55.22 | 56.32 | -1.10 | 62 | 66 | 0.791 |

Unparsed (label): RAG_ur 21 (4.59 %), A_r2_eval 15 (3.28 %), difference 1.31 pp ≤ 5.0. Truncated: RAG_ur 50 (10.92 %), A_r2_eval 26 (5.68 %), difference 5.24 pp > 5.0. Descriptive, no claim; the A2.2 rule would read it on L6.

**RAG_en vs A_r2_eval — descriptive**

| line | n | acc RAG_en | acc A_r2_eval | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 458 | 53.93 | 54.80 | -0.87 | 86 | 90 | 0.8212 |
| L2  E3, all rows | 458 | 54.37 | 55.46 | -1.09 | 73 | 78 | 0.7449 |
| L3  label, rows parsed in both arms | 422 | 56.40 | 57.11 | -0.71 | 77 | 80 | 0.8732 |
| L4  rows where E0 = E3 in both arms (descriptive) | 320 | 55.94 | 57.50 | -1.56 | 50 | 55 | 0.6965 |
| L5  label, the 408 rows (first 50 probe rows dropped) | 408 | 55.39 | 56.13 | -0.74 | 75 | 78 | 0.8716 |
| L6  label, rows truncated in neither arm | 377 | 55.70 | 54.38 | +1.33 | 76 | 71 | 0.7416 |
| L7  label, parsed in both and truncated in neither | 353 | 57.22 | 56.66 | +0.57 | 68 | 66 | 0.9312 |

Unparsed (label): RAG_en 23 (5.02 %), A_r2_eval 15 (3.28 %), difference 1.75 pp ≤ 5.0. Truncated: RAG_en 61 (13.32 %), A_r2_eval 26 (5.68 %), difference 7.64 pp > 5.0. Descriptive, no claim; the A2.2 rule would read it on L6.

## 7. Predictions (freeze §5), read on L1 of eval458

| prediction | observed | |
|---|---|---|
| T_direct_F_eval accuracy in [48, 60] | 63.32 | miss |
| API_direct_ur_eval accuracy in [62, 72] | 67.47 | hit |
| T_tr_R1 accuracy in [60, 76] | 75.11 | hit |
| T_tr_L1 within 4 pp of T_tr_R1 | T_tr_L1 − T_tr_R1 = -4.15 | miss |
| RAG_ur − A_r2_eval in [-6, +4] | -0.87 | hit |
| RAG_en − A_r2_eval in [-6, +4] | -0.87 | hit |
| RO_eval − A_r2_eval in [-3, +5] | +9.83 | miss |
| Llama8B_TR − Llama8B_A > 0 | +18.56 | hit |
| Gemma9B_TR − Gemma9B_A > 0 | +4.37 | hit |
| Qwen8B_TR − Qwen8B_A > 0 | +12.01 | hit |
| expected reading: TR and API_direct_ur_eval not distinguishable | not distinguishable | hit |

T_tr_R1 and T_tr_L1 are read with E0fa (their L1 label). The expected reading is the F1 wording of that test.

## 8. DEV200 — descriptive (freeze §3: lines L1-L4, raw p only, no wording; L6 and L7 added by Amendments 1 and 2; L5 is defined on eval458 only)

**TR vs T_direct_F — DEV200, descriptive**

| line | n | acc TR | acc T_direct_F | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 68.50 | 55.50 | +13.00 | 64 | 38 | 0.01292 |
| L2  E3, all rows | 200 | 71.50 | 55.00 | +16.50 | 59 | 26 | 0.0004472 |
| L3  label, rows parsed in both arms | 196 | 69.39 | 55.10 | +14.29 | 63 | 35 | 0.006093 |
| L4  rows where E0 = E3 in both arms (descriptive) | 172 | 73.84 | 56.40 | +17.44 | 55 | 25 | 0.001053 |
| L6  label, rows truncated in neither arm | 190 | 68.95 | 54.74 | +14.21 | 62 | 35 | 0.007959 |
| L7  label, parsed in both and truncated in neither | 188 | 69.68 | 54.26 | +15.43 | 62 | 33 | 0.003833 |

Unparsed (label): TR 3 (1.50 %), T_direct_F 1 (0.50 %), difference 1.00 pp ≤ 5.0. Truncated: TR 9 (4.50 %), T_direct_F 2 (1.00 %), difference 3.50 pp ≤ 5.0. Descriptive, raw p only.

**TR vs T_tr_R1 — DEV200, descriptive**

| line | n | acc TR | acc T_tr_R1 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 68.50 | 75.50 | -7.00 | 24 | 38 | 0.09795 |
| L2  E3, all rows | 200 | 71.50 | 75.50 | -4.00 | 22 | 30 | 0.3317 |
| L3  label, rows parsed in both arms | 190 | 69.47 | 77.89 | -8.42 | 19 | 35 | 0.04022 |
| L4  rows where E0 = E3 in both arms (descriptive) | 173 | 73.41 | 75.72 | -2.31 | 20 | 24 | 0.6516 |
| L6  label, rows truncated in neither arm | 187 | 70.05 | 77.01 | -6.95 | 22 | 35 | 0.1112 |
| L7  label, parsed in both and truncated in neither | 181 | 70.17 | 78.45 | -8.29 | 18 | 33 | 0.04887 |

Unparsed (label): TR 3 (1.50 %), T_tr_R1 7 (3.50 %), difference 2.00 pp ≤ 5.0. Truncated: TR 9 (4.50 %), T_tr_R1 4 (2.00 %), difference 2.50 pp ≤ 5.0. Descriptive, raw p only.

**TR vs T_tr_L1 — DEV200, descriptive**

| line | n | acc TR | acc T_tr_L1 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 68.50 | 71.00 | -2.50 | 35 | 40 | 0.6445 |
| L2  E3, all rows | 200 | 71.50 | 71.00 | +0.50 | 32 | 31 | 1 |
| L3  label, rows parsed in both arms | 188 | 69.15 | 73.94 | -4.79 | 28 | 37 | 0.3211 |
| L4  rows where E0 = E3 in both arms (descriptive) | 173 | 73.41 | 71.10 | +2.31 | 30 | 26 | 0.6889 |
| L6  label, rows truncated in neither arm | 186 | 69.89 | 73.66 | -3.76 | 30 | 37 | 0.4638 |
| L7  label, parsed in both and truncated in neither | 179 | 69.83 | 75.42 | -5.59 | 25 | 35 | 0.2451 |

Unparsed (label): TR 3 (1.50 %), T_tr_L1 9 (4.50 %), difference 3.00 pp ≤ 5.0. Truncated: TR 9 (4.50 %), T_tr_L1 5 (2.50 %), difference 2.00 pp ≤ 5.0. Descriptive, raw p only.

**TR vs API_direct_ur — DEV200, descriptive**

| line | n | acc TR | acc API_direct_ur | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 68.50 | 69.00 | -0.50 | 26 | 27 | 1 |
| L2  E3, all rows | 200 | 71.50 | 69.00 | +2.50 | 31 | 26 | 0.5966 |
| L3  label, rows parsed in both arms | 197 | 69.54 | 69.04 | +0.51 | 26 | 25 | 1 |
| L4  rows where E0 = E3 in both arms (descriptive) | 173 | 73.41 | 69.36 | +4.05 | 24 | 17 | 0.3489 |
| L6  label, rows truncated in neither arm | 191 | 69.11 | 69.63 | -0.52 | 24 | 25 | 1 |
| L7  label, parsed in both and truncated in neither | 189 | 69.84 | 69.31 | +0.53 | 24 | 23 | 1 |

Unparsed (label): TR 3 (1.50 %), API_direct_ur 0 (0.00 %), difference 1.50 pp ≤ 5.0. Truncated: TR 9 (4.50 %), API_direct_ur 0 (0.00 %), difference 4.50 pp ≤ 5.0. Descriptive, raw p only.

**TR vs API_tr_R1 — DEV200, descriptive**

| line | n | acc TR | acc API_tr_R1 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 68.50 | 79.00 | -10.50 | 19 | 40 | 0.008641 |
| L2  E3, all rows | 200 | 71.50 | 79.00 | -7.50 | 17 | 32 | 0.04438 |
| L3  label, rows parsed in both arms | 197 | 69.54 | 79.19 | -9.64 | 19 | 38 | 0.01635 |
| L4  rows where E0 = E3 in both arms (descriptive) | 173 | 73.41 | 78.03 | -4.62 | 16 | 24 | 0.2682 |
| L6  label, rows truncated in neither arm | 191 | 69.11 | 79.58 | -10.47 | 19 | 39 | 0.01193 |
| L7  label, parsed in both and truncated in neither | 189 | 69.84 | 79.37 | -9.52 | 19 | 37 | 0.02224 |

Unparsed (label): TR 3 (1.50 %), API_tr_R1 0 (0.00 %), difference 1.50 pp ≤ 5.0. Truncated: TR 9 (4.50 %), API_tr_R1 0 (0.00 %), difference 4.50 pp ≤ 5.0. Descriptive, raw p only.

**TR vs RAG_ur — DEV200, descriptive**

| line | n | acc TR | acc RAG_ur | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 68.50 | 58.50 | +10.00 | 43 | 23 | 0.01866 |
| L2  E3, all rows | 200 | 71.50 | 60.00 | +11.50 | 46 | 23 | 0.007621 |
| L3  label, rows parsed in both arms | 189 | 69.84 | 61.38 | +8.47 | 38 | 22 | 0.05189 |
| L4  rows where E0 = E3 in both arms (descriptive) | 152 | 74.34 | 65.13 | +9.21 | 30 | 16 | 0.05408 |
| L6  label, rows truncated in neither arm | 163 | 68.71 | 60.12 | +8.59 | 32 | 18 | 0.06491 |
| L7  label, parsed in both and truncated in neither | 157 | 69.43 | 62.42 | +7.01 | 29 | 18 | 0.1439 |

Unparsed (label): TR 3 (1.50 %), RAG_ur 9 (4.50 %), difference 3.00 pp ≤ 5.0. Truncated: TR 9 (4.50 %), RAG_ur 30 (15.00 %), difference 10.50 pp > 5.0. Descriptive, raw p only.

**TR vs RAG_en — DEV200, descriptive**

| line | n | acc TR | acc RAG_en | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 68.50 | 52.00 | +16.50 | 49 | 16 | 5.085e-05 |
| L2  E3, all rows | 200 | 71.50 | 54.50 | +17.00 | 54 | 20 | 9.613e-05 |
| L3  label, rows parsed in both arms | 188 | 69.68 | 55.32 | +14.36 | 43 | 16 | 0.0005844 |
| L4  rows where E0 = E3 in both arms (descriptive) | 147 | 73.47 | 59.18 | +14.29 | 34 | 13 | 0.003088 |
| L6  label, rows truncated in neither arm | 162 | 67.90 | 54.32 | +13.58 | 36 | 14 | 0.002602 |
| L7  label, parsed in both and truncated in neither | 155 | 69.03 | 56.77 | +12.26 | 33 | 14 | 0.007943 |

Unparsed (label): TR 3 (1.50 %), RAG_en 10 (5.00 %), difference 3.50 pp ≤ 5.0. Truncated: TR 9 (4.50 %), RAG_en 32 (16.00 %), difference 11.50 pp > 5.0. Descriptive, raw p only.

**RAG_ur vs A_r2 — DEV200, descriptive**

| line | n | acc RAG_ur | acc A_r2 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 58.50 | 60.00 | -1.50 | 31 | 34 | 0.8043 |
| L2  E3, all rows | 200 | 60.00 | 55.00 | +5.00 | 36 | 26 | 0.2529 |
| L3  label, rows parsed in both arms | 185 | 60.54 | 64.32 | -3.78 | 26 | 33 | 0.435 |
| L4  rows where E0 = E3 in both arms (descriptive) | 149 | 65.10 | 60.40 | +4.70 | 24 | 17 | 0.3489 |
| L6  label, rows truncated in neither arm | 164 | 59.76 | 60.37 | -0.61 | 24 | 25 | 1 |
| L7  label, parsed in both and truncated in neither | 156 | 60.90 | 62.82 | -1.92 | 21 | 24 | 0.766 |

Unparsed (label): RAG_ur 9 (4.50 %), A_r2 8 (4.00 %), difference 0.50 pp ≤ 5.0. Truncated: RAG_ur 30 (15.00 %), A_r2 11 (5.50 %), difference 9.50 pp > 5.0. Descriptive, raw p only.

**RAG_en vs A_r2 — DEV200, descriptive**

| line | n | acc RAG_en | acc A_r2 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 52.00 | 60.00 | -8.00 | 30 | 46 | 0.08465 |
| L2  E3, all rows | 200 | 54.50 | 55.00 | -0.50 | 28 | 29 | 1 |
| L3  label, rows parsed in both arms | 182 | 53.85 | 63.19 | -9.34 | 24 | 41 | 0.04635 |
| L4  rows where E0 = E3 in both arms (descriptive) | 138 | 57.97 | 60.87 | -2.90 | 18 | 22 | 0.6358 |
| L6  label, rows truncated in neither arm | 159 | 54.09 | 59.12 | -5.03 | 24 | 32 | 0.3497 |
| L7  label, parsed in both and truncated in neither | 148 | 56.08 | 60.81 | -4.73 | 21 | 28 | 0.3916 |

Unparsed (label): RAG_en 10 (5.00 %), A_r2 8 (4.00 %), difference 1.00 pp ≤ 5.0. Truncated: RAG_en 32 (16.00 %), A_r2 11 (5.50 %), difference 10.50 pp > 5.0. Descriptive, raw p only.

**TR_L1 vs T_tr_L1 — DEV200, descriptive**

| line | n | acc TR_L1 | acc T_tr_L1 | Δ pp | b | c | exact two-sided p |
|---|---|---|---|---|---|---|---|
| L1  label, all rows | 200 | 67.00 | 71.00 | -4.00 | 30 | 38 | 0.3961 |
| L2  E3, all rows | 200 | 70.00 | 71.00 | -1.00 | 26 | 28 | 0.8919 |
| L3  label, rows parsed in both arms | 189 | 67.72 | 74.60 | -6.88 | 24 | 37 | 0.1237 |
| L4  rows where E0 = E3 in both arms (descriptive) | 164 | 72.56 | 70.73 | +1.83 | 23 | 20 | 0.7608 |
| L6  label, rows truncated in neither arm | 189 | 67.72 | 73.02 | -5.29 | 26 | 36 | 0.2529 |
| L7  label, parsed in both and truncated in neither | 182 | 68.13 | 75.27 | -7.14 | 22 | 35 | 0.1112 |

Unparsed (label): TR_L1 2 (1.00 %), T_tr_L1 9 (4.50 %), difference 3.50 pp ≤ 5.0. Truncated: TR_L1 6 (3.00 %), T_tr_L1 5 (2.50 %), difference 0.50 pp ≤ 5.0. Descriptive, raw p only.

## 9. Retrieval title share (freeze §2, §7.3; D8 §3 title rule)

Gold titles = the titles of the row's evidence ids. A gold title is found if its corpus title (sealed title map) equals the title of at least one of the 5 retrieved chunks.

| set | arm | query | found / all gold titles | found / gold titles present in the corpus | rows with at least one found / rows | rows with at least one found / rows that have a gold title | retrieval s per query |
|---|---|---|---|---|---|---|---|
| eval458 | RAG_ur | question_ur | 46 / 1393 (3.30 %) | 46 / 995 (4.62 %) | 41 / 458 (8.95 %) | 41 / 457 (8.97 %) | 1.4129 |
| eval458 | RAG_en | R1 reading | 207 / 1393 (14.86 %) | 207 / 995 (20.80 %) | 188 / 458 (41.05 %) | 188 / 457 (41.14 %) | 1.3853 |
| dev200 | RAG_ur | question_ur | 24 / 636 (3.77 %) | 24 / 439 (5.47 %) | 23 / 200 (11.50 %) | 23 / 200 (11.50 %) | 1.4393 |
| dev200 | RAG_en | R1 reading | 79 / 636 (12.42 %) | 79 / 439 (18.00 %) | 73 / 200 (36.50 %) | 73 / 200 (36.50 %) | 1.4126 |

Title map eval458: `outputs/efbpt/d13/rag/eval458/retrieval/title_map.jsonl` (MD5 `a019cd9eababdfb55b3c8e993cc1192b`), 1279 distinct gold titles, 924 present in the corpus.
Title map dev200: `outputs/efbpt/d13/rag/dev200/retrieval/title_map.jsonl` (MD5 `20975cdf857982a2db8b3d38fa88d8fa`), 613 distinct gold titles, 427 present in the corpus.

## 10. Time, generated tokens and truncated rows per arm (freeze C6; Amendment 1)

| set | arm | rows | truncated rows (%) | seconds per row (mean) | generated tokens per row (mean) | seconds total | generated tokens total |
|---|---|---|---|---|---|---|---|
| eval458 | A_r2_eval | 458 | 26 (5.68) | — | 471.53 | — | 215961 |
| eval458 | TR | 458 | 20 (4.37) | — | 446.12 | — | 204324 |
| eval458 | TR_L1_eval | 458 | 19 (4.15) | — | 443.25 | — | 203010 |
| eval458 | T_direct_F_eval | 458 | 5 (1.09) | 10.12 | 151.62 | 4633 | 69441 |
| eval458 | T_tr_R1 | 458 | 7 (1.53) | 9.93 | 150.49 | 4548 | 68926 |
| eval458 | T_tr_L1 | 458 | 6 (1.31) | 9.91 | 150.07 | 4537 | 68733 |
| eval458 | API_direct_ur_eval | 458 | 0 (0.00) | 5.32 | 3.00 | 2436 | 1373 |
| eval458 | API_tr_R1 | 458 | 0 (0.00) | 1.02 | 1.02 | 466 | 464 |
| eval458 | RAG_ur | 458 | 50 (10.92) | 39.87 | 576.33 | 18262 | 263961 |
| eval458 | RAG_en | 458 | 61 (13.32) | 39.83 | 603.43 | 18242 | 276373 |
| eval458 | RO_eval | 458 | 21 (4.59) | 29.90 | 424.84 | 13696 | 194577 |
| eval458 | Llama8B_A | 458 | 70 (15.28) | 24.28 | 521.40 | 11120 | 238799 |
| eval458 | Llama8B_TR | 458 | 24 (5.24) | 13.64 | 296.34 | 6245 | 135722 |
| eval458 | Gemma9B_A | 458 | 7 (1.53) | 10.86 | 147.26 | 4973 | 67444 |
| eval458 | Gemma9B_TR | 458 | 0 (0.00) | 8.38 | 116.20 | 3838 | 53220 |
| eval458 | Qwen8B_A | 458 | 79 (17.25) | 34.17 | 524.63 | 15651 | 240281 |
| eval458 | Qwen8B_TR | 458 | 29 (6.33) | 27.01 | 441.81 | 12372 | 202351 |
| dev200 | A_r2 | 200 | 11 (5.50) | — | 481.41 | — | 96282 |
| dev200 | TR | 200 | 9 (4.50) | — | 480.97 | — | 96194 |
| dev200 | TR_L1 | 200 | 6 (3.00) | — | 473.69 | — | 94737 |
| dev200 | T_direct_F | 200 | 2 (1.00) | — | 158.25 | — | 31649 |
| dev200 | API_direct_ur | 200 | 0 (0.00) | — | 3.00 | — | 600 |
| dev200 | RO | 200 | 9 (4.50) | — | 435.72 | — | 87144 |
| dev200 | T_tr_R1 | 200 | 4 (2.00) | 10.59 | 160.34 | 2118 | 32067 |
| dev200 | T_tr_L1 | 200 | 5 (2.50) | 10.90 | 164.97 | 2180 | 32994 |
| dev200 | API_tr_R1 | 200 | 0 (0.00) | 1.03 | 1.01 | 206 | 202 |
| dev200 | RAG_ur | 200 | 30 (15.00) | 44.47 | 614.07 | 8894 | 122813 |
| dev200 | RAG_en | 200 | 32 (16.00) | 43.52 | 631.03 | 8703 | 126206 |

Sealed reference arms carry no per-row time (—). API arms: seconds = latency of the request, tokens = completion tokens reported by the provider; API_tr_R1 on eval458: means and totals over the 456 answered rows. Truncated = the row reached the token limit (1024 new tokens for the local models; finish reason "length" for the API).

## 11. Guard log from the START files (freeze §7.5; reported, not a check of this script)

- eval458 T_direct_F_eval: no guard note.
- eval458 T_tr_R1: readings equal to the row's own question_en: 57 of 458 (reader R1).
- eval458 T_tr_L1: readings equal to the row's own question_en: 39 of 458 (reader R3).
- eval458 API_direct_ur_eval: 3 runs (resumed; finished rows were never generated again).
- eval458 API_tr_R1: readings equal to the row's own question_en: 57 of 458 (reader R1); 2 runs (resumed; finished rows were never generated again).
- eval458 RAG_ur: strings inside retrieved text that equal an evidence id or a gold fact (logged, not fatal): 6 — qid fdff69018a970c7f3bbc rank 2 gold_fact 'Antarctica has no permanent residents.' in "Demographics of Antarctica"; qid cf1365a8abd4e35c2f0b rank 1 evidence_id 'B-1' in "2017 in aviation"; qid a1dabe439511af470303 rank 1 evidence_id 'B-1' in "Heavy bomber"; qid a1dabe439511af470303 rank 2 evidence_id 'B-1' in "Boeing B-52 Stratofortress"; qid a1dabe439511af470303 rank 2 evidence_id 'B-1' in "Boeing B-52 Stratofortress"; qid a1dabe439511af470303 rank 5 evidence_id 'B-1' in "LGM-30 Minuteman"; C1 regression PASS (job 116805).
- eval458 RAG_en: strings inside retrieved text that equal an evidence id or a gold fact (logged, not fatal): 1 — qid a1dabe439511af470303 rank 2 evidence_id 'B-1' in "LGM-30 Minuteman"; C1 regression PASS (job 116805).
- eval458 RO_eval: readings equal to the row's own question_en: 57 of 458 (reader R1); C1 regression PASS (job 116805).
- eval458 Llama8B_A: prompt layouts: A 458 (the user message is the sealed Qwen3-14B user message of the row; freeze §7.4); C2 PASS (job 116449); repeat of the 5 S_TR rows in the full job identical to TEST-5: 5 of 5.
- eval458 Llama8B_TR: prompt layouts: A 1, B 457 (the user message is the sealed Qwen3-14B user message of the row; freeze §7.4); C2 PASS (job 116449); repeat of the 5 S_TR rows in the full job identical to TEST-5: 5 of 5.
- eval458 Gemma9B_A: prompt layouts: A 458 (the user message is the sealed Qwen3-14B user message of the row; freeze §7.4); C2 PASS (job 116449); repeat of the 5 S_TR rows in the full job identical to TEST-5: 5 of 5.
- eval458 Gemma9B_TR: prompt layouts: A 1, B 457 (the user message is the sealed Qwen3-14B user message of the row; freeze §7.4); C2 PASS (job 116449); repeat of the 5 S_TR rows in the full job identical to TEST-5: 5 of 5.
- eval458 Qwen8B_A: prompt layouts: A 458 (the user message is the sealed Qwen3-14B user message of the row; freeze §7.4); C2 PASS (job 116449); repeat of the 5 S_TR rows in the full job identical to TEST-5: 5 of 5.
- eval458 Qwen8B_TR: prompt layouts: A 1, B 457 (the user message is the sealed Qwen3-14B user message of the row; freeze §7.4); C2 PASS (job 116449); repeat of the 5 S_TR rows in the full job identical to TEST-5: 5 of 5.
- dev200 T_tr_R1: readings equal to the row's own question_en: 25 of 200 (reader R1).
- dev200 T_tr_L1: readings equal to the row's own question_en: 14 of 200 (reader R3).
- dev200 API_tr_R1: readings equal to the row's own question_en: 25 of 200 (reader R1).
- dev200 RAG_ur: strings inside retrieved text that equal an evidence id or a gold fact (logged, not fatal): 0; C1 regression PASS (job 116805).
- dev200 RAG_en: strings inside retrieved text that equal an evidence id or a gold fact (logged, not fatal): 0; C1 regression PASS (job 116805).

