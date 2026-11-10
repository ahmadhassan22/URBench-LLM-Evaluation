# D11 FULL readout (DEV200): jobs 115172 (teacher R1/R2/R3) and 115173 (student)

Read-only readout of the completed D11 RTG FULL run (explicit reading + teacher facts; freeze `ce7d6880`, Amendment 1 `0b7202af`). Sources: `outputs/efbpt/d11/teacher/dev200/{R1,R2,R3}/` (2 sealed files + `T_SEAL.json` each), `outputs/efbpt/d11/student/dev200/` (7 sealed files + `S_SEAL.json`), `outputs/efbpt/d11/api_direct/dev200/` (`A_SEAL.json`), the sealed references `outputs/efbpt/d10/student/d10_armA_r2.jsonl`, `d10_armTF.jsonl`, `outputs/efbpt/d11o/student/d11o_armGF.jsonl`, `d11o_armTO1.jsonl`, the three sealed D11-R DEV200 reading files, `logs/d11_{teacher,student}_FULL_dev200_1151{72,73}.log`, `sacct`. Every number below was recomputed with Python from the sealed jsonl files; where the sealed `d11_summary.json` reports the same quantity the two agree on every value. No generation was re-run. Besides this file, the files written are `docs/D11_DECISION.md` (§6) and `outputs/efbpt/d11r/_scores/scores_dev200.json` (§7, the D11-R scorer's own output, gitignored). `docs/D11_GO_EVAL458.md` is **not** written: it waits for the hand check (§8).

Conventions: "wins" = the first-named arm is right and the second is wrong; "losses" = the reverse; b = wins, c = losses. p = exact two-sided binomial on the discordant pairs (`scipy.stats.binomtest(min(b,c), b+c, 0.5)`, the freeze's McNemar, as `d5_balanced_extract.mcnemar`). q = flip precision = wins/(wins+losses). Unparsed = wrong. Gold is yes on 87/200 and no on 113/200 (floor 56.50).

**Headline.** TR − A_r2 = **+8.5 pp** (68.5% vs 60.0%, wins 39, losses 22, q = 0.64, p = 0.0396) → **GO** (§7: ≥ +8.0 and p < 0.05). DEV200-best of {TR 68.5, TR2 65.0, TR_L1 67.0} = **TR**. All fatal checks pass (V1 spread 4.5 pp, V2' 5/5, V3 0 fatal, V4 three MD5s = freeze §2, V8 998 prompts). V5 TR_wrong − A_r2 = −9.5 (WRONG-FACTS HARM, expected direction). §6 predictions: 4 hit / 2 miss (TR landed at +8.5, just under the predicted +9..+13 band; TR2 is −3.5 from TR, outside ±2). The gain is entirely on gold-no rows (A_r2 64.6% → TR 80.5%); gold-yes is flat (54.0% → 52.9%). TR − A_r2 is +16.1 pp on the 124 rows whose R1 reading recovers every gold proper noun (row-M1 = 1) and −6.5 pp on the 31 rows where it does not. The reader model answering alone in Urdu (API_direct_ur, descriptive) is 69.0%, equal to TR within noise (TR − API_direct_ur = −0.5 pp, b 26 / c 27, p = 1).

---

## 1. Provenance

**SLURM** (`sacct -j 115172,115173`):

| job | name | State | ExitCode | node | Start | End | Elapsed | Timelimit |
|---|---|---|---|---|---|---|---|---|
| 115172 | d11_teacher_FULL_dev200 | COMPLETED | 0:0 | L20001 | 2026-11-09T15:16:12 | 2026-11-09T17:09:14 | 01:53:02 | 03:00:00 |
| 115173 | d11_student_FULL_dev200 | COMPLETED | 0:0 | L20001 | 2026-11-09T17:09:14 | 2026-11-10T02:22:54 | 09:13:40 | 16:00:00 |

Job S started the second Job T ended (A1.3 `--dependency=afterok`). Both START files record git head `24ee1068384a63b77ae6d9bd6dfbeb9cb9a7164b`, dirty false = HEAD at readout time, tree clean before this commit.

**Contracts** (MD5 in the seals vs recomputed now):

| item | MD5 in seals / START | recomputed |
|---|---|---|
| freeze `docs/EFBPT_D11_RTG_FREEZE.md` | `ce7d68801c302fa5a8d173bd3ee93eb7` (T_SEAL ×3, S_SEAL, A_SEAL) | match |
| Amendment 1 `docs/EFBPT_D11_RTG_AMENDMENT_1.md` | `0b7202af45eac170ec2d8fe281269e9e` (T_SEAL ×3, S_SEAL, A_SEAL) | match |
| D11-O freeze | `ebd638eb4b4e48cde00202c8488df124` (T_SEAL ×3) | match |
| D10 freeze | `b3a822d8199034c7fe58d76cf5cc428f` (T_SEAL ×3, S_SEAL) | match |
| teacher script `d11_teacher.py` | `39fe1cae5b140051bee7b75a6ca20893` (T_SEAL ×3) | as sealed |
| student script `d11_student.py` | `36e29f8a0b7f8fad96b5834f5cf6a106` (S_SEAL, S_START) | as sealed |
| `d11_api_direct.py` | `4af85505d859b481aaf1ab0caf26905f` (A_SEAL) | as sealed |
| Urdu instruction `prompts/efbpt/plan_a_instruction_ur.txt` | `f3b58d766fe3ec2573ff4f24761cf0c9` (S_START) | = freeze §1 |
| sealed A_r2 `outputs/efbpt/d10/student/d10_armA_r2.jsonl` | `7ccec942c8f0eee08156d258baaf806d` (S_SEAL) | match |
| D10 derangement `outputs/efbpt/d10/teacher/derangement.json` | `8f00e2f9e54f27138964a0577a8eff52` (S_SEAL, recomputed_equal true) | match; 0 fixed points; every TR_wrong `facts_source_qid` = map[qid] |

**Teacher seals R1–R3** (`T_SEAL.json`, slurm 115172, mode FULL, version v1; all MD5/bytes/lines recomputed and matching, and identical in `S_SEAL.teacher_seal_files`):

| reader | key | reading file (MD5 = freeze §2 = R_SEAL) | end_time | parse_ok | repaired | removed | `T_START.json` | `t_facts.jsonl` (bytes / lines) |
|---|---|---|---|---|---|---|---|---|
| api_qwen3.8-max (SELECTED) | R1 | `1f5b2db3c1403f63b3525a1562edd176` | 15:55:05 | 200/200 | 0 | 0 | `362d97492534b9172644a5f25d41321e` | `160406e12205635e69f5c95140f9936f` (715216 / 200) |
| api_deepseek-v4-pro (RUNNER-UP) | R2 | `cd77944e45a98292777382b9d9f04aef` | 16:33:34 | 198/200 | 0 | 0 | `0574daf482df90fcbe8c08c74bb5dfb4` | `ad7cab74c16d5158354954a979823185` (710288 / 200) |
| L1_llama70b_awq (best local) | R3 | `0d06ec1a926fe3c186c125ad723139cf` | 17:09:11 | 200/200 | 0 | 0 | `82b98e132ce67fc97b366c46051ec8b5` | `6cee1621ecbffe7bb4043c558d84e327` (711786 / 200) |

Teacher facts per row: R1 min 6 / median 6 / max 6, empty 0, truncated 0; R2 min 0 / 6 / 6, empty 2 (parse failures `06215d4b6f1a4b975d9c`, `3157a0f428531ebed7b9`, both `finish_reason` length), truncated 2; R3 min 5 / 6 / 6, empty 0. V5 verdict filter removed 0 everywhere. Teacher input = the sealed reading on all 600 rows (`teacher_input_reading` == reading, recomputed).

**S_SEAL files** (slurm 115173, end 2026-11-10 02:22:52, `complete: true`; all 7 MD5/bytes/lines recomputed and matching):

| file | MD5 | bytes | lines |
|---|---|---|---|
| `outputs/efbpt/d11/student/dev200/S_START.json` | `ef830d4f0a8b7d8587a80e6cd900029c` | 17949 | 663 |
| `outputs/efbpt/d11/student/dev200/d11_armTR.jsonl` | `bab987cc492012f4e32dbcd348be9668` | 494393 | 200 |
| `outputs/efbpt/d11/student/dev200/d11_armTR2.jsonl` | `17f1c43d6406414d66aa28d769640ab6` | 486520 | 200 |
| `outputs/efbpt/d11/student/dev200/d11_armTR_L1.jsonl` | `b8db599f8c4b427ddcb2d772fd8f09aa` | 490780 | 200 |
| `outputs/efbpt/d11/student/dev200/d11_armTR_wrong.jsonl` | `9519b9adf215ac90eb6de743ec1dca57` | 496097 | 200 |
| `outputs/efbpt/d11/student/dev200/d11_armTR_noq.jsonl` | `77e8c509c5a5fb10cc9f9ab66f792fa6` | 408629 | 200 |
| `outputs/efbpt/d11/student/dev200/d11_summary.json` | `d729abd5907c5f6741b1ef4f49c29beb` | 12272 | 454 |

Seal files themselves (not self-listed): `S_SEAL.json` `7c412ddbb2035a25f88eb74adab0489b`; `T_SEAL.json` R1/R2/R3 1059/1067/1059 bytes; `A_SEAL.json` `d81d1c7f5ed4d2d7a9304af67532624f`. API_direct_ur: `api_direct_ur.jsonl` `31c82fa180a2677ce9df85db2ac3e0d7` (187438 bytes, 200 lines), `A_START.json` `caa0be6404b164e2f06dd6e53827a862`; 200 calls, all `finish_reason` stop, mean 3.0 completion tokens. Sealed references read here: TF_v1 `c54b7553baa5ffa77204bed1dab37250`, GF `448fec4127bb8060c10422d9aa42699f`, TO1 `6548b408a68a8e94bb65b9ca6d023fdd` (all = the D11-O S_SEAL). Every arm file carries the 200 DEV200 qids once, gold agrees with the pinned item set on every row, and `correct` == (`pred` == `gold`) on every row.

---

## 2. Accuracy table

All arms, n = 200, floor 56.50. SEC = secondary (Devanagari-aware) scorer; truncation = `truncated` flag (student) or `finish_reason != stop` (API). References sealed, not re-run: A_r2 60.0 (D10), TF_v1 65.0 (D10), GF 76.0 and TO1 74.0 (D11-O), API_direct_ur 69.0 (A_SEAL).

| arm | n | correct | acc % | SEC acc % | yes % | unparsed % (n) | trunc % (n) |
|---|---|---|---|---|---|---|---|
| A_r2 | 200 | 120 | 60.00 | 60.00 | 41.00 | 4.00 (8) | 5.50 (11) |
| TR | 200 | 137 | 68.50 | 68.50 | 33.50 | 1.50 (3) | 4.50 (9) |
| TR2 | 200 | 130 | 65.00 | 65.00 | 31.50 | 1.50 (3) | 4.00 (8) |
| TR_L1 | 200 | 134 | 67.00 | 67.00 | 30.00 | 1.00 (2) | 3.00 (6) |
| TR_wrong | 200 | 101 | 50.50 | 50.50 | 24.00 | 5.50 (11) | 5.00 (10) |
| TR_noq | 200 | 94 | 47.00 | 47.00 | 28.00 | 18.50 (37) | 3.00 (6) |
| API_direct_ur | 200 | 138 | 69.00 | 69.00 | 27.50 | 0.00 (0) | 0.00 (0) |
| TF_v1 | 200 | 130 | 65.00 | 65.00 | 36.00 | 0.50 (1) | 2.50 (5) |
| GF | 200 | 152 | 76.00 | 76.00 | 37.00 | 1.00 (2) | 2.50 (5) |
| TO1 | 200 | 148 | 74.00 | 74.00 | 32.00 | 1.00 (2) | 7.00 (14) |


TR2's two layout-A rows (empty facts) are the R2 parse-failure qids. Layouts: A_r2 A×200; TR, TR_L1, TR_wrong B×200; TR2 B×198 + A×2; TR_noq G×200. Predicted-yes falls with facts (A_r2 41.0 → TR 33.5 → TR_wrong 24.0), as in D10 (TF 36.0, TF_wrong 19.0). Generation lengths were not re-tabulated.

---

## 3. Paired comparisons (exact two-sided McNemar)

TR vs TO1 and TR vs GF are joined by qid to the sealed D11-O generations (same student, same DEV200 rows). All 20 pairs recomputed; the 9 that appear in `d11_summary.json` match it exactly (b, c, p).

| comparison | acc first | acc second | Δ pp | b (wins) | c (losses) | discordant | q = b/(b+c) | exact two-sided p |
|---|---|---|---|---|---|---|---|---|
| TR vs A_r2 | 68.50 | 60.00 | +8.5 | 39 | 22 | 61 | 0.639 | 0.03962 |
| TR2 vs A_r2 | 65.00 | 60.00 | +5.0 | 35 | 25 | 60 | 0.583 | 0.2451 |
| TR_L1 vs A_r2 | 67.00 | 60.00 | +7.0 | 45 | 31 | 76 | 0.592 | 0.1354 |
| TR_wrong vs A_r2 | 50.50 | 60.00 | -9.5 | 30 | 49 | 79 | 0.380 | 0.04217 |
| TR_noq vs A_r2 | 47.00 | 60.00 | -13.0 | 32 | 58 | 90 | 0.356 | 0.008046 |
| API_direct_ur vs A_r2 | 69.00 | 60.00 | +9.0 | 46 | 28 | 74 | 0.622 | 0.04739 |
| TF_v1 vs A_r2 | 65.00 | 60.00 | +5.0 | 38 | 28 | 66 | 0.576 | 0.2678 |
| GF vs A_r2 | 76.00 | 60.00 | +16.0 | 49 | 17 | 66 | 0.742 | 0.0001015 |
| TO1 vs A_r2 | 74.00 | 60.00 | +14.0 | 46 | 18 | 64 | 0.719 | 0.0006174 |
| TR vs TR2 | 68.50 | 65.00 | +3.5 | 17 | 10 | 27 | 0.630 | 0.2478 |
| TR vs TR_L1 | 68.50 | 67.00 | +1.5 | 23 | 20 | 43 | 0.535 | 0.7608 |
| TR vs TR_wrong | 68.50 | 50.50 | +18.0 | 53 | 17 | 70 | 0.757 | 1.92e-05 |
| TR vs TR_noq | 68.50 | 47.00 | +21.5 | 60 | 17 | 77 | 0.779 | 8.91e-07 |
| TR_L1 vs TR2 | 67.00 | 65.00 | +2.0 | 21 | 17 | 38 | 0.553 | 0.6271 |
| TR vs TO1 | 68.50 | 74.00 | -5.5 | 11 | 22 | 33 | 0.333 | 0.08014 |
| TR vs GF | 68.50 | 76.00 | -7.5 | 17 | 32 | 49 | 0.347 | 0.04438 |
| TR vs TF_v1 | 68.50 | 65.00 | +3.5 | 26 | 19 | 45 | 0.578 | 0.3713 |
| TR vs API_direct_ur | 68.50 | 69.00 | -0.5 | 26 | 27 | 53 | 0.491 | 1 |
| TR_L1 vs TO1 | 67.00 | 74.00 | -7.0 | 14 | 28 | 42 | 0.333 | 0.04356 |
| TR_L1 vs GF | 67.00 | 76.00 | -9.0 | 16 | 34 | 50 | 0.320 | 0.01535 |


Reading: TR beats A_r2 (p = 0.040) and sits 5.5 pp below the oracle-reading arm TO1 (p = 0.080) and 7.5 pp below the gold-facts ceiling GF (p = 0.044). TR vs TR_L1 is a wash (+1.5, b 23 / c 20); TR vs TR2 +3.5 (b 17 / c 10, p = 0.25). TR_wrong is 18.0 pp below TR (p = 2e-5) and 9.5 pp below A_r2 (p = 0.042). API_direct_ur (the reader answering alone, Urdu in / one word out) is 69.0%, indistinguishable from TR (b 26 / c 27).

---

## 4. Validity checks

**Verbatim from the student log (lines 1273–1280, 1303):**

```
V1 unparsed % {'A_r2': 4.0, 'TR': 1.5, 'TR2': 1.5, 'TR_L1': 1.0, 'TR_wrong': 5.5, 'TR_noq': 18.5} -> spread 4.50 pp over ['A_r2', 'TR', 'TR2', 'TR_L1', 'TR_wrong'] -> PASS   (TR_noq 18.50% reported, excluded; A1.2)
V2' PASS: 5/5 byte-identical to the sealed D10 A_r2 generations; sealed A_r2 reused
V3 leak guards: none fatal (teacher A1.1 (a)(b); student question_en / evidence ids / readings) -> PASS; V6 gold-fact restatements: {'TR': 1, 'TR2': 3, 'TR_L1': 2, 'TR_wrong': 1, 'TR_noq': 1}
V4 readings = sealed files: {'R1': '1f5b2db3c1403f63b3525a1562edd176', 'R2': 'cd77944e45a98292777382b9d9f04aef', 'R3': '0d06ec1a926fe3c186c125ad723139cf'}
V7 teacher seals verified (6 files) -> PASS; V8 998 facts prompts -> PASS; V9 {'TR': 1.0, 'TR2': 1.0, 'TR_L1': 1.0, 'TR_wrong': 1.0, 'TR_noq': 1.0}
V5 TR_wrong - A_r2 = -9.50 pp -> expected <= -3.0: as expected; A2.1 sign reading: WRONG-FACTS HARM (expected direction)
```

Teacher log (each of R1, R2, R3): `[leak-guard] (a) template with the reading slot blanked: question_en, gold facts, evidence ids, 613 titles clean`; `(b) 200 readings: no other-row question_en, no gold fact, no evidence id -> PASS; equal to own gold question_en 25 / 22 / 14 (reported); readings containing a required title 108 / 110 / 102 (reported, reading quality)`.

**Recomputed from the sealed files:**

| check | rule | recomputed | status |
|---|---|---|---|
| V1 unparsed spread (A_r2, TR, TR2, TR_L1, TR_wrong; A1.2) | ≤ 5.0 pp | 8 / 3 / 3 / 2 / 11 of 200 = 4.0 / 1.5 / 1.5 / 1.0 / 5.5 % → spread 4.5 pp | PASS |
| V1 TR_noq (layout G, excluded) | reported | 37/200 = 18.5 % (D10 TF_noq 21.5 %) | reported |
| V2' | 5 A_r2 rows regenerated byte-identical | S_START `V2_prime`: 5/5 identical, mismatched [] | PASS |
| V3 leak guards | fatal if any hit | teacher (a)(b) clean ×3; student question_en / evidence ids / reading strings clean in all 6 arms (from the logs and S_START `leak_guard`) | PASS |
| V4 readings = sealed files | MD5 = freeze §2 = R_SEAL; one record per qid, all ok, non-empty | R1 `1f5b2db3…`, R2 `cd77944e…`, R3 `0d06ec1a…` all match; 200/200 ok and non-empty each; teacher input == reading on 600/600 rows | PASS |
| V4 readings containing yes/no/ہاں/نہیں | reported | R1 1, R2 1, R3 0: qid `2e508efbf0b72f1af2c2` "Can quartz be useful for humans if plants die and there is **no** oxygen?" (benign) | reported |
| V4 readings equal to gold question_en (A1.1) | reported | R1 25, R2 22, R3 14 of 200 | reported |
| V5 TR_wrong − A_r2 | ≤ −3.0 expected (A2.1 sign rule) | 50.5 − 60.0 = **−9.5 pp** (b 30 / c 49, p = 0.042) → WRONG-FACTS HARM, expected direction | as expected |
| V6 verbatim gold-fact substrings in prompts (per arm) | reported | TR 1, TR2 3, TR_L1 2, TR_wrong 1, TR_noq 1 (see below) | reported |
| V6 per reader (gold-fact substrings inside the sealed teacher facts) | reported | R1 1 fact / 1 qid; R2 3 / 3; R3 2 / 2; every hit is the row's own gold fact | reported |
| V7 teacher seals | 6 files | all 6 MD5 / bytes / lines match now | PASS |
| V8 prompt content (A1.6 `<|im_end|>` cut) | prompt facts == sealed post-T2 facts of `facts_source_qid`; header once; layout A ⇔ no facts | 200 + 198 + 200 + 200 + 200 = **998** fact-bearing prompts, 0 mismatches, 0 header faults, `n_facts_given` = teacher `n_facts` on all 1000 rows; own-qid arms source = qid on 800/800; TR_wrong source = D10 derangement on 200/200 | PASS |
| V9 Latin fraction of injected facts | reported | TR 91260/91260 = 1.0000 (1200 facts); TR2 88591/88594 = 0.99997 (1187 facts; the 3 non-Latin letters are `ʻ`, the Hawaiian ʻokina, U+02BB); TR_L1 90324/90324 = 1.0000 (1199); TR_wrong = TR_noq = TR | reported |

**V6 hits** (gold fact → arm, qid; all are the row's own gold fact, carried through TR_wrong to the derangement partner):

| gold fact | reader(s) | arms and qids |
|---|---|---|
| "Ubuntu is a Linux distribution." | R1, R2, R3 | TR, TR2, TR_L1, TR_noq at `a4de29904c45a2d17511`; TR_wrong at `24a3aff40ef7fd04e208` (donor `a4de2990…`) |
| "Elagabalus was the 25th Roman Emperor." | R2, R3 | TR2, TR_L1 at `f07a6acf27e25dedbb8d` |
| "Joe Biden is a member of the Democratic Party" | R2 | TR2 at `703db4629ff46d82a205` |

---

## 5. Gold-yes vs gold-no accuracy per arm

| arm | gold yes acc (87) | gold no acc (113) | pred yes \| gold yes | pred yes \| gold no | unparsed yes / no |
|---|---|---|---|---|---|
| A_r2 | 47/87 = 54.02% | 73/113 = 64.60% | 47 (54.0%) | 35 (31.0%) | 3 / 5 |
| TR | 46/87 = 52.87% | 91/113 = 80.53% | 46 (52.9%) | 21 (18.6%) | 2 / 1 |
| TR2 | 40/87 = 45.98% | 90/113 = 79.65% | 40 (46.0%) | 23 (20.4%) | 3 / 0 |
| TR_L1 | 41/87 = 47.13% | 93/113 = 82.30% | 41 (47.1%) | 19 (16.8%) | 1 / 1 |
| TR_wrong | 20/87 = 22.99% | 81/113 = 71.68% | 20 (23.0%) | 28 (24.8%) | 7 / 4 |
| TR_noq | 27/87 = 31.03% | 67/113 = 59.29% | 27 (31.0%) | 29 (25.7%) | 20 / 17 |
| API_direct_ur | 40/87 = 45.98% | 98/113 = 86.73% | 40 (46.0%) | 15 (13.3%) | 0 / 0 |
| TF_v1 | 45/87 = 51.72% | 85/113 = 75.22% | 45 (51.7%) | 27 (23.9%) | 0 / 1 |
| GF | 57/87 = 65.52% | 95/113 = 84.07% | 57 (65.5%) | 17 (15.0%) | 1 / 1 |
| TO1 | 50/87 = 57.47% | 98/113 = 86.73% | 50 (57.5%) | 14 (12.4%) | 1 / 1 |


Wins and losses vs A_r2 by gold label:

| comparison | wins gold yes | wins gold no | losses gold yes | losses gold no |
|---|---|---|---|---|
| TR vs A_r2 | 13 | 26 | 14 | 8 |
| TR_L1 vs A_r2 | 17 | 28 | 23 | 8 |
| TR2 vs A_r2 | 10 | 25 | 17 | 8 |
| TR_wrong vs A_r2 | 6 | 24 | 33 | 16 |


The whole TR gain is on gold-no rows: 64.6 % → 80.5 % (+15.9 pp; 26 wins / 8 losses), while gold-yes is flat, 54.0 % → 52.9 % (13 wins / 14 losses). Facts lower the predicted-yes rate on both labels (gold-no pred-yes 31.0 % → 18.6 %; gold-yes 54.0 % → 52.9 %). The same shape holds for TR2 and TR_L1 (gold-yes 46.0 % and 47.1 %, below A_r2) and in the extreme for TR_wrong (gold-yes 23.0 %: 33 of its 49 losses are gold-yes rows pushed to "no"). TO1 and GF differ exactly here: with oracle reading or gold facts the gold-yes side rises too (57.5 %, 65.5 %). API_direct_ur answers "no" most of all (pred-yes 27.5 %; gold-no 86.7 %, gold-yes 46.0 %).

---

## 6. §6 predictions and §7 decision

| §6 prediction | observed | hit |
|---|---|---|
| TR − A_r2 in [+9, +13] | +8.5 | **miss** (0.5 pp under the band) |
| TR2 within ±2 of TR | −3.5 | **miss** |
| TR_L1 − A_r2 in [+7, +11] | +7.0 | hit |
| TR_L1 ≥ TR − 2 (mechanism = the explicit reading stage, not the API model) | −1.5 | hit |
| TR_noq ≤ TR | −21.5 | hit |
| TR_wrong ≤ A_r2 − 3 | −9.5 | hit |

**§7 decision line (TR vs A_r2, exact McNemar):** Δ = +8.5 pp (137 vs 120 of 200), b = 39, c = 22, p = 0.0396. +8.5 ≥ +8.0 and p < 0.05 → **GO → eval458 once (§8)**. DEV200-best of {TR 68.5, TR2 65.0, TR_L1 67.0} = **TR** (no tie). Recorded in `docs/D11_DECISION.md` (written with this readout). `docs/D11_GO_EVAL458.md` is not written here; it waits for the hand check of §8.

Margin note (descriptive, not part of the rule): the GO gate is met by one discordant pair. With b = 38, c = 23 the delta would still be +8.0 but p = 0.072, so the verdict would be CONDITIONAL; the pre-registered band started at +9.

---

## 7. Mechanism

**D11-R scorer on DEV200** (`d11r_score.py --set dev200 --readers api_qwen3.8-max api_deepseek-v4-pro L1_llama70b_awq --rows`; output `outputs/efbpt/d11r/_scores/scores_dev200.json`, MD5 `174cca057bd22856aedb5323bc72d3d1`; 155/200 rows have ≥ 1 gold proper noun; the item file MD5 was asserted by the loader):

| reader | key | M1 macro | M1 micro | M2 chrF (corpus / sentence mean) | M3 exact | M4 title presence | RDEV300 M1 / M2 (D11-R readout) |
|---|---|---|---|---|---|---|---|
| api_qwen3.8-max | R1 | **89.69** | 90.29 | **70.94** / 70.72 | 15.0 | 91.51 | 91.55 / 73.16 |
| api_deepseek-v4-pro | R2 | 88.40 | 89.08 | 70.52 / 70.39 | 12.5 | 91.89 | 91.93 / 72.38 |
| L1_llama70b_awq | R3 | 82.84 | 84.71 | 64.32 / 64.29 | 8.0 | 83.40 | (best local) |

On DEV200 the ordering is R1 > R2 > R3 on both M1 and M2 (on RDEV300 R2 led M1 by 0.4). The student ordering TR 68.5 > TR_L1 67.0 > TR2 65.0 does not follow M1/M2 between R2 and R3 (TR_L1 − TR2 = +2.0, b 21 / c 17, p = 0.63: noise-level).

**Student accuracy split by row-M1 of the reading used** (row-M1 = share of the gold question's proper nouns recovered by the reading; `None` = the gold question has no proper noun, 45 rows):


TR split by R1 row-M1:

| R1 row-M1 | n | TR correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses |
|---|---|---|---|---|---|---|
| m1=1 | 124 | 94 (75.81%) | 74 (59.68%) | +16.1 | 29 | 9 |
| m1<1 | 31 | 13 (41.94%) | 15 (48.39%) | -6.5 | 4 | 6 |
| no proper nouns | 45 | 30 (66.67%) | 31 (68.89%) | -2.2 | 6 | 7 |

TR_L1 split by R3 row-M1:

| R3 row-M1 | n | TR_L1 correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses |
|---|---|---|---|---|---|---|
| m1=1 | 108 | 83 (76.85%) | 66 (61.11%) | +15.7 | 28 | 11 |
| m1<1 | 47 | 24 (51.06%) | 23 (48.94%) | +2.1 | 12 | 11 |
| no proper nouns | 45 | 27 (60.00%) | 31 (68.89%) | -8.9 | 5 | 9 |

TR2 split by R2 row-M1:

| R2 row-M1 | n | TR2 correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses |
|---|---|---|---|---|---|---|
| m1=1 | 122 | 87 (71.31%) | 74 (60.66%) | +10.7 | 24 | 11 |
| m1<1 | 33 | 16 (48.48%) | 15 (45.45%) | +3.0 | 6 | 5 |
| no proper nouns | 45 | 27 (60.00%) | 31 (68.89%) | -8.9 | 5 | 9 |


Reading: when the reading recovers every gold proper noun, teacher facts help a lot (TR +16.1 pp over A_r2 on 124 rows, 29 wins / 9 losses; TR_L1 +15.7 on 108 rows). When the reading drops or garbles a proper noun the facts are about the wrong thing and TR is at or below A_r2 (−6.5 pp on 31 rows for R1; +2.1 on 47 rows for R3). On the 45 rows without proper nouns (common-noun questions such as "do silicone suits make judo difficult?") facts do not help any arm (TR −2.2, TR_L1 −8.9, TR2 −8.9). A_r2 itself is 59.7 % / 48.4 % / 68.9 % on the three R1 strata, so the stratum with imperfect readings is also the harder stratum for the unaided student. These are post-hoc splits on 31–47-row strata; descriptive only.

---

## 8. Hand-check rows

Format: qid, gold, Urdu question, gold question_en, the reading used (row-M1 in parentheses; "-" = no gold proper noun), the first two teacher facts, A_r2 answer, arm answer. Answers show the frozen extractor's `pred` in bold and the first ~110 characters of the generation (whitespace collapsed). Rows are in DEV200 file (qid) order. Generations are shown as they are; the extractor is Rule 2 ("last mention wins"), so the bold label can differ from the opening word (5 of the 22 TR losses are rows where A_r2's generation opens with ہاں but is scored "no" and matches gold "no": rows 8, 13, 14, 20, 22; 3 of the 39 TR wins have TR opening with ہاں and scored "no": `0fe3edf35bd1a1e662a9`, `8d06b619a7045ed02f51`, `c70d40b57fa67ea13ed6`). Across all 200 rows the opening word disagrees with `pred` on 31 A_r2 rows (22 of them scored correct) and 24 TR rows (10 scored correct); this is the frozen scorer's behaviour, recorded for the hand check, not re-scored.

Of the 22 TR losses, 14 are gold-yes rows that TR turned to "no" (rows 1–7, 9, 11, 12, 17–19, 21) and 8 are gold-no rows that TR turned to "yes" (8, 10, 13–16, 20, 22). Reading faults visible in the losses: row 4 "celery" read as "ajwain" (carom), row 8 "John Key" read as "Janaki", row 11 "sables" read as "Sybil", row 12 "shark goddess" read as "Sharkboy", row 13 "Samsung Galaxy A10e" lost to "a GB", row 19 "Ugly Betty" dropped. Several rows with a perfect reading are lost because the facts are true but beside the point and the student defaults to "no" (rows 1, 3, 9, 18, 21).

### 8.1 TR vs A_r2: the 22 LOSS rows (A_r2 right, TR wrong)

| # | qid | gold | Urdu question | gold question_en | R1 reading (row-M1) | fact 1 | fact 2 | A_r2 | TR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `01c27885cede6741bc53` | yes | کیا سپرنیچرل نے 2001 میں دی سی ڈبلیو پر شروع ہونے والے شو کے سیزنز کا ریکارڈ توڑا؟ | Did Supernatural break 2001 CW debuting shows seasons record? | Did Supernatural break the record for seasons of a show that started on The CW in 2001? (1.00) | Supernatural is an American fantasy horror television series created by Eric Kripke. | The CW is an American English-language television network that was formed in 2006 through the merger of UPN and The WB. | **yes** — ہاں، سپرنیچرل (Supernatural) نے 2001 میں دی سی ڈبلیو (The CW) پر شروع ہونے والے شو کے سیزنز کا ریکارڈ توڑا۔ دی | **no** — نہیں، **Supernatural** نے 2001 میں The CW پر شروع ہونے والے شو کے سیزنز کا ریکارڈ توڑا نہیں۔ اس کی وجہ یہ ہے ک |
| 2 | `15885efe6f91724c16d9` | yes | کیا آوڈی آر ایٹ وی-ٹین پلس کے لیے صوتی رکاوٹ توڑنا ممکن نہیں؟ | Is sound barrier too much for Audi R8 V-10 Plus to break? | Is it not possible for the Audi R8 V10 Plus to break the sound barrier? (1.00) | The Audi R8 is a mid-engined, all-wheel drive supercar manufactured by Audi. | The Audi R8 V10 Plus is a high-performance variant of the Audi R8. | **yes** — ہاں، **آوڈی آر ایٹ وی-ٹین پلس (Audi A8 L TFSI)** کے لیے **صوتی رکاوٹ توڑنا (sound deadening)** ممکن ہے۔ اس کے  | **no** — نہیں، آوڈی آر8 V10 Plus کے لیے صوتی رکاوٹ (ساؤنڈ بیریئر) توڑنا ممکن نہیں۔ اس کی وجہ یہ ہے کہ آوڈی R8 V10 Plus  |
| 3 | `17b8fa74f450c4d0e56b` | yes | کیا ڈزنی کی ایلس ان ونڈر لینڈ میں کسی تہوار کی خوشی منائی جاتی ہے؟ | Does Disney's Alice in Wonderland involve the celebration of a holiday? | Is any festival celebrated in Disney's Alice in Wonderland? (1.00) | Disney's Alice in Wonderland is a 1951 American animated fantasy film produced by Walt Disney Productions. | The film is an adaptation of Lewis Carroll's 1865 novel Alice's Adventures in Wonderland. | **yes** — ہاں، ڈزنی کی فلم **"ایلس ان ونڈر لینڈ"** (Alice in Wonderland) میں **تہوار کی خوشی** منائی جاتی ہے۔ یہ تہوار * | **no** — اس سوال کا جواب یہ ہے کہ **ڈزنی کی "ایلس ان ونڈر لینڈ" (1951) میں کسی تہوار کی خوشی منائی نہیں جاتی**۔ فلم میں |
| 4 | `2e8e9e51f0f290e32d27` | yes | کیا کسی ثقافت نے اجوائن کو موت سے منسلک کیا؟ | Did any cultures associate celery with death? | Has any culture associated ajwain with death? (-) | Ajwain is a herb also known as Trachyspermum ammi or bishop's weed. | Ajwain seeds are traditionally used in Ayurvedic medicine and cooking in India. | **yes** — ہاں، کچھ ثقافتوں میں اجوائن (جیو) کو موت یا اس سے متعلق تصورات سے منسلک کیا گیا ہے۔ ایک مثال یہ ہے کہ یونانی ت | **no** — نہیں، کسی معروف ثقافت کے مطابق اجوائن (Ajwain) کو موت سے منسلک نہیں کیا گیا۔ درحقیقت، اجوائن کو مختلف ثقافتوں  |
| 5 | `3570d57f16a40488ff42` | yes | کیا سلیکون کے سوٹ جودو کو مشکل بناتے ہیں؟ | Do silicone suits make judo difficult? | Do silicone suits make judo difficult? (-) | Silicone is a synthetic rubber-like material that is flexible and resistant to extreme temperatures. | Judo is a combat sport that involves grappling and throwing techniques. | **yes** — ہاں، سلیکون کے سوٹ جو دو کو مشکل بناتے ہیں۔ یہ بیان اکثر ایسی کہانیوں یا محاوروں میں استعمال ہوتا ہے جہاں "سلی | **no** — نہیں، سلیکون کے سوٹ جودو کو مشکل نہیں بناتے۔ درحقیقت، سلیکون کا استعمال جودو کے سوٹ (جودو یو نیا یا جودو گی) ک |
| 6 | `38ef97eb7cdd4200fd00` | yes | کیا سنہری بالوں والی سبز آنکھوں والی سارہ پیکسٹن کو لاطینی سمجھا جاتا ہے؟ | Is blonde hair green eyed Sara Paxton considered a Latino? | Is blonde, green-eyed Sarah Paxton considered Latina? (0.67) | Sarah Paxton is an American actress. | Latino people are those from or with ancestry from Latin America. | **yes** — ہاں، سارہ پیکسٹن (Sarah Jessica Parker) کو عام طور پر "لاطینی" سمجھا جاتا ہے۔ وہ ایک امریکی اداکارہ ہیں جو فلم | **no** — نہیں، سنہری بالوں والی سبز آنکھوں والی Sarah Paxton کو لاطینی (Latino) نہیں سمجھا جاتا۔ اس کی وجہ یہ ہے کہ لاط |
| 7 | `48da75d87c66754ccc2e` | yes | کیا پرل ہاربر پر حملے کے دوران امریکی ہاؤس اسپیکر کے طور پر خدمات انجام دینے والی واحد خاتون زندہ تھیں؟ | Was only woman to serve as U.S. Speaker of the House alive during the attack on Pearl Harbor? | Was the only woman to serve as Speaker of the US House alive during the attack on Pearl Harbor? (1.00) | The Speaker of the United States House of Representatives is the presiding officer of the United States House of Representatives. | Nancy Pelosi is the only woman to have served as Speaker of the United States House of Representatives. | **yes** — اس سوال کا جواب یہ ہے: **ہاں، ہاں**۔ **نینا ہیس** (Nancy Pelosi) امریکی ہاؤس آف ریپریزنٹیٹیو (ہاؤس اسپیکر) کے  | **no** — اس سوال کا جواب یہ ہے: **نہیں**۔ **وجہ:** - **پرل ہاربر پر حملہ** 1941 کی 7 دسمبر کو ہوا تھا۔ - **نینسی پیلوسی |
| 8 | `5897ec22db850f7b416e` | no | کیا 2020 تک جان کی کے بعد زیادہ خواتین وزیر اعظم بنی ہیں بہ نسبت ان کے پہلے؟ | As of 2020 have more women succeeded John Key than preceded him? | Have more women become Prime Ministers after Janaki than before her, up to 2020? (0.33) | Janaki is the first female head of government in the world, serving as the Prime Minister of Ceylon from 1960 to 1965 and again from 1970 to 1977, and from 1994 to 2000. | Sirimavo Bandaranaike, also known as Janaki, was the world's first female head of government. | **no** — ہاں، 2020 تک جان کی کے بعد زیادہ خواتین وزیر اعظم بنی ہیں بہ نسبت ان کے پہلے۔ جان کی (Janet Jackson) 1990 میں  | **yes** — ہاں، 2020 تک جانکی (Janaki) کے بعد زیادہ خواتین وزیر اعظم بنی ہیں بہ نسبت ان کے پہلے۔ جانکی (Sirimavo Bandaran |
| 9 | `660bfdc17b47f42facac` | yes | کیا مووس کو ارینڈیل کے سلطنت کے قریب کام کے لیے استعمال کیا جاتا ہے؟ | Are moose used for work near the kingdom of Arendelle? | Are moose used for work near the kingdom of Arendelle? (1.00) | Moose are the largest members of the deer family and are native to northern regions of North America, Europe, and Asia. | Moose have been used for their meat, hides, and antlers, but are not typically used for work in the same way as horses or oxen. | **yes** — اس سوال کا جواب یہ ہے: **ہاں، مووس کو ارینڈیل کے سلطنت کے قریب کام کے لیے استعمال کیا جاتا ہے۔** یہ بیان ایک ا | **no** — نہیں، مووس کو ارینڈیل کی سلطنت کے قریب کام کے لیے استعمال نہیں کیا جاتا۔ اس کی وجہ یہ ہے کہ: - ارینڈیل ایک خیا |
| 10 | `703db4629ff46d82a205` | no | کیا کین اس سال کی پرائمریز میں جو بائیڈن کو چیلنج کر سکتا ہے؟ | Can Kane challenge Joe Biden in this year's primaries? | Can Ken challenge Joe Biden in this year's primaries? (0.67) | Joe Biden is the 46th President of the United States. | To be eligible to serve as the President of the United States, a person must be a natural-born citizen of the United States, at least 35 years old, and a resident of the United States for at least 14 years. | **no** — اس سال (2024) کی پرائمریز میں جو بائیڈن کو چیلنج کر سکتا ہے، اس کے بارے میں اب تک کوئی واضح اور معتبر نام نہیں | **yes** — جی ہاں، کوئی بھی شخص جو **US کا نیتی ہونے والے شہری** ہو، **35 سال یا اس سے زیادہ** ہو، اور **14 سال سے زیادہ  |
| 11 | `80aa769f55b14c1e4d8d` | yes | کیا سیبل وولورین سے متعلق ہیں؟ | Are sables related to wolverines? | Is Sybil related to Wolverine? (-) | Sybil is a fictional character in the Marvel Comics universe. | Wolverine is a fictional character and one of the most well-known superheroes in the Marvel Comics universe. | **yes** — ہاں، سیبل (Sable) وولورین (Wolverine) سے متعلق ہے۔ سیبل ایک خاص قسم کی چوہا ہے جو یوکرین اور روس کے شمالی علاق | **no** — سیبل وولورین سے متعلق ہیں، لیکن ان کا تعلق ایک دوسرے سے مختلف طریقے سے ہے۔ سیبل ایک موتی (mutant) ہے جو "ہیلیئ |
| 12 | `987260ffef60cb2c5439` | yes | کیا پرل ہاربر شارک دیوی کا خیالی گھر ہے؟ | Is Pearl Harbor the mythical home of a shark goddess? | Is Pearl Harbor the fictional home of Sharkboy? (1.00) | Pearl Harbor is a lagoon harbor on the island of Oahu, Hawaii, United States. | Pearl Harbor is the headquarters of the United States Pacific Fleet. | **yes** — ہاں، پرل ہاربر شارک دیوی کا خیالی گھر ہے۔ یہ ایک خیالی یا فیکشنل گھر ہے جو "شارک دیوی" (Shark Girl) کے کردار ک | **no** — نہیں، پرل ہاربر شارک دیوی کا خیالی گھر نہیں ہے۔ حقائق کے مطابق: - **پرل ہاربر** ہواہی کے جزیرہ **اوہو** پر ایک |
| 13 | `9bb59488a227a5f5cd62` | no | کیا آپ جی بی پر گیم آف تھرونز کے ہر ایچ ڈی قسط کو محفوظ کر سکتے ہیں؟ | Can you save every HD episode of Game of Thrones on Samsung Galaxy A10e? | Can you save every HD episode of Game of Thrones on a GB? (0.50) | Game of Thrones is an American fantasy drama television series that aired from 2011 to 2019. | Each episode of Game of Thrones has a typical runtime of around 50-80 minutes. | **no** — ہاں، آپ جیبی (Jio) پر گیم آف تھرونز (Game of Thrones) کے ہر ایچ ڈی (HD) قسط کو محفوظ کر سکتے ہیں۔ اس کے لیے آپ | **yes** — ہاں، آپ گیم آف تھرونز کے ہر ایچ ڈی (HD) قسط کو جی گیگا بائیٹ (GB) پر محفوظ کر سکتے ہیں۔ چونکہ ہر HD قسط کا فائ |
| 14 | `a8bbf2e84d0649b1c52c` | no | کیا راہبوں کو جنگ میں حصہ لینے سے منع کیا گیا ہے؟ | Are monks forbidden from engaging in warfare? | Are monks forbidden from taking part in war? (-) | A monk is a person who practices religious asceticism, living either alone or in a community with others who share the same goal. | Monasticism is a form of religious life in which monks or nuns live in a community under a common rule, and the main goal is to seek spiritual growth. | **no** — ہاں، راہبوں کو جنگ میں حصہ لینے سے منع کیا گیا ہے۔ اس کی بنیادی وجہ یہ ہے کہ راہب (monk) یا مونسٹر (monastic)  | **yes** — اس سوال کا جواب یہ ہے کہ **راہبوں کو جنگ میں حصہ لینے سے منع کیا گیا ہے یا نہیں**، یہ مختلف مذہبی تعلیمات اور  |
| 15 | `b3443bf3c312c9d144e4` | no | کیا پانتھیون میں خواتین کی پہلی دو تدفینوں کے درمیان مکمل نیپچون مدار موجود ہے؟ | Is there a full Neptunian orbit between the first two burials of women in the Panthéon? | Is there a full Neptune orbit between the first two burials of women in the Pantheon? (0.00) | Neptune is a planet in the Solar System. | The orbital period of Neptune is approximately 165 Earth years. | **no** — اس سوال کا جواب یہ ہے: **نہیں**۔ **پانتھیون** (Pantheon) رومی مذہبی گرجہ ہے جو روم، امریکہ میں واقع ہے۔ یہ ایک | **yes** — ہاں، پانتھیون میں خواتین کی پہلی دو تدفینوں کے درمیان مکمل نیپچون کا مدار موجود ہے۔ **تفصیل:** - **ماری کیوری* |
| 16 | `b8df6ac181c4c5b280f9` | no | کیا بوتلنوز ڈولفن نظریاتی طور پر انسانی خواتین سے زیادہ نسل پیدا کر سکتے ہیں؟ | Can bottlenose dolphins hypothetically outbreed human women? | Can bottlenose dolphins theoretically produce more offspring than human females? (-) | Bottlenose dolphins typically give birth to a single calf after a gestation period of almost 12 months. | Female bottlenose dolphins usually do not produce more than one calf every 3-6 years. | **no** — اس سوال کا جواب یہ ہے کہ **بوتلنوز ڈولفن (Bottlenose dolphins)** نظریاتی طور پر انسانی خواتین سے زیادہ نسل پید | **yes** — ہاں، **بوتلنوز ڈولفن** نظریاتی طور پر **انسانی خواتین سے زیادہ نسل پیدا کر سکتی ہیں**۔ یہ اس لیے ہے کہ: - **بو |
| 17 | `c921bdade889def26cf4` | yes | کیا آئسوپروپائل الکحل میں نمک کی حل پذیری کم ہے؟ | In isopropyl alcohol, is the solubility of salt low? | Is the solubility of salt low in isopropyl alcohol? (-) | Solubility is a measure of the maximum amount of a substance that can dissolve in a given amount of solvent at a given temperature and pressure. | Salt is the common name for the chemical compound sodium chloride (NaCl). | **yes** — ہاں، **اسوپروپائل الکحل** (Isopropyl alcohol) میں **نمک (NaCl)** کی **حل پذیری (solubility)** **کم** ہوتی ہے۔  | **no** — ہاں، **آئسوپروپائل الکحل میں نمک (سائیڈم چلورائیڈ یا NaCl) کی حل پذیری کم ہے**۔ اس کی وجوہات یہ ہیں: 1. **نمک  |
| 18 | `d47120efd0f09badd848` | yes | کیا مینٹھول کرسمس سے منسلک ہے؟ | Is menthol associated with Christmas? | Is menthol associated with Christmas? (1.00) | Menthol is a waxy, crystalline substance, clear or white in color, which is solid at room temperature. | Menthol is used in a variety of products, including cough drops, candies, and topical pain relief creams. | **yes** — ہاں، مینٹھول (Menthol) کرسمس سے منسلک ہے۔ یہ تعلق اس لیے ہے کیونکہ مینٹھول کا استعمال کرسمس کے دوران ہونے والی | **no** — ہاں، مینٹھول کرسمس سے منسلک ہے، لیکن اس کا تعلق کرسمس کے مذہبی پہلو سے نہیں، بلکہ ثقافتی اور مصنوعات کے استعما |
| 19 | `de62d38623fde2c2285a` | yes | کیا ایک ہفتہ ہر قسط دیکھنے کے لیے کافی وقت ہوگا؟ | Would a week be enough time to watch every episode of Ugly Betty? | Will one week be enough time to watch every episode? (0.00) | A week is a unit of time, equal to 7 days. | The length of a week is standardized across the world. | **yes** — ہاں، ایک ہفتہ ہر قسط دیکھنے کے لیے کافی وقت ہو سکتا ہے، لیکن یہ اس بات پر بھی منحصر ہے کہ: 1. **قسط کی لمبائی* | **no** — اس سوال کا جواب یہ ہے کہ **ایک ہفتہ ہر قسط دیکھنے کے لیے کافی وقت ہو یا نہیں**، یہ **قسط کی لمبائی** پر منحصر  |
| 20 | `e12fef0504a959e49b23` | no | کیا نیو انگلینڈ کے کسی شخص کو کافی اگا کر منافع ہو سکتا ہے؟ | Can someone from New England profit by growing coffee? | Can someone in New England make a profit growing coffee? (1.00) | New England is a region in the northeastern United States. | New England has a temperate climate with cold winters and warm summers. | **no** — ہاں، نیو انگلینڈ کے کسی شخص کو کافی اگانے سے منافع ہو سکتا ہے، لیکن یہ منافع کچھ عوامل پر منحصر ہے۔ یہاں کچھ ا | **yes** — ہاں، نیو انگلینڈ کے کسی شخص کو کافی اگا کر منافع ہو سکتا ہے، لیکن یہ کچھ خصوصی حالات اور تیکنیکی اقدامات کے سا |
| 21 | `e6d3973ed3feb8a42928` | yes | کیا افریقہائزڈ مکھیاں کثیر الثقافتی سمجھی جا سکتی ہیں؟ | Can Africanized bees be considered multicultural? | Can Africanized bees be considered multicultural? (1.00) | Africanized bees are a hybrid species of bee created by cross-breeding the European honey bee with the African honey bee. | Africanized bees are known for their aggressive behavior when threatened or provoked. | **yes** — ہاں، افریقہائزڈ مکھیاں (Africanized honey bees) کثیر الثقافتی (polylectic) سمجھی جا سکتی ہیں۔ "کثیر الثقافتی"  | **no** — نہیں، افریقہائزڈ مکھیاں کثیر الثقافتی (Multicultural) نہیں سمجھی جا سکتیں۔ اس کی وجہ یہ ہے کہ: 1. **کثیر الثقا |
| 22 | `ff3811735ededd8ec3a7` | no | کیا سلائم للیز اسپیرگس سے مختلف سائنسی خاندان میں ہیں؟ | Are slime lilies in a different scientific family than asparagus? | Are slime lilies in a different scientific family from asparagus? (-) | Slime lily is a common name for plants in the genus Glaucium. | Glaucium is a genus of flowering plants in the family Papaveraceae. | **no** — ہاں، **سلائم للیز** (Salamandra lanzari) اور **اسپیرگس** (Salamandra salamandra) دونوں **سلائم** (Amphibians)  | **yes** — ہاں، **Slime lily** (جس کا سائنسی نام **Glaucium** ہے) اور **Asparagus** (جس کا سائنسی نام **Asparagus** ہے) * |

### 8.2 TR vs A_r2: the first 15 of 39 WIN rows (TR right, A_r2 wrong)

| # | qid | gold | Urdu question | gold question_en | R1 reading (row-M1) | fact 1 | fact 2 | A_r2 | TR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `05e9cdc44f1b066badd7` | no | کیا خواتین میں مونچھیں ہونا عام بات ہے؟ | Is it common for women to have moustaches? | Is it normal for women to have a mustache? (-) | Women are adult human females. | A mustache is a growth of hair on the upper lip. | **yes** — ہاں، خواتین میں مونچھیں ہونا عام ہے۔ یہ ایک طبی طبیعت ہے جو کچھ خواتین میں پیدا ہوتی ہے۔ مونچھیں کی ترقی ہارمو | **no** — نہیں، خواتین میں مونچھیں ہونا عام بات نہیں ہے۔ عام طور پر، خواتین کے چہرے پر ہیلیوس ہیئر (vellus hair) ہوتا ہے |
| 2 | `064d8faaa8f164b55270` | yes | کیا اپالو 15 کے عملے کو ایک پہیے والی سائیکل چلانے میں دشواری ہوگی؟ | Would the crew of Apollo 15 have difficulty riding a unicycle? | Would the crew of Apollo 15 have difficulty riding a unicycle? (1.00) | Apollo 15 was the fourth manned mission to land on the Moon. | The crew of Apollo 15 consisted of David Scott, James Irwin, and Alfred Worden. | **no** — اپولو 15 کے عملے کو ایک پہیے والی سائیکل چلانے میں دشواری ہو سکتی ہے، کیونکہ ایک پہیے والی سائیکل چلانا دو پہی | **yes** — ہاں، اپولو 15 کے عملے کو ایک پہیے والی سائیکل (Unicycling) چلانے میں دشواری ہوسکتی ہے۔ یہ دشواری ان عوامل کی ب |
| 3 | `0fe3edf35bd1a1e662a9` | no | کیا گینڈالف فرضی طور پر چارمد کے بارباس کے لیے ایک زبردست دشمن ہے؟ | Is Gandalf hypothetically a formidable foe for Charmed's Barbas? | Would Gandalf hypothetically be a formidable foe for Charmed's Barbas? (1.00) | Gandalf is a powerful wizard in J.R.R. Tolkien's Middle-earth legendarium. | Gandalf is known for his magical abilities, including fire magic and illusions. | **yes** — ہاں، گینڈالف (Gendarme) فرضی طور پر چارمد (Charmode) کے بارباس (Barbar) کے لیے ایک زبردست دشمن ہے۔ یہ ایک گیم  | **no** — ہاں، گینڈالف فرضی طور پر چارمد کے بارباس کے لیے ایک زبردست دشمن ہوسکتا ہے۔ اس کی وجہ یہ ہے کہ گینڈالف ایک بہت  |
| 4 | `27368c21e50b6af694ab` | no | کیا ایک اخلاقیات کے پروفیسر سیزان پر کلاس پڑھائیں گے؟ | Would an ethics professor teach a class on Cezanne? | Will an ethics professor teach a class on Suzanne? (0.00) | Ethics is a branch of philosophy that involves systematizing, defending, and recommending concepts of right and wrong conduct. | A professor is an academic rank at universities and other post-secondary education and research institutions in most countries. | **yes** — ہاں، ایک اخلاقیات کے پروفیسر کو کلاس پڑھانے کا کام کر سکتا ہے۔ اس کی صلاحیت، تعلیم، تربیت اور تجربے کے برمحل ہ | **no** — اس سوال کا جواب یہ ہے: **نہیں، ایک اخلاقیات کے پروفیسر Suzanne نہیں پڑھائیں گے۔** اس کی وضاحت یہ ہے کہ: - "Suz |
| 5 | `3486b6b5c69878bb088d` | no | کیا ایریزونا میں کیلیفورنیا کے مقابلے میں زیادہ گوبھی اگتی ہے؟ | Do more Cauliflower grow in Arizona than California? | Does Arizona grow more cabbage than California? (0.67) | Arizona is a state in the southwestern United States. | California is a state on the West Coast of the United States. | **yes** — اس سوال کا جواب یہ ہے کہ **کیلیفورنیا** میں **گوبھی** (Cabbage) کی فصل کی پیداوار **ایریزونا** کے مقابلے میں * | **no** — اس سوال کا جواب یہ ہے: **نہیں، ایریزونا میں کیلیفورنیا کے مقابلے میں زیادہ گوبھی اگتی نہیں**۔ **دلیل:** - کیلی |
| 6 | `3503e8d41364eb0b33bc` | no | کیا ایک ذمہ دار بارٹینڈر میلی بوبی براؤن کے لیے مشروب بنائے گا؟ | Would a responsible bartender make a drink for Millie Bobby Brown? | Would a responsible bartender make a drink for Millie Bobby Brown? (1.00) | Millie Bobby Brown is a British actress and model. | Millie Bobby Brown was born on February 19, 2004. | **None (unparsed)** — اس سوال کا جواب یہ ہے کہ **ایک ذمہ دار بارٹینڈر میلی بوبی براؤن کے لیے مشروب بنائے گا**۔ کیونکہ میلی بوبی براؤ | **no** — نہیں، ایک ذمہ دار بارٹینڈر میلی بوبی براؤن کے لیے مشروب نہیں بنائے گا۔ اس کی وجہ یہ ہے کہ میلی بوبی براؤن 2004 |
| 7 | `3b38e00d8d1da26f7f02` | no | کیا ماریو تیز دوڑنے کے لیے مشروم استعمال کرتا ہے؟ | Does Mario use mushrooms to run faster? | Does Mario use a mushroom to run fast? (1.00) | Mario is a fictional character in the Super Mario video game series created by Nintendo. | In the Super Mario series, Mario can use power-ups to gain new abilities. | **yes** — ہاں، ماریو تیز دوڑنے کے لیے مشروم استعمال کرتا ہے۔ یہ ایک مشہور ویڈیو گیم "سوپر ماریو" میں دیکھا جاتا ہے جہاں  | **no** — نہیں، ماریو تیز دوڑنے کے لیے مشروم استعمال نہیں کرتا۔ Super Mushroom کا استعمال ماریو کو بڑا ہو جانے اور ایک ا |
| 8 | `3d7050c0e9bfbeef27c1` | yes | کیا میلانیا ٹرمپ کا پیشہ اولگا کیوریلنکو کے پیشے کے برابر تھا؟ | Did Melania Trump have same profession as Olga Kurylenko? | Was Melania Trump's profession the same as Olga Kurylenko's profession? (1.00) | Melania Trump is a former model and the wife of Donald Trump, the 45th President of the United States. | Melania Trump began her modeling career at a young age and worked with several prominent modeling agencies. | **no** — اس سوال کا جواب یہ ہے کہ **میلانیا ٹرمپ** اور **اولگا کیوریلنکو** کے پیشے مختلف ہیں، اور ان کے کام کے حوالے سے | **yes** — جی ہاں، میلانیا ٹرمپ اور اولگا کیوریلنکو دونوں نے ماڈلنگ کے شعبے میں کام کیا ہے، اور ان کا پیشہ ماڈلنگ تھا۔ اس |
| 9 | `3f5af97fdff20028347b` | no | کیا فرانسیسی انقلاب ٹیلیویژن پر دکھایا گیا تھا؟ | Was the French Revolution televised? | Was the French Revolution shown on television? (1.00) | The French Revolution was a period of radical social and political upheaval in France that lasted from 1789 to 1799. | Television is a telecommunication medium used for transmitting moving images in monochrome or in color, and in two or three dimensions. | **yes** — ہاں، فرانسیسی انقلاب کے مختلف پہلوؤں کو ٹیلیویژن پر دکھایا گیا ہے۔ ٹیلیویژن کے ذریعے تاریخی واقعات کو دکھانے ک | **no** — نہیں، فرانسیسی انقلاب ٹیلیویژن پر دکھایا گیا تھا۔ اس کی وجہ یہ ہے کہ فرانسیسی انقلاب 1789 سے 1799 تک ہوا، جبکہ |
| 10 | `4760fe3d0c80beb30edd` | no | اگر کوئی شخص ویگن ہو تو کیا وہ شہد کھائے گا؟ | If someone is a vegan, would they eat honey? | If a person is vegan, will they eat honey? (-) | Veganism is a lifestyle and dietary choice that seeks to exclude, as far as possible, all forms of exploitation of animals for food, clothing, and other purposes. | Vegans do not consume or use products that come from animals, including eggs, dairy products, and even gelatin. | **yes** — ہاں، ویگن ہونے کا مطلب یہ نہیں کہ کوئی شخص شہد کھانے سے گریز کرے۔ ویگن ڈائیٹ میں جانوروں کی پیداوار سے حاصل ہو | **no** — اس سوال کا جواب یہ ہے کہ **ویگن شخص کے لیے شہد کھانے کا فیصلہ ان کی ذاتی ایمانداری اور ویگنزم کی تعریف پر منحص |
| 11 | `4ec8cd118be8a53ec516` | no | کیا الیگزینڈر ہیملٹن کو کوالاز کے بارے میں معلوم ہوتا؟ | Would Alexander Hamilton have known about koalas? | Would Alexander Hamilton have known about Koalas? (1.00) | Alexander Hamilton was a founding father of the United States and lived from 1755 to 1804. | Koalas are native to Australia. | **yes** — ہاں، الیگزینڈر ہیملٹن کو کوالاز کے بارے میں معلوم ہوتا تھا۔ ہیملٹن ایک امریکی سیاست دان، قانونی ماہر، اور امری | **no** — نہیں، الیگزینڈر ہیملٹن کو کوالاز کے بارے میں معلوم نہیں ہوتا۔ اس کی وجہ یہ ہے کہ ہیملٹن کی زندگی (1755–1804) ک |
| 12 | `502d4dae6f08e73a5569` | yes | کیا ہوم ڈیپو سلیب پر چڑھانے کے سامان کے لیے ایک جامع دکان ہے؟ | Is Home Depot a one stop shop for crucifixion supplies? | Is Home Depot a one-stop shop for slab lifting equipment? (1.00) | The Home Depot is an American multinational home improvement retail corporation. | The Home Depot operates a chain of big-box stores offering a wide range of products and services for home improvement. | **no** — ہاں، ہوم ڈیپو (Home Depot) چڑھانے کے سامان کے لیے ایک جامع دکان ہے۔ یہ امریکا کی سب سے بڑی ہوم ڈیپو چینل ہے جو | **yes** — ہاں، ہوم ڈیپو سلیب لیولنگ یا سلیب پر چڑھانے کے سامان کے لیے ایک جامع دکان ہے۔ ہوم ڈیپو کے پاس سیمنٹ، ماسٹریکس، |
| 13 | `5e2d9519c7c521411817` | yes | کیا ٹی ایم این ٹی رنگ بھرنے کی کتاب میں پیزا ہوگا؟ | Would a TMNT coloring book have pizza in it? | Will there be pizza in the TMNT coloring book? (1.00) | Teenage Mutant Ninja Turtles is a popular American media franchise created by Kevin Eastman and Peter Laird. | The Teenage Mutant Ninja Turtles are known for their love of pizza. | **None (unparsed)** — اس سوال کا جواب یہ ہے کہ **ٹی ایم این ٹی (TNT)** ایک **انفیکٹیوز گیس** ہے جو **بلاک بوسٹر** یا **انفیکٹیوز ماد | **yes** — ہاں، ٹی ایم این ٹی (Teenage Mutant Ninja Turtles) کی رنگ بھرنے والی کتاب میں پیزا ہو سکتا ہے۔ یہ اس لیے ممکن ہ |
| 14 | `67079beff747978ed4c2` | no | کیا ایم ایف ڈوم فینٹاسٹک فور کا ولن ہے؟ | Is MF Doom a Fantastic Four villain? | Is MF Doom a villain of the Fantastic Four? (1.00) | MF DOOM was a British-American rapper and record producer. | MF DOOM was known for his comic book-inspired lyrics and aliases. | **yes** — ہاں، **ایم ایف ڈوم** (M.F. Doom) یا **دومن** (Doom) **فینٹاسٹک فور** (Fantastic Four) کا ایک اہم **ولن** ہے۔ * | **no** — نہیں، ایم ایف ڈوم (MF DOOM) فینٹاسٹک فور کا ولن نہیں ہے۔ MF DOOM ایک ریپر اور ریکارڈ پروڈیوسر تھا جو اپنی کامی |
| 15 | `690a36d5fed5b17969aa` | no | کیا جیک ڈیمپسی نے موجودہ WBC ہیوی ویٹ چیمپئن سے مقابلہ کیا تھا؟ | Did Jack Dempsey fight the current WBC heavyweight champion? | Did Jack Dempsey fight the current WBC heavyweight champion? (1.00) | Jack Dempsey was an American professional boxer who competed from 1914 to 1927. | Jack Dempsey held the World Heavyweight Championship from 1919 to 1926. | **yes** — جی ہاں، جیک ڈیمپسی (Jake Dempsey) نے موجودہ WBC ہیوی ویٹ چیمپئن سے مقابلہ کیا ہے۔ اگر آپ کے سوال میں "موجودہ"  | **no** — نہیں، جیک ڈیمپسی نے موجودہ WBC ہیوی ویٹ چیمپئن سے مقابلہ نہیں کیا۔ اس کی دو بنیادی وجوہات ہیں: 1. **زمانی فرق* |

### 8.3 TR_L1 vs A_r2: the 31 LOSS rows (R3 reading, R3 facts)

| # | qid | gold | Urdu question | gold question_en | R3 reading (row-M1) | fact 1 | fact 2 | A_r2 | TR_L1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `01c27885cede6741bc53` | yes | کیا سپرنیچرل نے 2001 میں دی سی ڈبلیو پر شروع ہونے والے شو کے سیزنز کا ریکارڈ توڑا؟ | Did Supernatural break 2001 CW debuting shows seasons record? | Did Supernatural break the record of the number of seasons of the show that started on The CW in 2001? (1.00) | Supernatural is an American fantasy horror television series that aired from 2005 to 2020. | The CW is an American English-language television network that was formed in 2006 through the merger of UPN and The WB. | **yes** — ہاں، سپرنیچرل (Supernatural) نے 2001 میں دی سی ڈبلیو (The CW) پر شروع ہونے والے شو کے سیزنز کا ریکارڈ توڑا۔ دی | **no** — نہیں، **Supernatural** نے 2001 میں The CW (یا اس کے پیشِن) پر شروع ہونے والے شو کے سیزنز کا ریکارڈ توڑا نہیں۔  |
| 2 | `06724dc213e0dae715f5` | yes | کیا ایک ایمیزون شیئر کبھی بیس سال کی نیٹ فلکس سبسکرپشن خرید سکتا ہے؟ | Could one Amazon share ever buy twenty year Netflix subscription? | Can an Amazon share ever buy a twenty-year Netflix subscription? (1.00) | Amazon is an American multinational technology company. | Netflix is an American media-services provider. | **yes** — اس سوال کا جواب یہ ہے: **ہاں، ایک ایمیزون شیئر کبھی بیس سال کی نیٹ فلکس سبسکرپشن خرید سکتا ہے۔** لیکن یہ ایک م | **no** — نہیں، ایک ایمیزون شیئر کبھی بیس سال کی نیٹ فلکس سبسکرپشن خرید نہیں سکتا۔ اس کی وجہ یہ ہے کہ: - **ایک شیئر** صر |
| 3 | `06f7878425a995c2a633` | no | کیا اسپاگیٹی پکنے کے بعد چکنا ہونا چاہیے؟ | Should spaghetti be slick when cooked? | Should spaghetti be greasy after cooking? (-) | Spaghetti is a type of long, thin, cylindrical pasta made from wheat flour and water. | Cooking is the process of preparing food for consumption by applying heat. | **no** — ہاں، اسپاگیٹی پکنے کے بعد چکنا ہونا چاہیے۔ یہ اس لیے ہے کہ چکنا اسپاگیٹی کو گوشت یا سبزیوں کے ساتھ مزید چکنے ک | **yes** — اس سوال کا جواب یہ ہے: **نہیں، اسپاگیٹی پکنے کے بعد چکنا ہونا چاہیے نہیں**۔ حقیقت یہ ہے کہ اسپاگیٹی کو پکنے کے |
| 4 | `15885efe6f91724c16d9` | yes | کیا آوڈی آر ایٹ وی-ٹین پلس کے لیے صوتی رکاوٹ توڑنا ممکن نہیں؟ | Is sound barrier too much for Audi R8 V-10 Plus to break? | Is it not possible to break the sound barrier for the Audi R8 V10 Plus? (1.00) | The Audi R8 is a mid-engined, all-wheel drive supercar manufactured by Audi. | The Audi R8 V10 Plus is a high-performance variant of the Audi R8. | **yes** — ہاں، **آوڈی آر ایٹ وی-ٹین پلس (Audi A8 L TFSI)** کے لیے **صوتی رکاوٹ توڑنا (sound deadening)** ممکن ہے۔ اس کے  | **no** — ہاں، آوڈی آر8 V10 Plus کے لیے صوتی رکاوٹ (ساؤنڈ بیریئر) توڑنا ممکن نہیں۔ اس کی وجہ یہ ہے کہ آوڈی R8 V10 Plus ک |
| 5 | `17b8fa74f450c4d0e56b` | yes | کیا ڈزنی کی ایلس ان ونڈر لینڈ میں کسی تہوار کی خوشی منائی جاتی ہے؟ | Does Disney's Alice in Wonderland involve the celebration of a holiday? | Is any festival celebrated in Disney's Alice in Wonderland? (1.00) | Disney's Alice in Wonderland is a 1951 American animated fantasy film produced by Walt Disney Productions. | The film is an adaptation of Lewis Carroll's 1865 novel Alice's Adventures in Wonderland. | **yes** — ہاں، ڈزنی کی فلم **"ایلس ان ونڈر لینڈ"** (Alice in Wonderland) میں **تہوار کی خوشی** منائی جاتی ہے۔ یہ تہوار * | **no** — اس سوال کا جواب یہ ہے کہ **ڈزنی کی "ایلس ان ونڈر لینڈ" (1951) میں کسی تہوار کی خوشی منائی نہیں جاتی**۔ فلم میں |
| 6 | `265dd54c248f8b048851` | no | کیا ریڈ گلوبو کے اینکرز چینی زبان بولتے ہیں؟ | Do the anchors on Rede Globo speak Chinese? | Do Red Globe's anchors speak Chinese? (0.33) | Red Globe is a Chinese state-run television network. | Red Globe is also known as China Global Television Network (CGTN). | **no** — ریڈ گلوبو (Red Globo) ایک چینی ٹی وی چینل ہے جو چین کے چینی گورنمنٹ کے زیر انتظام ہے۔ اس چینل کے اینکرز یا نیو | **yes** — ہاں، ریڈ گلوب (Red Globe) یا چائنا گلوبل ٹی وی نیٹورک (CGTN) کے اینکرز عام طور پر چینی زبان بولتے ہیں۔ چونکہ C |
| 7 | `2c9f3c00412a4018380f` | yes | کیا فرنیچر بال سے بنایا جا سکتا ہے؟ | Can furniture be made of hair? | Can furniture be made from balloons? (-) | Furniture is the movable objects intended to support various human activities such as seating, sleeping, cooking, and storage. | Furniture can be made from a wide range of materials, including wood, metal, and plastic. | **yes** — ہاں، فرنیچر بال سے بھی بنایا جا سکتا ہے۔ بال کو چمکدار اور گرم کرکے، اسے چمکدار چمچ یا چمکدار چمچ کی مدد سے چم | **no** — جی ہاں، فرنیچر بال سے بنایا جا سکتا ہے، لیکن یہ ایک بہت غیر معمولی اور کم احتمالی تجربہ ہوگا۔ چونکہ بالونز (با |
| 8 | `315575d5cbdd1d154744` | yes | کیا کرٹ کوبائن کو پروجیکٹ سیمی کولن سے فائدہ ہوتا؟ | Would Kurt Cobain have benefited from Project Semicolon? | Would Kurt Cobain have benefited from Project Semi Colin? (0.75) | Kurt Cobain was an American musician and artist, best known as the lead singer and guitarist of the grunge band Nirvana. | Kurt Cobain struggled with depression and anxiety throughout his life. | **yes** — ہاں، کرٹ کوبائن (Curt Cobain) کو پروجیکٹ سیمی کولن (Project Semmel) سے فائدہ ہوتا تھا۔ پروجیکٹ سیمی کولن ایک 1 | **no** — کرٹ کوبائن کو پروجیکٹ سیمی کولن سے فائدہ نہیں ہوتا ہوگا۔ اس کی وجہ یہ ہے کہ کرٹ کوبائن 1994 میں دنیا سے رخصت ہ |
| 9 | `3157a0f428531ebed7b9` | yes | کیا کسی مائنر لیگ بیس بال ٹیم کا نام بلی نما جانوروں کے نام پر ہے؟ | Are any minor league baseball teams named after felines? | Is any minor league baseball team named after felines? (-) | Minor league baseball is a hierarchy of professional baseball leagues in North America. | Minor league baseball teams often have unique names that reflect their local culture or geography. | **yes** — ہاں، کچھ مائنر لیگ بیس بال ٹیموں کے نام بلی نما جانوروں پر ہیں۔ مثلاً: - **"Cats"** (کیٹس): کچھ ٹیموں کا نام " | **no** — ہاں، کچھ مائنر لیگ بیس بال ٹیمیں بلی نما جانوروں کے نام پر ہیں۔ مثال کے طور پر: - **New Orleans Baby Cakes** ( |
| 10 | `3570d57f16a40488ff42` | yes | کیا سلیکون کے سوٹ جودو کو مشکل بناتے ہیں؟ | Do silicone suits make judo difficult? | Do silicon suits make judo difficult? (-) | Silicon is a chemical element with the symbol Si and atomic number 14. | Silicon is used in various forms, including silicones, which are synthetic polymers. | **yes** — ہاں، سلیکون کے سوٹ جو دو کو مشکل بناتے ہیں۔ یہ بیان اکثر ایسی کہانیوں یا محاوروں میں استعمال ہوتا ہے جہاں "سلی | **no** — نہیں، سلیکون کے سوٹ جودو کو مشکل نہیں بناتے۔ سلیکون ایک ریمیٹ کیمیکل عنصر ہے جو جودو کے سوٹ (judogi) کی تیاری  |
| 11 | `38ef97eb7cdd4200fd00` | yes | کیا سنہری بالوں والی سبز آنکھوں والی سارہ پیکسٹن کو لاطینی سمجھا جاتا ہے؟ | Is blonde hair green eyed Sara Paxton considered a Latino? | Is Sarah Paxton with golden hair and green eyes considered Latin? (0.67) | Sarah Paxton is an American actress. | Latin peoples refers to the ethnic groups that speak Romance languages, which descended from Latin. | **yes** — ہاں، سارہ پیکسٹن (Sarah Jessica Parker) کو عام طور پر "لاطینی" سمجھا جاتا ہے۔ وہ ایک امریکی اداکارہ ہیں جو فلم | **no** — نہیں، سنہری بالوں والی سبز آنکھوں والی Sarah Paxton کو لاطینی (Latin) سمجھا نہیں جاتا۔ اس کی وجہ یہ ہے کہ: 1.  |
| 12 | `4125c5f4c7a536916608` | yes | کیا خُرده فروشی کا ملازم خُرده فروش کے سی ای او کی تنخواہ پر حسد کرے گا؟ | Would a retail associate envy the retailer's CEO's pay? | Will a retail employee be jealous of the retail CEO's salary? (1.00) | Retail is the sale of goods and services from individuals or businesses to the end-user. | An employee is an individual who is hired to provide labor to a company in exchange for compensation. | **yes** — اس سوال کا جواب یہ ہے کہ: **ہاں، خُرده فروشی کا ملازم خُرده فروش کے سی ای او کی تنخواہ پر حسد کر سکتا ہے۔** اس | **no** — اس سوال کا جواب یہ ہے کہ **خُرده فروشی کا ملازم خُرده فروش کے سی ای او کی تنخواہ پر حسد کر سکتا ہے**، لیکن یہ  |
| 13 | `48da75d87c66754ccc2e` | yes | کیا پرل ہاربر پر حملے کے دوران امریکی ہاؤس اسپیکر کے طور پر خدمات انجام دینے والی واحد خاتون زندہ تھیں؟ | Was only woman to serve as U.S. Speaker of the House alive during the attack on Pearl Harbor? | Was the only woman serving as US House Speaker during the Pearl Harbor attack still alive? (1.00) | The attack on Pearl Harbor occurred on December 7, 1941. | Sam Rayburn was the Speaker of the United States House of Representatives from 1940 to 1947. | **yes** — اس سوال کا جواب یہ ہے: **ہاں، ہاں**۔ **نینا ہیس** (Nancy Pelosi) امریکی ہاؤس آف ریپریزنٹیٹیو (ہاؤس اسپیکر) کے  | **no** — نہیں، پرل ہاربر پر حملے کے دوران (7 دسمبر 1941) امریکی ہاؤس آف ریپریزنٹیٹیوZ کے اسپیکر کے طور پر کوئی خاتون خد |
| 14 | `598be3dcddbd7775a827` | no | کیا فرضی طور پر باب مارلے کے بچے کبلائی خان کے بچوں کے خلاف رسی کشی جیت سکتے ہیں؟ | Could Bob Marley's children hypothetically win tug of war against Kublai Khan's children? | Can Bob Marley's hypothetical children beat Genghis Khan's children in a tug of war? (0.75) | Bob Marley was a Jamaican singer, songwriter, and musician. | Genghis Khan was the founder and emperor of the Mongol Empire. | **no** — اس سوال کا جواب دیا جاتا ہے کہ **باب مارلے کے بچے کبلائی خان کے بچوں کے خلاف رسی کشی میں جیت نہیں سکتے**۔ اس ک | **yes** — اس سوال کا جواب دیا جاسکتا ہے کہ **فرضی طور پر**، باب مارلے کے بچے کبلائی خان کے بچوں کے خلاف **رسی کشی میں جی |
| 15 | `660bfdc17b47f42facac` | yes | کیا مووس کو ارینڈیل کے سلطنت کے قریب کام کے لیے استعمال کیا جاتا ہے؟ | Are moose used for work near the kingdom of Arendelle? | Is Mouse used for work near the kingdom of Arendelle? (1.00) | Arendelle is a fictional kingdom and the main setting of the Disney movie Frozen. | Arendelle is not a real-world location. | **yes** — اس سوال کا جواب یہ ہے: **ہاں، مووس کو ارینڈیل کے سلطنت کے قریب کام کے لیے استعمال کیا جاتا ہے۔** یہ بیان ایک ا | **no** — اس سوال کا جواب یہ ہے: **نہیں**۔ مووس (کمپیوٹر ماؤس) کو ارینڈیل کے سلطنت کے قریب کام کے لیے استعمال نہیں کیا ج |
| 16 | `703db4629ff46d82a205` | no | کیا کین اس سال کی پرائمریز میں جو بائیڈن کو چیلنج کر سکتا ہے؟ | Can Kane challenge Joe Biden in this year's primaries? | Can Cain challenge Biden in this year's primaries? (0.33) | Cain is likely referring to Herman Cain, an American businessman and politician. | Herman Cain was a candidate in the 2012 Republican presidential primaries. | **no** — اس سال (2024) کی پرائمریز میں جو بائیڈن کو چیلنج کر سکتا ہے، اس کے بارے میں اب تک کوئی واضح اور معتبر نام نہیں | **yes** — ہاں، کین (Herman Cain) اس سال (2024) کی ڈیموکریٹک پرائمریز میں جو بائیڈن کو چیلنج کر سکتا ہے۔ یہ ممکن ہے کہ وہ |
| 17 | `7870b1cef39a4f685911` | yes | کیا آدم سینڈلر کو کول سپروس اور ایک سکوبا آدمک کے حوالے سے کوئی بات سمجھ آئے گی؟ | Would Adam Sandler get a reference to Cole Spouse and a scuba man doll? | Will Adam Sandler understand anything about Cold Sore and a Scuba Diver? (0.50) | Adam Sandler is an American actor, comedian, and film producer. | Cold sore, also known as fever blister, is a viral infection caused by the herpes simplex virus. | **yes** — ہاں، آدم سینڈلر کو کول سپروس (CoolSculpting) اور ایک سکوبا آدمک (Scooba) کے حوالے سے کچھ سمجھ آئے گی۔ **کول سپ | **no** — اس سوال کا جواب یہ ہے کہ **ہرگز نہیں**۔ **کیوں؟** کیونکہ: - **آدم سینڈلر** ایک **ایکٹر، کامیڈین اور فلم پروڈیو |
| 18 | `80aa769f55b14c1e4d8d` | yes | کیا سیبل وولورین سے متعلق ہیں؟ | Are sables related to wolverines? | Are Sabre and Wolverine related? (-) | Sabre is a fictional character in the Marvel Comics universe. | Wolverine is a fictional character and one of the most well-known superheroes in the Marvel Comics universe. | **yes** — ہاں، سیبل (Sable) وولورین (Wolverine) سے متعلق ہے۔ سیبل ایک خاص قسم کی چوہا ہے جو یوکرین اور روس کے شمالی علاق | **no** — نہیں، سیبل (Sabre) وولورین (Wolverine) سے متعلق نہیں ہے۔ اس کے باوجود دونوں چارٹر ہیں اور مارول کامیکس کی دنیا |
| 19 | `9247b5afb2eef4c9f95b` | yes | کیا ایک مائنوٹور فرضی طور پر فٹ بال کھیلتے ہوئے ٹیبیا کو زخمی کر سکتا ہے؟ | Can a minotaur hypothetically injure a tibia playing football? | Can a minotaur injure the tibia while playing football hypothetically? (-) | In Greek mythology, the Minotaur is a creature with the body of a man and the head and horns of a bull. | The tibia, also known as the shinbone, is the larger and stronger of the two bones in the leg below the knee. | **yes** — ہاں، ایک مائنوٹور (minor) فرضی طور پر فٹ بال کھیلتے ہوئے ٹیبیا (tibia) کو زخمی کر سکتا ہے۔ ٹیبیا ہڈی کا ایک حص | **no** — ہاں، ایک مائنوٹور فرضی طور پر فٹ بال کھیلتے ہوئے ٹیبیا کو زخمی کر سکتا ہے۔ اس کی وضاحت یہ ہے کہ مائنوٹور ایک ب |
| 20 | `9384dfc0aabc73026925` | yes | کیا جسمانی اور علامتی دل نمایاں طور پر مختلف نظر آتے ہیں؟ | Do anatomical and symbolic hearts look remarkably different? | Do physical and symbolic hearts appear distinctly different? (-) | The heart is a muscular organ in humans and other animals that pumps blood through the circulatory system. | The heart is often symbolically represented as a stylized drawing, typically red and heart-shaped. | **yes** — ہاں، جسمانی اور علامتی دل (یا جذبات) نمایاں طور پر مختلف نظر آتے ہیں۔ یہ دونوں ایک شخص کے مزاج، تجربے اور حالا | **no** — ہاں، جسمانی اور علامتی دل نمایاں طور پر مختلف نظر آتے ہیں۔ یہ دونوں ایک دوسرے سے مکمل طور پر مختلف ہیں، کیونکہ |
| 21 | `987260ffef60cb2c5439` | yes | کیا پرل ہاربر شارک دیوی کا خیالی گھر ہے؟ | Is Pearl Harbor the mythical home of a shark goddess? | Is Pearl Harbor the fictional home of Shark Lady? (1.00) | Pearl Harbor is a lagoon harbor on the island of Oahu, Hawaii, United States. | Pearl Harbor is a real geographic location and a U.S. naval base. | **yes** — ہاں، پرل ہاربر شارک دیوی کا خیالی گھر ہے۔ یہ ایک خیالی یا فیکشنل گھر ہے جو "شارک دیوی" (Shark Girl) کے کردار ک | **no** — نہیں، پرل ہاربر شارک دیوی کا خیالی گھر نہیں ہے۔ حقیقت یہ ہے کہ پرل ہاربر (Pearl Harbor) ہوائی ہائی کے جزیرہ او |
| 22 | `9bb59488a227a5f5cd62` | no | کیا آپ جی بی پر گیم آف تھرونز کے ہر ایچ ڈی قسط کو محفوظ کر سکتے ہیں؟ | Can you save every HD episode of Game of Thrones on Samsung Galaxy A10e? | Can you save each HD episode of Game of Thrones on GB? (0.50) | Game of Thrones is an American fantasy drama television series that aired from 2011 to 2019. | Each episode of Game of Thrones has a typical runtime of around 50-80 minutes. | **no** — ہاں، آپ جیبی (Jio) پر گیم آف تھرونز (Game of Thrones) کے ہر ایچ ڈی (HD) قسط کو محفوظ کر سکتے ہیں۔ اس کے لیے آپ | **yes** — ہاں، آپ **Game of Thrones** کے ہر ایچ ڈی (HD) قسط کو **جی بی (GB)** کے ذریعے محفوظ کر سکتے ہیں۔ چونکہ ہر قسط ک |
| 23 | `b8df6ac181c4c5b280f9` | no | کیا بوتلنوز ڈولفن نظریاتی طور پر انسانی خواتین سے زیادہ نسل پیدا کر سکتے ہیں؟ | Can bottlenose dolphins hypothetically outbreed human women? | Can bottlenose dolphins theoretically produce more offspring than human women? (-) | Bottlenose dolphins are marine mammals that give birth to live young. | The average gestation period for bottlenose dolphins is approximately 12-14 months. | **no** — اس سوال کا جواب یہ ہے کہ **بوتلنوز ڈولفن (Bottlenose dolphins)** نظریاتی طور پر انسانی خواتین سے زیادہ نسل پید | **None (unparsed)** — اس سوال کا جواب یہ ہے کہ **بوتلنوز ڈولفن نظریاتی طور پر انسانی خواتین سے زیادہ نسل پیدا کر سکتے ہیں**۔ ### تفص |
| 24 | `c921bdade889def26cf4` | yes | کیا آئسوپروپائل الکحل میں نمک کی حل پذیری کم ہے؟ | In isopropyl alcohol, is the solubility of salt low? | Is the solubility of salt lower in isopropyl alcohol? (-) | Solubility is a measure of the maximum amount of a substance that can dissolve in a given amount of solvent at a given temperature and pressure. | Salt is the common name for the chemical compound sodium chloride (NaCl). | **yes** — ہاں، **اسوپروپائل الکحل** (Isopropyl alcohol) میں **نمک (NaCl)** کی **حل پذیری (solubility)** **کم** ہوتی ہے۔  | **no** — ہاں، **آئسوپروپائل الکحل (Isopropyl alcohol)** میں **نمک (سائیل یا سوڈیم کلورائیڈ - NaCl)** کی **حل پذیری کم ہ |
| 25 | `d47120efd0f09badd848` | yes | کیا مینٹھول کرسمس سے منسلک ہے؟ | Is menthol associated with Christmas? | Is menthol associated with Christmas? (1.00) | Menthol is a waxy, crystalline substance, clear or white in color, which is solid at room temperature. | Menthol is used in a variety of products, including cough drops, candies, and topical pain relief creams. | **yes** — ہاں، مینٹھول (Menthol) کرسمس سے منسلک ہے۔ یہ تعلق اس لیے ہے کیونکہ مینٹھول کا استعمال کرسمس کے دوران ہونے والی | **no** — ہاں، مینٹھول کرسمس سے منسلک ہے، لیکن اس کا تعلق کرسمس کے مذہبی پہلو سے نہیں، بلکہ ثقافتی اور مصنوعات کے استعما |
| 26 | `d808a0c72dbe605309ab` | no | کیا فیلکس پوٹون کا ڈاج بال ٹیم میں کوئی مقام ہے؟ | Does Felix Potvin have a position on a dodgeball team? | Does Felix Pauton have a place in the Dodgeball team? (0.50) | Felix Pauton is a fictional character in the 2004 film Dodgeball: A True Underdog Story. | Dodgeball is a team sport in which players throw balls at each other. | **no** — فیلکس پوٹون (Felix Potton) کا ڈاج بال (Dog Ball) ٹیم میں کوئی مقام نہیں ہے۔ یہ ایک جھوٹا یا غلط نام ہے، اور اس | **yes** — ہاں، فیلکس پوٹون کے ڈodgeبال ٹیم میں ایک مقام ہے۔ وہ فلم **"Dodgeball: A True Underdog Story"** (2004) میں **" |
| 27 | `d8fd81e475df8c2f6972` | no | کیا کسواری کرین کے مقابلے میں زیادہ محفوظ پالتو ہے؟ | Is a Cassowary safer pet than a crane? | Is the Casowary more safe than the Crane as a pet? (0.00) | Cassowaries are large, flightless birds native to the tropical regions of Asia and Australia. | Cassowaries are known for their aggressive behavior, particularly when they feel threatened or when they are protecting their young. | **no** — ہاں، **کسواری** (کتے) کے مقابلے میں **کرین** (چمگادڑ) عام طور پر **زیادہ محفوظ** پالتو ہے۔ یہ کہنا کہ کرین زیا | **yes** — کسواری (Cassowary) کرین (Cranes) کے مقابلے میں **کم محفوظ** پالتو ہے۔ یہاں تک کہ حقائق سے واضح ہے کہ: - **کسوا |
| 28 | `de62d38623fde2c2285a` | yes | کیا ایک ہفتہ ہر قسط دیکھنے کے لیے کافی وقت ہوگا؟ | Would a week be enough time to watch every episode of Ugly Betty? | Will one week be enough time to watch each episode? (0.00) | A week is a unit of time, equal to 7 days. | The length of an episode can vary greatly depending on the type of show and the country of origin. | **yes** — ہاں، ایک ہفتہ ہر قسط دیکھنے کے لیے کافی وقت ہو سکتا ہے، لیکن یہ اس بات پر بھی منحصر ہے کہ: 1. **قسط کی لمبائی* | **no** — اس سوال کا جواب یہ ہے کہ **ایک ہفتہ ہر قسط دیکھنے کے لیے کافی یا کم ہو سکتا ہے**، یہ اس پر منحصر ہے کہ: 1. **ق |
| 29 | `e6d3973ed3feb8a42928` | yes | کیا افریقہائزڈ مکھیاں کثیر الثقافتی سمجھی جا سکتی ہیں؟ | Can Africanized bees be considered multicultural? | Can Afrikanized bees be considered multicultural? (0.00) | Africanized bees are a hybrid species of bee created by cross-breeding the European honey bee with the African honey bee. | Africanized bees are known for their aggressive behavior when threatened or provoked. | **yes** — ہاں، افریقہائزڈ مکھیاں (Africanized honey bees) کثیر الثقافتی (polylectic) سمجھی جا سکتی ہیں۔ "کثیر الثقافتی"  | **no** — اس سوال کا جواب یہ ہے کہ ** africized bees (افریقہائزڈ مکھیاں) کثیر الثقافتی (multicultural) نہیں سمجھی جا سکت |
| 30 | `f43533225534420816d6` | yes | کیا آپ سولجر فیلڈ میں دیوقامت اسکویڈ رکھ سکتے ہیں؟ | Can you house a giant squid at Soldier Field? | Can you keep a giant squid in Soldier Field? (1.00) | Giant Squid are deep-sea cephalopods that can grow up to 13 meters in length. | Giant Squid require a large amount of space and specific water conditions to survive. | **yes** — ہاں، آپ سولجر فیلڈ (Soldier Field) میں دیوقامت اسکویڈ (Dee Whyte Squad) رکھ سکتے ہیں۔ سولجر فیلڈ چکاگو، امریکہ | **no** — نہیں، آپ سولجر فیلڈ میں دیوقامت اسکویڈ (Giant Squid) رکھ نہیں سکتے۔ اس کی کچھ اہم وجوہات ہیں: 1. **سولجر فیلڈ  |
| 31 | `f8e6087cd52c925fedad` | yes | کیا قدیم ٹونی بینیٹ 2020 میں بچہ پیدا کر سکتے تھے؟ | Could ancient Tony Bennett have a baby in 2020? | Could ancient Tony Bennett have given birth to a child in 2020? (1.00) | Tony Bennett is a male singer of popular music. | Tony Bennett was born on August 3, 1926. | **yes** — نہیں، قدیم ٹونی بینیٹ (Tony Bennett) 2020 میں بچہ پیدا کر سکتے تھے۔ ٹونی بینیٹ 1926 کی پیدائش کے ساتھ 2023 میں | **no** — نہیں، قدیم ٹونی بینیٹ 2020 میں بچہ پیدا کر سکتے تھے کیونکہ ٹونی بینیٹ مرد ہیں اور مردوں کو بیولوجیکل طور پر بچ |

Of the 31 TR_L1 losses, 23 are gold-yes rows turned to "no" and 8 are gold-no rows turned to "yes" (§5 table). 16 of the 22 TR losses are also TR_L1 losses (`01c27885…`, `15885efe…`, `17b8fa74…`, `3570d57f…`, `38ef97eb…`, `48da75d8…`, `660bfdc1…`, `703db462…`, `80aa769f…`, `987260ff…`, `9bb59488…`, `b8df6ac1…`, `c921bdad…`, `d47120ef…`, `de62d386…`, `e6d3973e…`); the 6 TR-only losses are `2e8e9e51…`, `5897ec22…`, `a8bbf2e8…`, `b3443bf3…`, `e12fef05…`, `ff381173…`.

---

## 9. Files written and open items

Written: `docs/handoff/D11_FULL_READOUT.md` (this file), `docs/D11_DECISION.md`, `outputs/efbpt/d11r/_scores/scores_dev200.json` (scorer output, gitignored). Not written: `docs/D11_GO_EVAL458.md` (waits for the hand check). No sealed file was modified; no job submitted.

Open items for the hand check: (1) the 22 TR loss rows and 15 win rows above, in particular whether the losses are reading faults, fact faults, or the student's "no" default under facts; (2) the 5 loss rows where the Rule 2 label and the opening word disagree; (3) the eval458 protocol needs TR's reader (R1) eval458 reading file, sealed in D11-R and unopened; §8 arms A_r2_eval, TR, TR_wrong_eval, gate +5.0 and p < 0.05, one run.
