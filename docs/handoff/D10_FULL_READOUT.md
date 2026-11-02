# D10 FULL readout: jobs 109581 (teacher) and 109582 (student)

Read-only readout of the completed D10 FULL run (teacher fact injection, freeze `b3a822d8`, Amendments 1 and 2 applied). Sources: `outputs/efbpt/d10/teacher/` (5 sealed files + `T_SEAL.json`), `outputs/efbpt/d10/student/` (7 sealed files + `S_SEAL.json`), `logs/d10_teacher_FULL_109581.log` (170 lines), `logs/d10_student_FULL_109582.log` (1066 lines), `sacct`. I recomputed every number below with Python from the sealed jsonl files. Besides this file, the only other file written is `docs/handoff/D10_LOSS_WIN_ROWS.md` (§6.4). No edits, commits, or sbatch.

Conventions: "wins" = the first-named arm is right and the second is wrong; "losses" = the reverse. p = exact two-sided binomial on the discordant pairs (`min(1, 2·Σ_{i≤min(b,c)} C(n,i)/2^n)`), the freeze's McNemar. Unparsed = wrong. Gold is yes on 87/200 and no on 113/200.

**Headline.** R1 TF − A_r2 = +5.0 pp (65.0% vs 60.0%, wins 38, losses 28, p = 0.268) → **CONDITIONAL**. R2: T-best = T_direct_F (55.5%); TF ≥ T-best → outcome (a). R3: TF_wrong − A_r2 = −5.5 pp, which under Amendment 2 A2.1 is **WRONG-FACTS HARM**, not FORMAT-EFFECT (the script's `FORMAT_EFFECT` label and its §7 override line are superseded by A2.1; the R1 claim stands). TF_noq 44.0% < A_r2 → no LEAK FLAG. R4 teacher entity reading 34.0% on D6-71 (ref 21.0%) → no MECHANISM FLAG. §7 branch: **R1 CONDITIONAL → exactly ONE fix iteration, declared in an amendment, then one FULL re-run.**

---

## 1. Run status

**Last 3 lines of each log:**

```
D10 JOB T COMPLETE. Sealed 5 files.
=== end: 2026年 11月 01日 星期日 13:19:31 CST ===
```
```
D10 JOB S COMPLETE. Sealed 7 files.
=== end: 2026年 11月 01日 星期日 19:37:07 CST ===
```

**T_SEAL.json**: job `T`, mode `FULL`, slurm_job_id `109581`, end_time `2026-11-01 13:19:29`, n_rows 200, t1_parse_ok 199, t2_removed_total 0.
**S_SEAL.json**: job `S`, mode `FULL`, slurm_job_id `109582`, end_time `2026-11-01 19:37:06`; it re-lists the 5 teacher MD5s (identical to T_SEAL).

| item | MD5 in seal / START | recomputed now |
|---|---|---|
| freeze `docs/EFBPT_D10_TEACHER_FREEZE.md` | `b3a822d8199034c7fe58d76cf5cc428f` (both seals, both START files) | match |
| teacher script `eval/error_analysis_tests/efbpt/d10_teacher.py` | `60bca5ba3d60e28e93fac1c643bf6802` (T_SEAL, T_START, S_START.teacher_seal) | match |
| student script `eval/error_analysis_tests/efbpt/d10_student.py` | `21be7cb0e72b669f79113d202a3e856d` (S_SEAL, S_START) | match |
| Urdu instruction `prompts/efbpt/plan_a_instruction_ur.txt` | `f3b58d766fe3ec2573ff4f24761cf0c9` (both START files) | match |
| Amendment 1 `docs/EFBPT_D10_AMENDMENT_1.md` | not in any seal | `ac828661e26d77ca9bf00695ce4b8b93` |
| Amendment 2 `docs/EFBPT_D10_AMENDMENT_2.md` | not in any seal | `c7968a1699b837d48ecf9be6fd7e1605` |
| git head (both START files) | `b24113fffa1c957000c00e240155db0a3653eb7e`, dirty false | = current HEAD `b24113f`, tree clean |

Sealed files (all 12 MD5s, byte counts and line counts recomputed and matching both seals):

| file | MD5 | bytes | lines |
|---|---|---|---|
| `outputs/efbpt/d10/teacher/T_START.json` | `54be61f2ef00d249faa8355335d3f40f` | 7881 | 282 |
| `outputs/efbpt/d10/teacher/t_facts.jsonl` | `b711b0a1c14533a2ba1227000077e3b3` | 656645 | 200 |
| `outputs/efbpt/d10/teacher/t_direct.jsonl` | `bac96cd9a9773b5c5e719e982ecfe499` | 203457 | 200 |
| `outputs/efbpt/d10/teacher/t_direct_f.jsonl` | `40ebfe643e244c5a1a3fff729f23462c` | 239223 | 200 |
| `outputs/efbpt/d10/teacher/derangement.json` | `8f00e2f9e54f27138964a0577a8eff52` | 10516 | 206 |
| `outputs/efbpt/d10/student/S_START.json` | `7a4e5209a2c857a3c21b8372b67d7d94` | 10199 | 395 |
| `outputs/efbpt/d10/student/d10_armA_r2.jsonl` | `7ccec942c8f0eee08156d258baaf806d` | 366128 | 200 |
| `outputs/efbpt/d10/student/d10_armTF.jsonl` | `c54b7553baa5ffa77204bed1dab37250` | 480538 | 200 |
| `outputs/efbpt/d10/student/d10_armTF_wrong.jsonl` | `f509e36c7c32e2caaf6b5f79af6bdd9f` | 481140 | 200 |
| `outputs/efbpt/d10/student/d10_armTF_noq.jsonl` | `113b21b8cc42d5517cd7913e35380315` | 389091 | 200 |
| `outputs/efbpt/d10/student/tf_facts_sample30.jsonl` | `7c7dedefbcd7d0e76f4618bc8ff6e774` | 10439 | 30 |
| `outputs/efbpt/d10/student/d10_summary.json` | `4d671d53ef5cce718a6c25b2eda2b635` | 96913 | 4525 |

The seal files themselves (not self-listed): `T_SEAL.json` MD5 `af305699058f33405195d23c3b418761` (1022 bytes, 37 lines); `S_SEAL.json` MD5 `5e6ac0521ddffc58c8ca013ecdbad46c` (1729 bytes, 51 lines).

**SLURM** (`sacct -j 109581,109582`):

| job | name | State | ExitCode | node | Start | End | Elapsed | Timelimit |
|---|---|---|---|---|---|---|---|---|
| 109581 | d10_teacher_FULL | COMPLETED | 0:0 | L20001 | 2026-11-01T11:39:23 | 2026-11-01T13:19:31 | 01:40:08 | 04:00:00 |
| 109582 | d10_student_FULL | COMPLETED | 0:0 | L20001 | 2026-11-01T13:19:31 | 2026-11-01T19:37:07 | 06:17:36 | 12:00:00 |

Job S started the second Job T ended (A2.3 `--dependency=afterok`). Log start banners: T 11:39:25, S 13:19:32. `T_START.json` start_time 11:40:07 (after preflight); `S_START.json` start_time 13:41:21 (after preflight and the V2 probe).

**Teacher job** (vLLM, Llama-3.1-70B-Instruct-AWQ-INT4, max_num_seqs 1, greedy). Model load 1:24 (9 shards). Final progress line of each pass:

| pass | arm | rows | total s | s/row (log) | s/row (exact) | finish reasons |
|---|---|---|---|---|---|---|
| T1 | T_facts | 200 | 2006.7 | 10.0 | 10.03 | stop 200 |
| T3 | T_direct | 200 | 1753.7 | 8.8 | 8.77 | stop 199, length 1 |
| T3F | T_direct_F | 200 | 2099.3 | 10.5 | 10.50 | stop 198, length 2 |

T1 log line: `parse_ok 199/200 (99.5%)   facts per row min=0 median=6 max=6  empty=1   removed (V5) 0   truncated 0`. Recomputed: 199 rows with exactly 6 facts, 1 row (`ba04b5ebc2edd682c4dd`) with parse_ok false and 0 facts; 0 removed lines; derangement has no fixed point and every TF_wrong row's `facts_source_qid` equals `derangement[qid]`.

**Student job** (HF, Qwen3-14B nf4, batch size 1, greedy, thinking off, max_new 1024). V2 probe: 20 prompts twice, 597.3 s and 590.5 s. Final progress line of each arm:

| arm | rows | total s | s/row (log) | s/row (exact) | layouts | empty-facts rows |
|---|---|---|---|---|---|---|
| A_r2 | 200 | 5821.2 | 29.1 | 29.11 | A 200 | 200 |
| TF | 200 | 5766.2 | 28.8 | 28.83 | B 199, A 1 | 1 |
| TF_wrong | 200 | 5713.0 | 28.6 | 28.57 | B 199, A 1 | 1 |
| TF_noq | 200 | 4043.5 | 20.2 | 20.22 | G 200 | 1 |

The one layout-A row in TF and TF_noq's one empty-facts row are the parse-fail qid `ba04b5ebc2edd682c4dd`; in TF_wrong the layout-A row is `b77170e15c0b1bd72340`, whose donor is `ba04b5ebc2edd682c4dd`. Prompt token lengths (log `[len]`): A_r2 min 54 / median 79 / max 114; TF 63 / 202 / 292; TF_wrong 70 / 200 / 297; TF_noq 44 / 161 / 246.

---

## 2. Validity

**Verbatim from the student log (lines 1020–1028):**

```
V1 unparsed spread A_r2/TF/TF_wrong: {'A_r2': 4.0, 'TF': 0.5, 'TF_wrong': 5.0} -> 4.50 pp -> PASS   (TF_noq unparsed 21.50%, reported)
V2 determinism probe 20/20 identical -> PASS
V3 leak guards: none fatal in either job -> PASS
V4 teacher parse 199/200 = 99.5% -> PASS
V5 verdict-filter removed 0 -> ok
V6 verbatim gold-fact substrings (reported, not a gate): A_r2 0 in 0 qid(s), TF 2 in 2 qid(s), TF_wrong 2 in 2 qid(s), TF_noq 2 in 2 qid(s)
V7 T_SEAL verified (5 files) -> PASS
V8 prompt-content check: 598 facts prompts -> PASS
V9 Latin fraction of injected facts 1.0000
```

Preflight lines (student log 31–46): `[V7] T_SEAL ...: 5 files verified (md5, bytes, lines) -> PASS`; `[leak-guard] A_r2 200 prompts clean (200 question_en, 532 gold facts, 862 evidence ids checked)`; TF / TF_wrong / TF_noq `question_en and evidence ids clean; 2 verbatim gold-fact substring(s) in 2 prompt(s), reported as V6`; `[V8] prompt-content check: 598 facts prompts equal their sealed post-T2 facts -> PASS`; `[V9] Latin fraction of injected teacher facts 1.0000 (87418/87418 letters; D8 GEN 0.9972)`. Teacher log: `[leak-guard] T1/facts 200 prompts clean`, same for T3 and T3F (600 prompts, 0 fatal; `T_START.leak_guard.fatal_hits` 0).

**Recomputed from the sealed files:**

| check | summary value | recomputed | status |
|---|---|---|---|
| V1 unparsed A_r2 / TF / TF_wrong | 4.0 / 0.5 / 5.0 → spread 4.5 pp (limit 5.0) | 8 / 1 / 10 of 200 → 4.5 pp | PASS |
| V1 TF_noq unparsed (reported) | 21.5% | 43/200 | reported |
| V2 probe | 20/20 identical | S_START `V2_determinism_probe` identical 20 | PASS |
| V3 leak guards | teacher 600 prompts, student 800 prompts, 0 fatal | — (from logs/START) | PASS |
| V4 teacher parse | 199/200 = 99.5% (min 190) | 199 `parse_ok` true | PASS |
| V5 verdict filter | 0 removed (max 100) | Σ `n_removed` = 0; all `removed` lists empty | ok |
| V6 gold-fact substrings | TF 2, TF_wrong 2, TF_noq 2, A_r2 0 | see §7 | reported |
| V7 T_SEAL | 5 files | all 5 MD5/bytes/lines match now | PASS |
| V8 prompt content | 598 facts prompts | 199 + 199 + 200 fact-bearing prompts = 598; TF `n_facts_given` = teacher `n_facts` on all 200 | PASS |
| V9 Latin fraction | 1.0000 (87418 letters, 1194 facts) | 1194 facts, 87418 alphabetic chars; the only non-ASCII letter is `é` (5 occurrences), counted as Latin by the script | reported |

Flags in `d10_summary.json`: DETERMINISM false, FORMAT false, TEACHER_ANSWERING false, FORMAT_EFFECT **true** (see §3, superseded by A2.1), LEAK false, MECHANISM false, V1_VOID false.

---

## 3. Readings

**Verbatim from the student log (lines 1033–1050):**

```
  TF       vs A_r2       b=38  c=28   p=0.2678
R1 TF - A_r2 = 5.0 pp, wins 38 losses 28, p = 0.2678116648465871 -> CONDITIONAL
  TF       vs T_direct   b=96  c=20   p=0.0000
  TF       vs T_direct   b=16  c=20   p=0.6177
  TF       vs T_direct_F   b=58  c=39   p=0.0671
R2 T_direct   acc 27.0%  unparsed 59.5%  pred-yes 14.5%  | TF - T_direct = 38.0 pp, wins 96 losses 20, p = 4.248735618228405e-13  [parsed-only: TF 61.73% vs T_direct 66.67%, n=81]
R2 T_direct_F acc 55.5%  unparsed 0.5%  pred-yes 74.5%  | TF - T_direct_F = 9.5 pp, wins 58 losses 39, p = 0.06705179535289094
R2 T-best = T_direct_F (55.5%); TF 65.0% -> (a) TF >= T-best: combination beats both parts
  TF_wrong vs A_r2       b=32  c=43   p=0.2480
  TF_noq   vs A_r2       b=27  c=59   p=0.0007
  TF       vs TF_wrong   b=41  c=20   p=0.0099
R3 TF_wrong - A_r2 = -5.5 pp -> FORMAT-EFFECT FLAG
R3 TF_noq 44.0% vs A_r2 60.0% -> ok (< A_r2); TF_noq parsed-only 56.05% (D1 arm G: 42.0 overall, 63.2 parsed-only)
R4 teacher entity reading (a) D6-71 34.0% (ref 21.0%) -> ok; (b) all 32.39%
  [verified] 30 lines -> outputs/efbpt/d10/student/tf_facts_sample30.jsonl

§7 branch: R1 CONDITIONAL: exactly ONE fix iteration, declared in an amendment before it runs; then one FULL re-run
§7 override: FORMAT-EFFECT FLAG: TF - TF_wrong is reported as the content effect; the method claim is reduced to that difference
```

**Recomputed from the arm jsonl files** (all 200 qids present in every arm; gold agrees across all six arms on all 200 rows; teacher-direct correctness computed as `pred == gold`):

| comparison | n | first arm correct (acc) | second arm correct (acc) | Δ pp | wins | losses | discordant | exact two-sided p | matches d10_summary.json |
|---|---|---|---|---|---|---|---|---|---|
| **R1** TF vs A_r2 | 200 | 130 (65.00%) | 120 (60.00%) | +5.0 | 38 | 28 | 66 | 0.267811664846587 | yes |
| R2 TF vs T_direct | 200 | 130 (65.00%) | 54 (27.00%) | +38.0 | 96 | 20 | 116 | 4.25e-13 | yes |
| R2 TF vs T_direct, T_direct-parsed rows only (freeze T3 clause, unparsed 59.5% > 10%) | 81 | 50 (61.73%) | 54 (66.67%) | −4.9 | 16 | 20 | 36 | 0.6177 | yes (log) |
| **R2** TF vs T_direct_F (= T-best) | 200 | 130 (65.00%) | 111 (55.50%) | +9.5 | 58 | 39 | 97 | 0.06705179535289083 | yes |
| **R3** TF_wrong vs A_r2 | 200 | 109 (54.50%) | 120 (60.00%) | −5.5 | 32 | 43 | 75 | 0.24804574189252795 | yes |
| R3 TF_noq vs A_r2 | 200 | 88 (44.00%) | 120 (60.00%) | −16.0 | 27 | 59 | 86 | 0.0007317100653124087 | yes |
| TF vs TF_wrong (descriptive under A2.1) | 200 | 130 (65.00%) | 109 (54.50%) | +10.5 | 41 | 20 | 61 | 0.009853448195259264 | yes |

TF_noq parsed-only: 157 parsed, 88 correct = 56.05% (summary: same). R4 (b) recomputed from `R4_rows.b_all_rows`: Σ hits 206 / Σ gold titles 636 = 32.39% (summary: same); R4 (a) D6-71 68/200 = 34.0% is taken from the summary (the D6 sealed outputs were not re-read here).

**R1.** Δ = +5.0 pp ≥ +5.0 (conditional threshold) and < +8.0 (gate); p = 0.268 > 0.05 → **CONDITIONAL**, as the script says.

**R2** (Amendment 1 A1.2). T-best = T_direct_F (55.5% > T_direct 27.0%). TF 65.0% ≥ 55.5% → outcome **(a) TF ≥ T-best: combination beats both parts** (Δ = +9.5 pp, p = 0.067, not significant at 0.05). T_direct is unparsed on 119/200 rows (59.5%), so the freeze-T3 parsed-only clause applies: on its 81 parsed rows T_direct is 66.67% vs TF 61.73% (p = 0.62). T_direct_F is unparsed on 1/200.

**R3, TF_wrong — Amendment 2 A2.1 applied.** TF_wrong − A_r2 = **−5.5 pp**, below −4.0. Under A2.1 the sign decides: a negative excess is **WRONG-FACTS HARM** (facts about the wrong entity hurt, the D8 GEN mechanism), not FORMAT-EFFECT. The script's `FORMAT_EFFECT: true` flag, the log line `R3 TF_wrong - A_r2 = -5.5 pp -> FORMAT-EFFECT FLAG`, and the `§7 override` line are **superseded by A2.1** (the script predates A2.1's sign rule and labels by absolute value). Consequences, per A2.1: the **R1 claim (TF − A_r2 = +5.0 pp, CONDITIONAL) stands**; TF − TF_wrong = +10.5 pp (wins 41, losses 20, p = 0.0099) is reported as **descriptive**, not as the method claim; the harm is reported as a finding. The freeze §7 "FORMAT-EFFECT FLAG" override therefore does **not** apply.

**R3, TF_noq.** 44.0% < A_r2 60.0% → no LEAK FLAG (p = 0.0007 in the harmful direction). Parsed-only 56.05% (157 rows) beside D1 arm G's 42.0% overall / 63.2% parsed-only. Predicted-yes 28%, unparsed 21.5%.

**R4.** Teacher entity reading 34.0% on D6-71 (68/200 required titles; reference Qwen3-14B D6 G4 21.0%) > 30% → **no MECHANISM FLAG**. On all 200 rows 206/636 = 32.39%. Per qid, 154/200 rows have at least one gold title among the teacher's entities (§6).

**Descriptive (freeze §6, never gates).** Predicted-yes rate: A_r2 41.0%, TF 36.0%, TF_wrong 19.0%, TF_noq 28.0% (D8: A_r 47%, GEN 26%). Facts per question min 0 / median 6 / max 6, empty 1. Teacher parse failures 1. V5 removed 0. TF wins/losses by parse_ok: parse_ok true (n = 199) wins 38 losses 28; parse_ok false (n = 1) wins 0 losses 0. Truncation rate A_r2 5.5%, TF 2.5%, TF_wrong 4.5%, TF_noq 2.5%. Seed-8 sample of 30 TF facts written to `tf_facts_sample30.jsonl` (status `PENDING hand check`; 29 distinct qids, one qid drawn twice).

---

## 4. Accuracy table

**Verbatim from the student log (lines 1053–1061):**

```
ACCURACY TABLE   floor 56.50%
====================================================================================
arm        facts                                    n     acc%   unpars%  SEC acc%     yes%
A_r2       none (re-run, floor)                   200    60.00      4.00     60.00    41.00
TF         teacher facts, own qid                 200    65.00      0.50     65.00    36.00
TF_wrong   teacher facts, derangement(qid)        200    54.50      5.00     54.50    19.00
TF_noq     teacher facts, no question             200    44.00     21.50     44.00    28.00
T_direct   teacher alone on the student prompt    200    27.00     59.50     27.00    14.50
T_direct_F teacher alone + final-answer line (A1.1)   200    55.50      0.50     55.50    74.50
```

**Per-arm counts** (recomputed from the jsonl files):

| arm | n | correct | acc | predicted yes (rate) | predicted no | unparsed | truncated | gen tokens min / median / max |
|---|---|---|---|---|---|---|---|---|
| A_r2 | 200 | 120 | 60.00% | 82 (41.00%) | 110 | 8 | 11 | 147 / 435 / 1024 |
| TF | 200 | 130 | 65.00% | 72 (36.00%) | 127 | 1 | 5 | 174 / 439 / 1024 |
| TF_wrong | 200 | 109 | 54.50% | 38 (19.00%) | 152 | 10 | 9 | 121 / 432 / 1024 |
| TF_noq | 200 | 88 | 44.00% | 56 (28.00%) | 101 | 43 | 5 | 55 / 318 / 1024 |
| T_direct | 200 | 54 | 27.00% | 29 (14.50%) | 52 | 119 | 1 | 14 / 48 / 1024 |
| T_direct_F | 200 | 111 | 55.50% | 149 (74.50%) | 50 | 1 | 2 | 13 / 74 / 1024 |

All counts agree with `d10_summary.json` `accuracy_table`. In all six arms `pred_secondary == pred` on every row, so the SEC column equals the primary column. Against the 56.50% floor: A_r2 and TF are above it; TF_wrong, TF_noq, T_direct and T_direct_F (55.5%) are below it.

---

## 5. §7 branch

`d10_summary.json` `decision_tree.branch`: **"R1 CONDITIONAL: exactly ONE fix iteration, declared in an amendment before it runs; then one FULL re-run"**. The fix must be one of the freeze §7 list: (i) two format examples from the training pool; (ii) teacher self-consistency filter (3 samples at T = 0.7, keep facts present in ≥ 2); (iii) decomposition prompt (sub-questions before facts). No second iteration.

`decision_tree.overrides` lists the FORMAT-EFFECT override. Under A2.1 (§3) that override is void for a negative TF_wrong excess; no §7 override applies. No LEAK FLAG, so TF is read as fact injection, not teacher-answer transfer. R2 outcome (a) is recorded but does not change the branch, which is set by R1 alone.

---

## 6. LOSS/WIN ANALYSIS (descriptive; every number from the sealed files)

Join key: qid. Sources per column: A_r2 / TF / TF_wrong / TF_noq `pred`, `correct` from the arm files; `parse_ok`, `n_facts`, `entities`, `facts` from `t_facts.jsonl`; gold evidence titles (normalised `required_titles`), `hits` from `d10_summary.json` `R4_rows.b_all_rows`. **R4 hit** for a qid = `hits` non-empty (at least one gold title found in the teacher's entities): 154 yes / 46 no. Of the 46 no-hit rows, 1 has an empty entities list (the parse-fail row); the other 45 have entities that match no gold title. Sub-sections 6.2–6.7 come first; the full 200-row join (6.1) is placed last because of its length.

### 6.2 TF vs A_r2 and TF_wrong vs A_r2, split by R4 hit

| comparison | R4 hit | n | first arm correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses | discordant | exact p |
|---|---|---|---|---|---|---|---|---|---|
| TF vs A_r2 | yes | 154 | 104 (67.53%) | 95 (61.69%) | +5.8 | 31 | 22 | 53 | 0.2717 |
| TF vs A_r2 | no | 46 | 26 (56.52%) | 25 (54.35%) | +2.2 | 7 | 6 | 13 | 1.0 |
| TF vs A_r2 | all | 200 | 130 (65.00%) | 120 (60.00%) | +5.0 | 38 | 28 | 66 | 0.2678 |
| TF_wrong vs A_r2 | yes | 154 | 82 (53.25%) | 95 (61.69%) | −8.4 | 22 | 35 | 57 | 0.1112 |
| TF_wrong vs A_r2 | no | 46 | 27 (58.70%) | 25 (54.35%) | +4.3 | 10 | 8 | 18 | 0.8145 |
| TF_wrong vs A_r2 | all | 200 | 109 (54.50%) | 120 (60.00%) | −5.5 | 32 | 43 | 75 | 0.2480 |

(The R4 hit for TF_wrong rows is still the row's own qid; the injected facts there come from the donor qid.)

### 6.3 TF wins and losses by gold label

TF and A_r2 give the same prediction on 131/200 rows (including both-unparsed).

| | gold yes | gold no | total |
|---|---|---|---|
| TF wins (TF right, A_r2 wrong) | 15 | 23 | 38 |
| TF losses (TF wrong, A_r2 right) | 17 | 11 | 28 |
| rows overall | 87 | 113 | 200 |

Losses by (gold, TF pred): **gold yes → TF "no": 17**; gold no → TF "yes": 10; gold no → TF unparsed: 1. So 17 of the 28 losses are gold-yes rows that TF turned to "no".
Wins by (gold, A_r2 pred): gold no → A_r2 "yes": 20; gold yes → A_r2 "no": 13; gold no → A_r2 unparsed: 3; gold yes → A_r2 unparsed: 2. Wins by TF pred: 23 × "no" (gold no), 15 × "yes" (gold yes).
Losses by R4 hit: yes 22 / no 6. Wins by R4 hit: yes 31 / no 7.
For comparison, TF_wrong vs A_r2: 32 wins (gold no → A_r2 yes 24, gold yes → A_r2 no 5, gold no → A_r2 unparsed 3); 43 losses (gold yes → TF_wrong "no" **32**, gold no → TF_wrong yes 9, gold no → unparsed 2).

### 6.4 The 28 losses and 38 wins, row by row

In `docs/handoff/D10_LOSS_WIN_ROWS.md`: for each row qid, Urdu question (from the sealed A_r2 prompt), gold, A_r2 pred, TF pred, R4 hit and hits, teacher entities, all teacher facts (≤ 6), and the gold evidence titles.

### 6.5 T_direct_F yes-bias

| arm | P(pred yes \| gold yes) | P(pred yes \| gold no) | pred no \| gold yes | pred no \| gold no | unparsed \| gold yes | unparsed \| gold no |
|---|---|---|---|---|---|---|
| **T_direct_F** | **74/87 = 85.06%** | **75/113 = 66.37%** | 13 | 37 | 0 | 1 |
| T_direct | 16/87 = 18.39% | 13/113 = 11.50% | 14 | 38 | 57 | 62 |
| A_r2 | 47/87 = 54.02% | 35/113 = 30.97% | 37 | 73 | 3 | 5 |
| TF | 45/87 = 51.72% | 27/113 = 23.89% | 42 | 85 | 0 | 1 |
| TF_wrong | 20/87 = 22.99% | 18/113 = 15.93% | 63 | 89 | 4 | 6 |
| TF_noq | 26/87 = 29.89% | 30/113 = 26.55% | 39 | 62 | 22 | 21 |

T_direct_F answers "yes" on 149/200 rows; its 55.5% is 74 gold-yes hits + 37 gold-no hits. The gold yes-rate is 43.5%.

### 6.6 The 2 V6 verbatim gold-fact hits in TF

| qid | gold | gold-fact substring (V6) | teacher fact containing it | teacher entities | gold titles | A_r2 pred | TF pred | TF_noq pred |
|---|---|---|---|---|---|---|---|---|
| `4ec8cd118be8a53ec516` | no | Alexander Hamilton died in 1804. | "Alexander Hamilton died in 1804." (identical) | Alexander Hamilton; Cowpox | alexander hamilton; koala | yes (wrong) | no (right) | no (right) |
| `d1142162a82aab83611e` | no | Gandhi was assassinated on January 30, 1948. | "Mahatma Gandhi was assassinated on January 30, 1948." | Mahatma Gandhi; Bonanza (TV series) | bonanza; family of mahatma gandhi; mahatma gandhi | no (right) | no (right) | no (right) |

The same two teacher facts produce the TF_wrong V6 hits through the derangement: `1eb8a833f2f5f1698b52` (donor `d1142162…`) and `c928aaf3842cb87e3642` (donor `4ec8cd11…`). One of the two TF V6 rows (`4ec8cd11…`) is a TF win.

### 6.7 Fact-count effect

| facts in TF prompt | n | TF correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses | p |
|---|---|---|---|---|---|---|---|
| 6 | 199 | 130 (65.33%) | 120 (60.30%) | +5.0 | 38 | 28 | 0.2678 |
| fewer than 6 (= 0, parse fail) | 1 | 0 (0.00%) | 0 (0.00%) | 0.0 | 0 | 0 | 1.0 |

The teacher returned exactly 6 facts on every parsed row (199/200) and 0 on the one parse failure (`ba04b5ebc2edd682c4dd`, where the TF prompt equals the A_r2 prompt and both arms are wrong), so there is no within-run variation in fact count to compare. Same for TF_wrong by donor fact count: 199 rows with 6 donor facts (Δ −5.5 pp), 1 row with 0.

### 6.1 Per-row join (all 200 rows)

Columns: gold; pred and ✓ (1 = correct, unparsed shown as —) for A_r2, TF, TF_wrong, TF_noq; teacher parse_ok, n_facts; R4 hit; the gold titles hit; teacher entities. Row order = file order.

| # | qid | gold | A_r2 pred | A_r2 ✓ | TF pred | TF ✓ | TF_wrong pred | TF_wrong ✓ | TF_noq pred | TF_noq ✓ | parse_ok | n_facts | R4 hit | hits | entities |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `00e13b8d03d35929c049` | no | yes | 0 | yes | 0 | yes | 0 | — | 0 | 1 | 6 | no | — | 7-Eleven; Convenience store |
| 2 | `01430717f590ad3f4e74` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | the home depot | The Home Depot; Vertebral column |
| 3 | `018f6ea10facddae3b2e` | yes | yes | 1 | yes | 1 | no | 0 | — | 0 | 1 | 6 | yes | porsche | Dorothy Wendling; Porsche |
| 4 | `01c27885cede6741bc53` | yes | yes | 1 | no | 0 | no | 0 | yes | 1 | 1 | 6 | yes | the cw | Supernatural (TV series); The CW; 2001 |
| 5 | `05e9cdc44f1b066badd7` | no | yes | 0 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | moustache | Women; Moustache |
| 6 | `06215d4b6f1a4b975d9c` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | space needle | Federal Reserve; Space Needle |
| 7 | `064d8faaa8f164b55270` | yes | no | 0 | no | 0 | no | 0 | yes | 1 | 1 | 6 | yes | apollo 15; unicycle | Apollo 15; Unicycle |
| 8 | `06724dc213e0dae715f5` | yes | yes | 1 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | yes | netflix | Amazon; Netflix; Stock; Subscription |
| 9 | `068379affae04debca2b` | yes | yes | 1 | yes | 1 | yes | 1 | — | 0 | 1 | 6 | yes | nepalese civil war | Nepalese Civil War; India |
| 10 | `069c88860654c9c5df0b` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | no | — | Sea Turtle; Tennis Racket; Tennis |
| 11 | `06ebd3ca1fe98d5796f7` | no | no | 1 | no | 1 | no | 1 | — | 0 | 1 | 6 | yes | douglas adams; email | Douglas Adams; Email |
| 12 | `06f7878425a995c2a633` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | spaghetti | Spaghetti; Cooking |
| 13 | `076f3f72b7e6d17073eb` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | sound barrier | MTR; Airport Express (MTR); Sound barrier |
| 14 | `076f905ae36408b5ee69` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | beach; rainbow | Rainbow; Beach; Sun |
| 15 | `0d22526961c82ad6ef4a` | yes | yes | 1 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | yes | fairy; valkyrie | Fairy; Valkyrie; Fairy tale |
| 16 | `0fe3edf35bd1a1e662a9` | no | yes | 0 | no | 1 | yes | 0 | no | 1 | 1 | 6 | yes | gandalf | Gandalf; Barbossa; Pirates of the Caribbean |
| 17 | `10325585370ba368ca9a` | no | yes | 0 | yes | 0 | no | 1 | yes | 0 | 1 | 6 | yes | goat | Goat; Abyssinian Highlands |
| 18 | `1534ef2130cb48d6abb4` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | atherosclerosis | Gelatin; Atherosclerosis |
| 19 | `15885efe6f91724c16d9` | yes | yes | 1 | no | 0 | yes | 1 | no | 0 | 1 | 6 | yes | audi r8; sound barrier | Audi R8; Sound barrier |
| 20 | `16c08e5f354b035c41a5` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | the godfather; usb flash drive | The Godfather; USB flash drive |
| 21 | `16d7c42444ef13ef4643` | no | yes | 0 | yes | 0 | yes | 0 | yes | 0 | 1 | 6 | no | — | Cat Goslin; Bandie |
| 22 | `17b8fa74f450c4d0e56b` | yes | yes | 1 | yes | 1 | yes | 1 | yes | 1 | 1 | 6 | no | — | Alice's Adventures in Wonderland; Disney |
| 23 | `1a4eb85c31188af515b0` | yes | yes | 1 | yes | 1 | yes | 1 | — | 0 | 1 | 6 | yes | los angeles county, california | The Walt Disney Company; Los Angeles County, California |
| 24 | `1c56958c04a98ccd5d1e` | no | no | 1 | no | 1 | no | 1 | — | 0 | 1 | 6 | no | — | Mercenary; Jalilianwala Bagh Massacre; England |
| 25 | `1d923b8a2778448a9716` | no | no | 1 | no | 1 | — | 0 | — | 0 | 1 | 6 | yes | cactus | Cactus; Soil |
| 26 | `1eb8a833f2f5f1698b52` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | disneyland; tom and jerry | Tom and Jerry; Disneyland |
| 27 | `20693ee66ad7ca890e08` | no | no | 1 | no | 1 | no | 1 | — | 0 | 1 | 6 | yes | great pyramid of giza | Great Pyramid of Giza; Pyramid |
| 28 | `20abc7ba538a2fd1dd76` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | cellulose | Small intestine; Cellulose |
| 29 | `24418db1d18a7fdf79f3` | yes | no | 0 | no | 0 | no | 0 | — | 0 | 1 | 6 | yes | hypervitaminosis d | Ocimum tenuiflorum; Hypervitaminosis D |
| 30 | `24a3aff40ef7fd04e208` | yes | no | 0 | no | 0 | no | 0 | — | 0 | 1 | 6 | yes | richard dawkins | Richard Dawkins; Secular humanism |
| 31 | `25a088d9d2ce674e639a` | yes | yes | 1 | yes | 1 | yes | 1 | no | 0 | 1 | 6 | no | — | Jackfruit; Food allergy; Anaphylaxis |
| 32 | `265dd54c248f8b048851` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | no | — | Red Globo; Chinese language |
| 33 | `267609f49ccd3c0839c3` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | swiss guard; virginia general assembly | Swiss Guard; Virginia General Assembly |
| 34 | `27368c21e50b6af694ab` | no | yes | 0 | yes | 0 | no | 1 | yes | 0 | 1 | 6 | no | — | Cezanne; Professor of Ethics |
| 35 | `2a4e1d4524997b6263bf` | no | no | 1 | yes | 0 | no | 1 | — | 0 | 1 | 6 | yes | hermes; iris (mythology) | Iris (mythology); Hermes; UPS |
| 36 | `2aba85e0e138ad86035e` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | boxing | Sugar Ray Robinson; Iran; Boxing |
| 37 | `2bc94c6dfc79d7a86fa2` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | christopher walken | Christopher Walken; United States Marine Corps |
| 38 | `2c9f3c00412a4018380f` | yes | yes | 1 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | no | — | Furniture; Bamboo |
| 39 | `2e508efbf0b72f1af2c2` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | oxygen; quartz | Quartz; Oxygen; Plant |
| 40 | `2e8e9e51f0f290e32d27` | yes | yes | 1 | no | 0 | yes | 1 | no | 0 | 1 | 6 | no | — | Ajwain; Culture; Death |
| 41 | `300927fd135e8d94af16` | yes | yes | 1 | yes | 1 | yes | 1 | yes | 1 | 1 | 6 | yes | prime number | Prime number; Days of the week |
| 42 | `30fcf62f58849cb4ab05` | yes | no | 0 | yes | 1 | no | 0 | — | 0 | 1 | 6 | yes | great pyramid of giza; theocracy | Great Pyramid of Giza; Theocracy |
| 43 | `310968736449680ee61e` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | no | — | Ima Roberts; Nancy Drew; Private investigator |
| 44 | `315575d5cbdd1d154744` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | kurt cobain | Kurt Cobain; Project Sami Collen |
| 45 | `3157a0f428531ebed7b9` | yes | yes | 1 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | yes | minor league baseball | Minor League Baseball; Feline |
| 46 | `3295844627a9bd1b9135` | yes | yes | 1 | yes | 1 | no | 0 | no | 0 | 1 | 6 | yes | harry potter; the matrix | Harry Potter; Warner Bros.; The Matrix |
| 47 | `3486b6b5c69878bb088d` | no | yes | 0 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | arizona; california | Arizona; California; Cabbage |
| 48 | `3503e8d41364eb0b33bc` | no | — | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | no | — | Milly Bobby Brown; Bartender |
| 49 | `3570d57f16a40488ff42` | yes | yes | 1 | no | 0 | no | 0 | — | 0 | 1 | 6 | yes | judo | Silicon; Judo |
| 50 | `35a5b27d84294ebf1fe0` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | no | — | Zodiac; Satanism |
| 51 | `38ef97eb7cdd4200fd00` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | no | — | Sarah Paxton; Latin Americans |
| 52 | `3a61dcbed7e358b3aee8` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | great pyramid of giza | Great Pyramid of Giza; Egypt |
| 53 | `3ae8644164c547560eda` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | anne boleyn; king arthur | King Arthur; Anne Boleyn; Execution of Anne Boleyn |
| 54 | `3b1834a6dd331eb1d791` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | chief executive officer | Chief executive officer; Toilet |
| 55 | `3b38e00d8d1da26f7f02` | no | yes | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | no | — | Mario; Super Mushroom |
| 56 | `3bba4ea27500361cb141` | no | no | 1 | no | 1 | yes | 0 | no | 1 | 1 | 6 | yes | lincoln tunnel; oasis-class cruise ship | Oasis-class cruise ship; Lincoln Tunnel |
| 57 | `3cec996f26ec99d0a519` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | the godfather | Wu-Tang Clan; The Godfather |
| 58 | `3d7050c0e9bfbeef27c1` | yes | no | 0 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | yes | melania trump; olga kurylenko | Melania Trump; Olga Kurylenko |
| 59 | `3de88c80f0914eb08caf` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | leaf | Leaf; Color |
| 60 | `3ebb62af96f836d2d4d6` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | no | — | Film; Con artist |
| 61 | `3f5af97fdff20028347b` | no | yes | 0 | yes | 0 | yes | 0 | yes | 0 | 1 | 6 | yes | french revolution; television | French Revolution; Television |
| 62 | `3f8a1bd6bf3a967cdeb6` | no | no | 1 | no | 1 | no | 1 | — | 0 | 1 | 6 | yes | hedgehog; vertebrate | Hedgehog; Vertebrate |
| 63 | `3fc11f59d64ea3b65136` | yes | yes | 1 | yes | 1 | no | 0 | — | 0 | 1 | 6 | no | — | Thanksgiving; Mourning |
| 64 | `4125c5f4c7a536916608` | yes | yes | 1 | yes | 1 | yes | 1 | yes | 1 | 1 | 6 | no | — | Retail; Chief executive officer; Salary; Jealousy |
| 65 | `42c947475244ed02629e` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | guitar | Capybara; Guitar |
| 66 | `43da475d5c2db929c458` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | eiffel tower | Eiffel Tower; Arctic Ocean |
| 67 | `4589898e4852e5389728` | yes | yes | 1 | no | 0 | no | 0 | yes | 1 | 1 | 6 | yes | ymca | YMCA; Headquarters |
| 68 | `46428dac0ceefa1ad73b` | yes | no | 0 | no | 0 | no | 0 | — | 0 | 1 | 6 | no | — | Erick Clapton; United States; Minimum age for purchasing tobacco products in the United States |
| 69 | `4760fe3d0c80beb30edd` | no | yes | 0 | no | 1 | no | 1 | — | 0 | 1 | 6 | yes | honey; veganism | Veganism; Honey |
| 70 | `4773d577873ff9e04a88` | no | no | 1 | no | 1 | no | 1 | — | 0 | 1 | 6 | yes | woodrow wilson | Woodrow Wilson; African Americans |
| 71 | `4810297f474b0e97ed78` | yes | no | 0 | yes | 1 | no | 0 | no | 0 | 1 | 6 | yes | child; seawater | Gallon; Seawater; Child |
| 72 | `48da75d87c66754ccc2e` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | speaker of the united states house of representatives | Pearl Harbor; Speaker of the United States House of Representatives |
| 73 | `4bbf4b169b4981ad5a34` | no | yes | 0 | yes | 0 | yes | 0 | yes | 0 | 1 | 6 | yes | miami | New York; Miami; Swimming |
| 74 | `4de177ae8c827fd0ffb1` | no | no | 1 | yes | 0 | yes | 0 | — | 0 | 1 | 6 | yes | retail | Retail; Employment |
| 75 | `4e1b65e81ec09397b26e` | yes | no | 0 | no | 0 | no | 0 | yes | 1 | 1 | 6 | no | — | Colossal Squid; Habitat |
| 76 | `4ec8cd118be8a53ec516` | no | yes | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | alexander hamilton | Alexander Hamilton; Cowpox |
| 77 | `502d4dae6f08e73a5569` | yes | no | 0 | yes | 1 | no | 0 | no | 0 | 1 | 6 | no | — | Home Depot; Sleek |
| 78 | `557e389d7efe47c5bc4d` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | ultramarine | Ultramarine; Pigment; Lapis lazuli; Tyrian purple; Vantablack; YInMn blue |
| 79 | `5897ec22db850f7b416e` | no | no | 1 | yes | 0 | no | 1 | yes | 0 | 1 | 6 | yes | jacinda ardern | List of female heads of government; Jacinda Ardern |
| 80 | `593e66cbdec8e3a15852` | no | yes | 0 | yes | 0 | yes | 0 | — | 0 | 1 | 6 | no | — | Survivor (TV series); Ancient Greek alphabet |
| 81 | `598be3dcddbd7775a827` | no | no | 1 | yes | 0 | no | 1 | no | 1 | 1 | 6 | yes | bob marley | Bob Marley; Genghis Khan; Tug of war |
| 82 | `5ab66b27a0caee886970` | no | no | 1 | no | 1 | yes | 0 | no | 1 | 1 | 6 | yes | miami; penguin | Miami; Penguin |
| 83 | `5e2d9519c7c521411817` | yes | — | 0 | yes | 1 | no | 0 | — | 0 | 1 | 6 | no | — | TMNT; Coloring book; Pizza |
| 84 | `5f24cbd41798ed5c7608` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | saint peter; television | Saint Peter; Television |
| 85 | `63fb8f40ca6c2226dd01` | yes | no | 0 | no | 0 | — | 0 | — | 0 | 1 | 6 | no | — | Helen Palmer Geisel; Seuss |
| 86 | `64354cb6ee6d987d4f0f` | yes | no | 0 | no | 0 | — | 0 | no | 0 | 1 | 6 | yes | food; ice | Ice; Food; Human |
| 87 | `660bfdc17b47f42facac` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | no | — | Mouse; Arendelle |
| 88 | `67079beff747978ed4c2` | no | yes | 0 | yes | 0 | no | 1 | yes | 0 | 1 | 6 | yes | fantastic four | Mephisto (comics); Fantastic Four |
| 89 | `684d4c03be354635b80f` | no | no | 1 | no | 1 | yes | 0 | — | 0 | 1 | 6 | yes | afterlife; sheol | Christianity; Afterlife; Sheol |
| 90 | `690a36d5fed5b17969aa` | no | yes | 0 | no | 1 | no | 1 | — | 0 | 1 | 6 | yes | jack dempsey | Jack Dempsey; World Boxing Council; Heavyweight |
| 91 | `6c43fa359095fd0845f5` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | bactrian camel | Bactrian camel; Humps |
| 92 | `6f4c8a2789c0305c4f63` | yes | yes | 1 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | yes | maya angelou | Maya Angelou; Children of Maya Angelou |
| 93 | `703db4629ff46d82a205` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | joe biden | Joe Biden; United States presidential primary |
| 94 | `70b2122323424f6e80c2` | no | yes | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | lebron james; yuri gagarin | LeBron James; Yuri Gagarin |
| 95 | `71ca1a8a75c5c582ab3f` | no | — | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | head coach | High school; Head coach; NCAA |
| 96 | `7281474f2760dce03f39` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | no | — | Crane (machine); Slurpee |
| 97 | `73c52134ab2a903d86db` | no | yes | 0 | no | 1 | — | 0 | no | 1 | 1 | 6 | yes | lemon | Lemon; Milk |
| 98 | `73ca2ef1da65b2a2ebe6` | yes | yes | 1 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | yes | darth vader | Darth Vader; Spouse |
| 99 | `7419826e4373eb1e83ff` | yes | no | 0 | yes | 1 | — | 0 | no | 0 | 1 | 6 | yes | dog; grey seal | Dog; Grey seal; Bell |
| 100 | `7571100f05bc56919c78` | yes | no | 0 | yes | 1 | no | 0 | no | 0 | 1 | 6 | yes | funeral; happy hardcore | Funeral; Happy hardcore |
| 101 | `7870b1cef39a4f685911` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | no | — | Adam Sandler; Cold Sore; Scuba Diving |
| 102 | `7b84d2bc643ddc2085f0` | no | yes | 0 | no | 1 | — | 0 | no | 1 | 1 | 6 | yes | lincoln tunnel; noah's ark | Noah's Ark; Lincoln Tunnel |
| 103 | `7be51fdef30345de666e` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | crustacean | Crustacean; Ocean |
| 104 | `7c3759cc1da78e9fbd79` | yes | no | 0 | yes | 1 | yes | 1 | no | 0 | 1 | 6 | yes | diwali | Diwali; Christianity |
| 105 | `7dc74fd9b62e514750c2` | yes | no | 0 | no | 0 | no | 0 | yes | 1 | 1 | 6 | yes | apple | Apple; Queen bee |
| 106 | `7ea958a52fc626ac5577` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | evander holyfield | Evander Holyfield; Boxing |
| 107 | `7f139bf51273517efd4e` | yes | no | 0 | no | 0 | no | 0 | yes | 1 | 1 | 6 | yes | christopher nolan | Christopher Nolan; Bob Kane |
| 108 | `7f4effbc97ab2b5fd4a7` | yes | yes | 1 | yes | 1 | yes | 1 | — | 0 | 1 | 6 | yes | breakfast | United States; United Kingdom; Breakfast; Dinner |
| 109 | `806a13f37055bd21fddc` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | george fox; stoning | George Fox; Stoning |
| 110 | `80aa769f55b14c1e4d8d` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | wolverine | Sibel; Wolverine |
| 111 | `80ca095e38ab73b908ff` | no | yes | 0 | yes | 0 | no | 1 | no | 1 | 1 | 6 | no | — | Norwegian Forest Cat; Child |
| 112 | `84a83f07a543ae3024c9` | yes | no | 0 | no | 0 | no | 0 | — | 0 | 1 | 6 | yes | yellow pages | High school; Yellow Pages |
| 113 | `867a5425c26092b30fbf` | no | yes | 0 | no | 1 | no | 1 | yes | 0 | 1 | 6 | no | — | The Walt Disney Company; Snow White and the Seven Dwarfs |
| 114 | `86ead3f15417204affc2` | yes | no | 0 | no | 0 | no | 0 | — | 0 | 1 | 6 | yes | dustin hoffman | Dustin Hoffman; Jewish people |
| 115 | `8cfde6ee28d059a5aff6` | yes | yes | 1 | yes | 1 | yes | 1 | yes | 1 | 1 | 6 | no | — | Atacama Large Millimeter/submillimeter Array; Walking; Death |
| 116 | `8d06b619a7045ed02f51` | no | yes | 0 | yes | 0 | no | 1 | — | 0 | 1 | 6 | yes | shirley bassey | Shirley Bassey; Diamonds (Shirley Bassey and Diamonds album) |
| 117 | `8d3ddaee20ad48edc066` | yes | yes | 1 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | yes | brazilian jiu-jitsu; gracie family | Brazilian jiu-jitsu; Gracie family |
| 118 | `8d708927b1593d8b9b42` | yes | no | 0 | yes | 1 | no | 0 | no | 0 | 1 | 6 | yes | plessy v. ferguson; woodrow wilson | Woodrow Wilson; Plessy v. Ferguson |
| 119 | `8e03ed9fce31618b095a` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | no | — | The Three Musketeers; D'Artagnan; Athos; Porthos; Aramis |
| 120 | `8eecf2ff459ef5a9b98a` | yes | yes | 1 | yes | 1 | no | 0 | no | 0 | 1 | 6 | yes | cactus | Cactus; Stomata |
| 121 | `9052063382d30870a2ed` | no | — | 0 | yes | 0 | no | 1 | — | 0 | 1 | 6 | no | — | Citalopram; Major depressive disorder; Selective serotonin reuptake inhibitor |
| 122 | `90fc9a677ae0df6f2974` | no | no | 1 | no | 1 | yes | 0 | no | 1 | 1 | 6 | yes | nostradamus | Nostradamus; Dante Alighieri; Inferno; Third Circle of Hell |
| 123 | `91917c5e30dbf8552b34` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | veganism | Veganism; Traditional paya |
| 124 | `9247b5afb2eef4c9f95b` | yes | yes | 1 | yes | 1 | yes | 1 | no | 0 | 1 | 6 | yes | minotaur; tibia | Minotaur; Association football; Tibia |
| 125 | `92c263f997f09dbb74a8` | no | no | 1 | no | 1 | yes | 0 | no | 1 | 1 | 6 | yes | maya angelou | CHE101; Maya Angelou |
| 126 | `9384dfc0aabc73026925` | yes | yes | 1 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | no | — | Physical heart; Symbolic heart |
| 127 | `9576ab36999e196b87bb` | no | no | 1 | yes | 0 | no | 1 | no | 1 | 1 | 6 | yes | eastenders | EastEnders; Water |
| 128 | `959fb200fccf056f00d7` | no | no | 1 | no | 1 | — | 0 | no | 1 | 1 | 6 | yes | naruto | Naruto; Temple of Doom |
| 129 | `9711b3fc634e6f4f471d` | no | — | 0 | yes | 0 | — | 0 | no | 1 | 1 | 6 | yes | alan turing | World War II; Alan Turing |
| 130 | `9850eb1534206facd067` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | central processing unit; integrated circuit | Central processing unit; Integrated circuit; Microchip |
| 131 | `987260ffef60cb2c5439` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | pearl harbor | Pearl Harbor; Shark Girl |
| 132 | `9883d6c816d4855035ea` | yes | yes | 1 | yes | 1 | yes | 1 | no | 0 | 1 | 6 | yes | onion; tree | Onion; Tree |
| 133 | `9b491a31c4212e70b18c` | no | yes | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | jane austen; middle child syndrome | Jane Austen; Middle child syndrome |
| 134 | `9bb59488a227a5f5cd62` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | game of thrones | Game of Thrones; GB; HD video |
| 135 | `9cfa675eeebfbbdb2487` | no | yes | 0 | yes | 0 | no | 1 | no | 1 | 1 | 6 | yes | bern; xenophobia | Bern; Xenophobia; Swiss citizenship |
| 136 | `9e78c01675a198781cca` | yes | no | 0 | yes | 1 | yes | 1 | no | 0 | 1 | 6 | yes | michael bloomberg | Michael Bloomberg; Federated States of Micronesia; Decade |
| 137 | `9fee16a5e703c2fdcd33` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | no | — | Blair White; Traditionalist conservatism |
| 138 | `a0896de3fd13cd0f3e16` | no | no | 1 | yes | 0 | no | 1 | yes | 0 | 1 | 6 | yes | atlantic cod | Atlantic cod; Cod as food; Sandwich |
| 139 | `a2dfceff60d5b18fc70d` | yes | — | 0 | no | 0 | — | 0 | no | 0 | 1 | 6 | yes | carolina reaper | Carolina Reaper; Salsa |
| 140 | `a406e06cd0164c1c6d15` | yes | yes | 1 | yes | 1 | yes | 1 | yes | 1 | 1 | 6 | no | — | Auburn, New York; September; Tree; Fall foliage |
| 141 | `a4de29904c45a2d17511` | yes | yes | 1 | yes | 1 | no | 0 | — | 0 | 1 | 6 | no | — | Focal Fossa; Linux |
| 142 | `a54b16376f82a6e14bf9` | no | no | 1 | yes | 0 | yes | 0 | no | 1 | 1 | 6 | no | — | Earth's atmosphere; Stone |
| 143 | `a6f593e93ca87c53a0a1` | yes | no | 0 | yes | 1 | no | 0 | — | 0 | 1 | 6 | yes | crucifixion; eighth amendment to the united states constitution | Crucifixion; Eighth Amendment to the United States Constitution |
| 144 | `a895b73bdfb2397b2e27` | yes | yes | 1 | yes | 1 | yes | 1 | — | 0 | 1 | 6 | no | — | Friday; 2015 |
| 145 | `a8bbf2e84d0649b1c52c` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | no | — | Monk; War |
| 146 | `abd771e9a20b107e7e56` | yes | no | 0 | no | 0 | yes | 1 | yes | 1 | 1 | 6 | no | — | Nikola Tesla; Brand |
| 147 | `acb7edac010554509519` | yes | no | 0 | yes | 1 | no | 0 | — | 0 | 1 | 6 | yes | chlorophyll | Human; Chlorophyll |
| 148 | `ae0891799990442fbb0e` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | paul bunyan | Paul Bunyan; Urban planning |
| 149 | `b2dddce981183d5a083d` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | robert downey jr. | Robert Downey Jr.; Resident Evil; Milla Jovovich |
| 150 | `b3443bf3c312c9d144e4` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | neptune | Pantheon, Paris; Neptune; Women in France |
| 151 | `b4414a29cba573a24c6c` | no | no | 1 | no | 1 | no | 1 | — | 0 | 1 | 6 | yes | flag of the united states | Leukophobia; Flag of the United States |
| 152 | `b64be915de0e671e8548` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | easter bunny | Easter Bunny; September |
| 153 | `b6713901d33bc1020596` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | richard wagner; wolfgang amadeus mozart | Wolfgang Amadeus Mozart; Richard Wagner |
| 154 | `b747938f597b09e43603` | yes | yes | 1 | yes | 1 | no | 0 | — | 0 | 1 | 6 | yes | jia sidao | Goguryeo; Jia Sidao |
| 155 | `b77170e15c0b1bd72340` | no | yes | 0 | no | 1 | yes | 0 | yes | 0 | 1 | 6 | yes | durian | Durian; Fruit |
| 156 | `b8df6ac181c4c5b280f9` | no | no | 1 | — | 0 | no | 1 | — | 0 | 1 | 6 | yes | bottlenose dolphin; human | Bottlenose Dolphin; Human; Reproduction |
| 157 | `ba04b5ebc2edd682c4dd` | no | yes | 0 | yes | 0 | no | 1 | yes | 0 | 0 | 0 | no | — | (none) |
| 158 | `bcd77b200a2b90ca2ecc` | yes | yes | 1 | yes | 1 | yes | 1 | no | 0 | 1 | 6 | yes | bloodstain pattern analysis; jackson pollock | Bloodstain pattern analysis; Jackson Pollock |
| 159 | `bd734e6cb6674e25dbe8` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | l. ron hubbard; tom cruise | Tom Cruise; L. Ron Hubbard |
| 160 | `bfb0f6c573076a1a0634` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | kings of judah | Solomon; Kings of Judah; Prophets of Islam |
| 161 | `c51fd5841358ded6eaec` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | ten commandments | Ten Commandments; Bible; Judaism |
| 162 | `c67945e7c471a9692c88` | no | yes | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | snowy owl; sonoran desert | Snowy Owl; Sonoran Desert |
| 163 | `c70d40b57fa67ea13ed6` | no | yes | 0 | yes | 0 | yes | 0 | yes | 0 | 1 | 6 | yes | tonsure | Tonsure; Hair |
| 164 | `c921bdade889def26cf4` | yes | yes | 1 | yes | 1 | no | 0 | — | 0 | 1 | 6 | yes | isopropyl alcohol; solubility | Isopropyl alcohol; Solubility; Sodium chloride |
| 165 | `c928aaf3842cb87e3642` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | lake urmia | Saltwater Crocodile; Lake Urmia; Battle of Lake Urmia |
| 166 | `cac049d9ec0b86a7f911` | no | yes | 0 | yes | 0 | no | 1 | — | 0 | 1 | 6 | yes | starbucks | Starbucks; China; United States |
| 167 | `cac8e6300a98cf6128af` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | no | — | 19th century; President of the United States; Hong Kong; Numerology |
| 168 | `cb9eae3948f28e236477` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | giant panda; gorilla | Giant Panda; Gorilla; Bear |
| 169 | `d0145341694e7034555d` | no | yes | 0 | no | 1 | no | 1 | — | 0 | 1 | 6 | yes | javits center; united nations framework convention on climate change | United Nations Framework Convention on Climate Change; Javits Center |
| 170 | `d0183768701c74f966c5` | no | yes | 0 | yes | 0 | — | 0 | yes | 0 | 1 | 6 | yes | shiva | Shiva; Hindu pilgrimage sites |
| 171 | `d1142162a82aab83611e` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | mahatma gandhi | Mahatma Gandhi; Bonanza (TV series) |
| 172 | `d15a33b8f88dc5ce32b0` | no | yes | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | no | — | Dholak; Stringed instrument |
| 173 | `d1dbb41001ab4ad3b924` | no | yes | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | no | — | Blue lips; Normality |
| 174 | `d47120efd0f09badd848` | yes | yes | 1 | no | 0 | no | 0 | — | 0 | 1 | 6 | yes | menthol | Menthol; Christmas |
| 175 | `d808a0c72dbe605309ab` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | dodgeball | Felix Potvin; Dodgeball |
| 176 | `d8fd81e475df8c2f6972` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | cassowary | Cassowary; Domestication |
| 177 | `d9aeae10998c093c0cc9` | no | no | 1 | no | 1 | yes | 0 | yes | 0 | 1 | 6 | yes | toyota supra | Toyota Supra; Turbocharger |
| 178 | `dabf94934721c26b2422` | no | no | 1 | yes | 0 | no | 1 | yes | 0 | 1 | 6 | yes | european wildcat | European Rabbit; European Wildcat |
| 179 | `dce3695a2aaa8d7dba5a` | no | no | 1 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | bill nye; franklin d. roosevelt | Bill Nye; Franklin D. Roosevelt |
| 180 | `de62d38623fde2c2285a` | yes | yes | 1 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | yes | week | Week; Episode |
| 181 | `e12a00a2c45fcb4f38e7` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | othello | Othello; William Shakespeare; Scheherazade |
| 182 | `e12fef0504a959e49b23` | no | no | 1 | yes | 0 | yes | 0 | no | 1 | 1 | 6 | yes | coffee | New England; Coffee |
| 183 | `e26a5601edb90a738d56` | no | no | 1 | no | 1 | no | 1 | — | 0 | 1 | 6 | yes | the simpsons | Featurettes; The Simpsons |
| 184 | `e530682239bb5e59ebf0` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | barcelona; madrid; panzer viii maus | Cat; Barcelona; Madrid; Panzer VIII Maus |
| 185 | `e6d3973ed3feb8a42928` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | africanized bee; multiculturalism | Africanized bee; Multiculturalism |
| 186 | `e87b63e92165b417d37f` | yes | no | 0 | no | 0 | no | 0 | yes | 1 | 1 | 6 | yes | michael vick | Michael Vick; Animal welfare |
| 187 | `f07a6acf27e25dedbb8d` | no | no | 1 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | andrew johnson; elagabalus | Andrew Johnson; Elagabalus |
| 188 | `f087d393e2ab1f8783f5` | no | yes | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | 1980 united states presidential election | 1980 United States presidential election; Photo finish |
| 189 | `f2859b2ce17b5f5a6ad9` | no | yes | 0 | no | 1 | no | 1 | no | 1 | 1 | 6 | yes | chinchilla; rodent | Chinchilla; Rodent; Felis catus |
| 190 | `f43533225534420816d6` | yes | yes | 1 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | soldier field | Soldier Field; Squid |
| 191 | `f4a336168f5165de7f0d` | yes | no | 0 | yes | 1 | yes | 1 | no | 0 | 1 | 6 | yes | northern fur seal | July; Emergency surgery; Northern fur seal; General anaesthesia |
| 192 | `f4a731f739533c52becd` | yes | no | 0 | no | 0 | no | 0 | no | 0 | 1 | 6 | yes | prime minister of the united kingdom | Prime Minister of the United Kingdom; United Kingdom |
| 193 | `f69c8412d57ad726d658` | yes | yes | 1 | yes | 1 | no | 0 | no | 0 | 1 | 6 | yes | cholera | Cholera |
| 194 | `f7a32d14efd589015fe1` | no | — | 0 | no | 1 | yes | 0 | no | 1 | 1 | 6 | yes | herpes simplex virus; venus | Herpes simplex virus; Venus |
| 195 | `f7b1cc5b3fab95aa1be4` | yes | no | 0 | no | 0 | no | 0 | — | 0 | 1 | 6 | no | — | Breast Cancer; Rib |
| 196 | `f8e6087cd52c925fedad` | yes | yes | 1 | no | 0 | no | 0 | yes | 1 | 1 | 6 | yes | tony bennett | Tony Bennett; 2020 |
| 197 | `fba624d8c01833419760` | yes | no | 0 | yes | 1 | yes | 1 | yes | 1 | 1 | 6 | yes | august | August; Season; Southern Hemisphere; Northern Hemisphere |
| 198 | `fc9339f21f44a3841a7c` | yes | — | 0 | yes | 1 | no | 0 | yes | 1 | 1 | 6 | yes | koala | Koala; Eucalyptus |
| 199 | `fdeec181aeee06303113` | no | yes | 0 | no | 1 | no | 1 | yes | 0 | 1 | 6 | yes | karachi | Karachi; Cross |
| 200 | `ff3811735ededd8ec3a7` | no | no | 1 | yes | 0 | no | 1 | — | 0 | 1 | 6 | no | — | Salim Ali; Lalita Kumari; Sylvia Sparrow; Family (biology) |

---

## 7. V5 sample and V6 matches

**V5** (`d10_summary.json` `hand_check_sample`: seed 8, n = 30, file `outputs/efbpt/d10/student/tf_facts_sample30.jsonl`, status `PENDING hand check`; each line carries `truth: null`, `relevance: null` to be filled by hand):

| # | qid | teacher entities | fact |
|---|---|---|---|
| 1 | `557e389d7efe47c5bc4d` | Ultramarine; Pigment; Lapis lazuli; Tyrian purple; Vantablack; YInMn blue | Tyrian purple is a natural dye that was highly valued in antiquity for its rarity and expense. |
| 2 | `9576ab36999e196b87bb` | EastEnders; Water | The show is known for its portrayal of working-class life in London's East End. |
| 3 | `9711b3fc634e6f4f471d` | World War II; Alan Turing | World War II was a global conflict that lasted from 1939 to 1945. |
| 4 | `315575d5cbdd1d154744` | Kurt Cobain; Project Sami Collen | Kurt Cobain was an American musician and artist, best known as the lead singer and guitarist of the grunge band Nirvana. |
| 5 | `43da475d5c2db929c458` | Eiffel Tower; Arctic Ocean | The Eiffel Tower is not designed to be submerged in water. |
| 6 | `0d22526961c82ad6ef4a` | Fairy; Valkyrie; Fairy tale | Fairy tales often feature supernatural creatures, including fairies, and are frequently set in a fantasy world or a medieval setting. |
| 7 | `24a3aff40ef7fd04e208` | Richard Dawkins; Secular humanism | Richard Dawkins is a British evolutionary biologist and science writer. |
| 8 | `3486b6b5c69878bb088d` | Arizona; California; Cabbage | The United States Department of Agriculture (USDA) tracks data on cabbage production by state. |
| 9 | `63fb8f40ca6c2226dd01` | Helen Palmer Geisel; Seuss | Dr. Seuss was an American author, poet, and illustrator best known for his children's books. |
| 10 | `d47120efd0f09badd848` | Menthol; Christmas | Peppermint flavoring is often used in holiday treats and drinks during the Christmas season. |
| 11 | `48da75d87c66754ccc2e` | Pearl Harbor; Speaker of the United States House of Representatives | As of 1941, the Speaker of the United States House of Representatives was Sam Rayburn. |
| 12 | `9fee16a5e703c2fdcd33` | Blair White; Traditionalist conservatism | Blair White's views on social issues have been criticized by some conservatives. |
| 13 | `06ebd3ca1fe98d5796f7` | Douglas Adams; Email | Email is a method of exchanging digital messages from an sender to one or more recipients. |
| 14 | `bcd77b200a2b90ca2ecc` | Bloodstain pattern analysis; Jackson Pollock | Pollock's paintings are characterized by intricate webs of lines and patterns. |
| 15 | `cb9eae3948f28e236477` | Giant Panda; Gorilla; Bear | Giant pandas are members of the Ursidae family, which includes bears. |
| 16 | `b77170e15c0b1bd72340` | Durian; Fruit | Durian is banned in many public places in Southeast Asia due to its strong odor. |
| 17 | `9bb59488a227a5f5cd62` | Game of Thrones; GB; HD video | GB is an abbreviation for gigabyte, a unit of digital information. |
| 18 | `d0183768701c74f966c5` | Shiva; Hindu pilgrimage sites | Hindus often visit sacred sites to perform rituals and seek blessings from their deities. |
| 19 | `fba624d8c01833419760` | August; Season; Southern Hemisphere; Northern Hemisphere | The Northern Hemisphere experiences winter while the Southern Hemisphere experiences summer during the months of December, January, and February. |
| 20 | `43da475d5c2db929c458` | Eiffel Tower; Arctic Ocean | The Arctic Ocean's deepest point is in the Molloy Deep, with a depth of approximately 5,370 meters (17,650 feet). |
| 21 | `a0896de3fd13cd0f3e16` | Atlantic cod; Cod as food; Sandwich | A sandwich is a food item typically consisting of two or more slices of bread with fillings between them. |
| 22 | `25a088d9d2ce674e639a` | Jackfruit; Food allergy; Anaphylaxis | Jackfruit has been reported to cause allergic reactions in some individuals, including anaphylaxis. |
| 23 | `cac8e6300a98cf6128af` | 19th century; President of the United States; Hong Kong; Numerology | Numerology is the belief in the mystical relationship between numbers and their influence on human life. |
| 24 | `593e66cbdec8e3a15852` | Survivor (TV series); Ancient Greek alphabet | The number of letters in the Ancient Greek alphabet has remained relatively consistent throughout history. |
| 25 | `064d8faaa8f164b55270` | Apollo 15; Unicycle | Unicycles require balance and coordination to operate. |
| 26 | `6f4c8a2789c0305c4f63` | Maya Angelou; Children of Maya Angelou | Maya Angelou was an American poet, memoirist, and civil rights activist. |
| 27 | `dce3695a2aaa8d7dba5a` | Bill Nye; Franklin D. Roosevelt | Bill Nye was born on November 27, 1955. |
| 28 | `a406e06cd0164c1c6d15` | Auburn, New York; September; Tree; Fall foliage | Auburn, New York is a city located in Cayuga County, New York, United States. |
| 29 | `c70d40b57fa67ea13ed6` | Tonsure; Hair | The appearance of a person's hair can be influenced by their genetics, hormones, and environmental factors. |
| 30 | `9850eb1534206facd067` | Central processing unit; Integrated circuit; Microchip | A microchip is a miniaturized electronic device that contains a set of electronic circuits on a small flat piece of semiconductor material. |

**V6 matches** (`d10_summary.json` `V6_gold_fact_verbatim`; reported, not a gate; details in §6.6):

| arm | count | qids | qid | matching gold-fact substring |
|---|---|---|---|---|
| A_r2 | 0 | 0 | | |
| TF | 2 | 2 | `4ec8cd118be8a53ec516` | Alexander Hamilton died in 1804. |
| | | | `d1142162a82aab83611e` | Gandhi was assassinated on January 30, 1948. |
| TF_wrong | 2 | 2 | `1eb8a833f2f5f1698b52` | Gandhi was assassinated on January 30, 1948. |
| | | | `c928aaf3842cb87e3642` | Alexander Hamilton died in 1804. |
| TF_noq | 2 | 2 | `4ec8cd118be8a53ec516` | Alexander Hamilton died in 1804. |
| | | | `d1142162a82aab83611e` | Gandhi was assassinated on January 30, 1948. |

---

## 8. Log grep

`grep -n -E 'WARNING|Traceback|FATAL|LEAK|VOID' logs/d10_student_FULL_109582.log` returns **no matches**.

The same grep on `logs/d10_teacher_FULL_109581.log` returns two vLLM start-up lines only (log lines 102 and 127: `Enforce eager set, overriding optimization level to -O0` and `Inductor compilation was disabled by user settings…`), both expected from `enforce_eager=True` in the frozen vLLM args. No Traceback, FATAL, LEAK or VOID in either log. The lowercase `[leak-guard]` lines (teacher log 34–36, student log 39–42) report every prompt clean except the 2 gold-fact substrings per facts arm listed as V6.

---

## 9. Open items carried forward

- D10 V5 hand check of the 30 TF facts (`tf_facts_sample30.jsonl`, truth and relevance fields null).
- D8 V5 hand check of 30 Xp facts (still open from D8).
- §7: one fix iteration to be chosen from (i)/(ii)/(iii) and declared in a D10 amendment before it runs.
