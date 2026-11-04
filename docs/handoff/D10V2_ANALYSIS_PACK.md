# D10 v2 analysis pack: jobs 110652–110655 (T1-v2 decomposition teacher, Amendments 3/4/5)

Read-only analysis of the completed D10 v2 FULL run (teacher T1-v2 fact injection, freeze `b3a822d8`, Amendments 1–5). Sources: `outputs/efbpt/d10v2/teacher/` (`t_facts.jsonl`, `derangement.json`, `T_START.json`, `T_SEAL.json`), `outputs/efbpt/d10v2/student/` (4 arm files, `d10_summary.json`, `S_START.json`, `S_SEAL.json`, `tf_facts_sample30.jsonl`), the sealed v1 files under `outputs/efbpt/d10/{teacher,student}/`, `logs/d10v2_*_11065{2,3,4,5}.log`, `sacct`, `docs/EFBPT_D10_AMENDMENT_3.md`, `_4.md`, `_5.md`, `docs/handoff/D10_LOSS_WIN_ROWS.md`. Gold question_en, gold evidence titles and gold facts (audit columns only) come from the DEV200 file `data/strategyqa_official/dev200_seed4242.jsonl`; the eval458 pre-check (§10) reads `data/sdfr_splits/strategyqa_eval.jsonl` and `data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl`. Every number below was recomputed with Python from the jsonl files; where a number also exists in `d10_summary.json` the last column says whether it matches. This file is the only file written. No edits, commits, or sbatch.

Conventions as in `docs/handoff/D10_FULL_READOUT.md`: "wins" = the first-named arm is right and the second is wrong; "losses" = the reverse. p = exact two-sided binomial on the discordant pairs (`min(1, 2·Σ_{i≤min(b,c)} C(n,i)/2^n)`), the freeze's McNemar. Unparsed = wrong. Gold is yes on 87/200 and no on 113/200. A_r2 is the v2 job's own re-run of the no-facts arm (its file is byte-identical to the v1 A_r2 file, §5a). "v1" = jobs 109581/109582, "v2" = jobs 110654/110655.

**Headline.** R1 TF − A_r2 = +4.5 pp (64.50% vs 60.00%, wins 41, losses 32, p = 0.3492) → below the +5.0 pp conditional threshold → **NO-GO** under A3.5; the script's §7 branch is "the teacher-facts route is closed; freeze §7 last branch". V1 spread 6.00 pp fails the 5.0 limit only through TF_wrong (unparsed 6.5%); under A3.4 R1 is valid and every TF_wrong reading is VOID. R2: T-best = T_direct_F (55.5%, sealed v1); TF ≥ T-best → outcome (a) (Δ +9.0 pp, p = 0.0982). R3 TF_noq 45.00% < A_r2 → no LEAK FLAG. R4 72/200 = 36.00% on D6-71 (ref 21.0%) → no MECHANISM FLAG. Against v1 (TF 65.0%, +5.0 pp, CONDITIONAL): the v2 teacher gives fewer facts (median 3 vs 6), fixes 13 of the 28 v1 TF losses but loses 10 of the 38 v1 TF wins, and among rows concordant in v1 creates 13 new wins and 17 new losses (§5d); net −1 TF correct.

---

## 1. Run status

**SLURM** (`sacct -j 110652,110653,110654,110655`, job steps omitted):

| job | name | State | ExitCode | node | Start | End | Elapsed | Timelimit |
|---|---|---|---|---|---|---|---|---|
| 110652 | d10v2_teacher_TEST | COMPLETED | 0:0 | L20002 | 2026-11-02T21:27:47 | 2026-11-02T21:33:14 | 00:05:27 | 01:00:00 |
| 110653 | d10v2_student_TEST | COMPLETED | 0:0 | L20005 | 2026-11-02T21:33:14 | 2026-11-02T21:47:51 | 00:14:37 | 02:00:00 |
| 110654 | d10v2_teacher_FULL | COMPLETED | 0:0 | L20005 | 2026-11-02T21:47:51 | 2026-11-02T22:28:42 | 00:40:51 | 02:00:00 |
| 110655 | d10v2_student_FULL | COMPLETED | 0:0 | L20005 | 2026-11-02T22:28:42 | 2026-11-03T04:28:19 | 05:59:37 | 12:00:00 |

The four jobs ran as one `--dependency=afterok` chain (A5.2): each started the second the previous one ended. TEST-5 teacher ran on L20002; TEST-5 student, FULL teacher and FULL student on L20005. Both FULL jobs COMPLETED with exit 0 inside their limits (teacher 00:40:51 of 02:00:00; student 05:59:37 of 12:00:00). Last lines of the two TEST logs: `TEST COMPLETE. Nothing above is a result.` Last line of the FULL logs: `D10 v2 JOB T COMPLETE. Sealed 3 files.` / `D10 v2 JOB S COMPLETE. Sealed 7 files.`

**T_SEAL.json** (v2): job `T`, version `v2 (Amendments 3, 4, 5)`, mode `FULL`, slurm_job_id `110654`, end_time `2026-11-02 22:28:39`, n_rows 200, t1_parse_ok 198, t1_repaired_rows 35, t2_removed_total 0.
**S_SEAL.json** (v2): job `S`, version `v2 (Amendment 3)`, mode `FULL`, slurm_job_id `110655`, end_time `2026-11-03 04:28:18`; it re-lists the 3 teacher MD5s (identical to T_SEAL) and pins the 3 v1 files (`t_direct.jsonl`, `t_direct_f.jsonl`, `derangement.json`).

**Sealed files, all MD5s recomputed now:**

| seal | file | MD5 (seal) | bytes (seal) | lines (seal) | MD5 now | bytes now | lines now | match |
|---|---|---|---|---|---|---|---|---|
| T_SEAL | `outputs/efbpt/d10v2/teacher/T_START.json` | `cdd7323a10654a11aece4f22a3100845` | 8431 | 309 | `cdd7323a10654a11aece4f22a3100845` | 8431 | 309 | match |
| T_SEAL | `outputs/efbpt/d10v2/teacher/t_facts.jsonl` | `6261fb6340c9c71123e37437473e2851` | 747686 | 200 | `6261fb6340c9c71123e37437473e2851` | 747686 | 200 | match |
| T_SEAL | `outputs/efbpt/d10v2/teacher/derangement.json` | `8f00e2f9e54f27138964a0577a8eff52` | 10516 | 206 | `8f00e2f9e54f27138964a0577a8eff52` | 10516 | 206 | match |
| S_SEAL | `outputs/efbpt/d10v2/student/S_START.json` | `441804af10df94d5f5f2670b7c52d3eb` | 13813 | 537 | `441804af10df94d5f5f2670b7c52d3eb` | 13813 | 537 | match |
| S_SEAL | `outputs/efbpt/d10v2/student/d10_armA_r2.jsonl` | `7ccec942c8f0eee08156d258baaf806d` | 366128 | 200 | `7ccec942c8f0eee08156d258baaf806d` | 366128 | 200 | match |
| S_SEAL | `outputs/efbpt/d10v2/student/d10_armTF.jsonl` | `1a3cdba1f26ccca4b010a00676197f0a` | 395475 | 200 | `1a3cdba1f26ccca4b010a00676197f0a` | 395475 | 200 | match |
| S_SEAL | `outputs/efbpt/d10v2/student/d10_armTF_wrong.jsonl` | `a9a4884383d578b861187a36eb3f0209` | 427302 | 200 | `a9a4884383d578b861187a36eb3f0209` | 427302 | 200 | match |
| S_SEAL | `outputs/efbpt/d10v2/student/d10_armTF_noq.jsonl` | `6cbf2c7d645fedb5eaef0717b54e95c9` | 305052 | 200 | `6cbf2c7d645fedb5eaef0717b54e95c9` | 305052 | 200 | match |
| S_SEAL | `outputs/efbpt/d10v2/student/tf_facts_sample30.jsonl` | `42c5ba9103a70c587ae6be3553a9469f` | 10095 | 30 | `42c5ba9103a70c587ae6be3553a9469f` | 10095 | 30 | match |
| S_SEAL | `outputs/efbpt/d10v2/student/d10_summary.json` | `7ea88ae8d040d2efe01c2d744cfbd7a4` | 186624 | 7866 | `7ea88ae8d040d2efe01c2d744cfbd7a4` | 186624 | 7866 | match |

S_SEAL `teacher_seal_files` (3 MD5s) equal T_SEAL `files`: yes. S_SEAL `v1_pinned_files` recomputed now: all match (`t_direct.jsonl` `bac96cd9a9773b5c5e719e982ecfe499`, `t_direct_f.jsonl` `40ebfe643e244c5a1a3fff729f23462c`, v1 `derangement.json` `8f00e2f9e54f27138964a0577a8eff52`); the v2 `derangement.json` is byte-identical to the v1 file (same MD5 `8f00e2f9e54f27138964a0577a8eff52`). The seal files themselves (not self-listed): `T_SEAL.json` MD5 `bbf8512e4a9ddf3dff0f3ee0c084945b` (964 bytes, 32 lines); `S_SEAL.json` MD5 `1a39be2ea1f9b74b614111eca74bc628` (1952 bytes, 56 lines).

**Freeze, amendments, scripts, instruction** (MD5 as recorded in the START/SEAL files vs recomputed now):

| item | recorded where | MD5 recorded | MD5 now | match |
|---|---|---|---|---|
| freeze `docs/EFBPT_D10_TEACHER_FREEZE.md` | T_START, T_SEAL, S_START, S_SEAL | `b3a822d8199034c7fe58d76cf5cc428f` | `b3a822d8199034c7fe58d76cf5cc428f` | match |
| Amendment 3 `docs/EFBPT_D10_AMENDMENT_3.md` | T_START, T_SEAL, S_START, S_SEAL | `eb097415ba4fa254c96300e6e3d7755e` | `eb097415ba4fa254c96300e6e3d7755e` | match |
| Amendment 4 `docs/EFBPT_D10_AMENDMENT_4.md` | T_START, T_SEAL (not in S_*) | `28a39b3d2adea6eca309b40fc15a00e5` | `28a39b3d2adea6eca309b40fc15a00e5` | match |
| Amendment 5 `docs/EFBPT_D10_AMENDMENT_5.md` | T_START, T_SEAL (not in S_*) | `53026e77e67a08124a4ee42aa38c556f` | `53026e77e67a08124a4ee42aa38c556f` | match |
| teacher script `eval/error_analysis_tests/efbpt/d10v2_teacher.py` | T_START, T_SEAL, S_START.teacher_seal | `39d35ac316fc2a34fcd38324d406729c` | `39d35ac316fc2a34fcd38324d406729c` | match |
| student script `eval/error_analysis_tests/efbpt/d10v2_student.py` | S_START, S_SEAL | `fc5802f1667769eb23af0ff8771dea01` | `fc5802f1667769eb23af0ff8771dea01` | match |
| Urdu instruction `prompts/efbpt/plan_a_instruction_ur.txt` | T_START, S_START | `f3b58d766fe3ec2573ff4f24761cf0c9` | `f3b58d766fe3ec2573ff4f24761cf0c9` | match |

Amendment 1 and 2 MD5s are not recorded in any v2 START/SEAL file (recomputed now: `ac828661e26d77ca9bf00695ce4b8b93`, `c7968a1699b837d48ecf9be6fd7e1605`). The Amendment 5 MD5 is in the teacher files only; the student script pins the freeze and Amendment 3 (its code is unchanged by A4/A5, MD5 `fc5802f1…` as A5.2 states).

**Git head**: T_START `04638fbb4d5bf4b15af809b0520ac24f319c21f6` (dirty False), S_START `04638fbb4d5bf4b15af809b0520ac24f319c21f6` (dirty False); current HEAD `04638fbb4d5bf4b15af809b0520ac24f319c21f6`, working tree dirty (before this file was written). Same commit.

**Start times.** Teacher log banner 21:47:54; `T_START.start_time` 2026-11-02 21:48:15 (after preflight); model load 125 s (9 AWQ shards). Student log banner 22:28:44; `S_START.start_time` 2026-11-02 22:49:44 (after preflight and the V2 probe, 610.1 s + 609.9 s for 20 prompts twice).

**Per-arm time** (final progress line of each pass in the logs; s/row exact = total/200; eval458 column = 458 × exact s/row, generation only, no model load or probe):

| job | pass / arm | rows | total s (log) | s/row (log) | s/row (exact) | finish / unparsed | projected 458 rows |
|---|---|---|---|---|---|---|---|
| T 110654 | T1-v2 teacher facts | 200 | 2299.3 | 11.5 | 11.50 | stop 200, truncated 0 | 5265 s = 1.46 h |
| S 110655 | A_r2 | 200 | 6021.0 | 30.1 | 30.11 | unparsed 8 | 13788 s = 3.83 h |
| S 110655 | TF | 200 | 5100.9 | 25.5 | 25.50 | unparsed 1 | 11681 s = 3.24 h |
| S 110655 | TF_wrong | 200 | 5828.3 | 29.1 | 29.14 | unparsed 13 | 13347 s = 3.71 h |
| S 110655 | TF_noq | 200 | 3362.6 | 16.8 | 16.81 | unparsed 47 | 7700 s = 2.14 h |

Student arm totals sum to 20312.8 s (5.64 h) of the 05:59:37 job; the rest is model load, the V2 probe (1220 s) and the readings. Teacher T1-v2 at 11.50 s/row is 1.15× the v1 T1 (10.03 s/row) because of the longer JSON (five keys). For an eval458 run with the same four student arms at these rates: 46516 s ≈ 12.9 h of generation plus ≈ 25 min load + probe; teacher ≈ 5265 s ≈ 1.5 h plus 2 min load. Student prompt lengths (`[len]` lines): A_r2 54/79/114 tokens (min/median/max), TF 64/145/262, TF_wrong 63/147/240, TF_noq 44/105/206 (v1 TF was 63/202/292).

---

## 2. Teacher v2 outputs (`t_facts.jsonl`, 200 rows, arm `T_facts_v2`)

| quantity | recomputed | d10_summary.json / T_SEAL | match |
|---|---|---|---|
| parse_ok rows | 198/200 = 99.0% | V4 198/200 = 99.0%; T_SEAL t1_parse_ok 198 | yes |
| repaired rows by tag | brace 34, bracket 1, bracket_brace 0, not repaired 165 | T_SEAL t1_repaired_rows 35; teacher log `{'brace': 34, 'bracket': 1}` | yes |
| finish reasons | {'stop': 200} | teacher log `{'stop': 200}` | yes |
| truncated | 0 | teacher log `truncated 0` | yes |
| V5 removed (Σ n_removed / Σ len(removed)) | 0 / 0 | V5 removed_lines 0; T_SEAL t2_removed_total 0 | yes |
| n_gen_tokens min / median / max | 73 / 167.5 / 293 | not in summary | — |
| facts kept (post-T2) | 683 | V9 n_facts 683 | yes |
| Latin fraction of kept facts | 1.0000 (47140/47140 alphabetic chars) | V9 1.0000 (47140/47140) | yes |
| facts_parsed == facts on every row (T2 removed nothing) | yes | — | — |

Non-ASCII letters inside the kept facts (all counted as Latin by the script's rule): `é` ×6.

**Histograms** (per row; recomputed vs `A3_3_descriptives` in the summary):

| per-row count | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ | median | summary match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| facts (post-T2 = parsed) | 4 | 3 | 28 | 87 | 50 | 5 | 23 | 0 | 0 | 3.0 | yes |
| sub_questions | 2 | 0 | 8 | 110 | 79 | 1 | 0 | 0 | 0 | 3.0 | yes |
| unknown_terms | 182 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | yes |
| entities | 4 | 26 | 112 | 43 | 11 | 4 | 0 | 0 | 0 | 2.0 | not in summary |

Rows with 0 facts: 4 — `265dd54c248f8b048851` (parse_ok True, repaired False, unknown_terms ['Red Globe']); `2c9f3c00412a4018380f` (parse_ok True, repaired False, unknown_terms ['bal']); `bcd77b200a2b90ca2ecc` (parse_ok False, repaired False, unknown_terms []); `e530682239bb5e59ebf0` (parse_ok False, repaired False, unknown_terms []). The two parse failures contribute 0 by construction; the other two (`265dd54c…`, `2c9f3c00…`) parsed with an empty `facts` list, each with one unknown term (`Red Globe`, `bal`). Rows with exactly 6 facts: 23 (v1: 199). Rows with ≥ 1 unknown term: 18 (summary rows_with_any 18, match yes); no row has more than one. The sub_questions key was empty on 2 rows (both parse failures) and the prompt's 2–4 range was exceeded once (5).

The 18 unknown terms (qid → term): `265dd54c` → "Red Globe"; `2c9f3c00` → "bal"; `315575d5` → "Project Semi-Colon"; `3cec996f` → "Wai Yip"; `4589898e` → "ریکلیٹ"; `660bfdc1` → "Arendelle"; `684d4c03` → "Sheol"; `703db462` → "Cain"; `92c263f9` → "CHE101"; `9576ab36` → "East Enders series"; `959fb200` → "مندرِ قیامت"; `987260ff` → "Shark Girl"; `9b491a31` → "Middle Child Syndrome"; `a0896de3` → "veggie might sandwich"; `b747938f` → "Jia Sidao"; `c928aaf3` → "Lake Urmia"; `e26a5601` → "Featurettes 2020"; `f2859b2c` → "Felis catus mouse".

**The 2 unparsed rows** (parse_ok false, repaired false, finish reason stop, not truncated; raw text verbatim):

`bcd77b200a2b90ca2ecc` — gold yes, n_gen_tokens 218. Gold question_en (audit): "Is it understandable to compare a blood spatter pattern to a Jackson Pollock piece?".

```
{
"question_en": "Can a blood splatter pattern be compared to a Jackson Pollock artwork?",
"unknown_terms": [],
"sub_questions": [
"What are the characteristics of blood splatter patterns?",
"What are the characteristics of Jackson Pollock's artworks?",
"Can the patterns of blood splatter be considered a form of art?",
"Are there any scientific studies comparing blood splatter patterns to art?"),
"entities": [
"Bloodstain pattern analysis",
"Jackson Pollock"],
"facts": [
"Bloodstain pattern analysis is a technique used in forensic science to reconstruct crimes.",
"Jackson Pollock was an American painter and a major figure in the abstract expressionist movement.",
"Pollock's technique of dripping and splashing paint onto a canvas created complex patterns.",
"Bloodstain patterns can be influenced by various factors, including the surface texture and the velocity of the blood droplets.",
"Pollock's artworks are known for their intricate web of lines and colors.",
"Bloodstain pattern analysis is used to help investigators reconstruct the events surrounding a crime." 
}
```

`e530682239bb5e59ebf0` — gold no, n_gen_tokens 227. Gold question_en (audit): "Could a cat ride Panzer VIII Maus tank missile from Barcelona to Madrid?".

```
{
  "question_en": Can a cat travel from Barcelona to Madrid on a Panzer VIII Maus tank missile?,
  "unknown_terms": [],
  "sub_questions": [
    "Is the Panzer VIII Maus a type of missile?",
    "Is the Panzer VIII Maus capable of transporting a cat?",
    "Is the Panzer VIII Maus a functional vehicle?",
    "What is the distance between Barcelona and Madrid?"
  ],
  "entities": [
    "Panzer VIII Maus",
    "Barcelona",
    "Madrid",
    "Cat"
  ],
  "facts": [
    "The Panzer VIII Maus was a German World War II super-heavy tank.",
    "The Panzer VIII Maus was not a missile, but a tank.",
    "The Panzer VIII Maus was a prototype and only two hulls and one turret were completed before the end of the war.",
    "The distance between Barcelona and Madrid is approximately 505 kilometers (314 miles).",
    "Cats are domesticated mammals that are not typically transported on military vehicles."
  ]
}
```

Why neither repair fired: `bcd77b20…` closes the `sub_questions` list with `)` instead of `]` and omits the `]` before the final `}`, so the balanced block fails `json.loads` and the A5.1 single `]` insertion before the final `}` still leaves the stray `)`. `e5306822…` has an unquoted `question_en` string value; the block is balanced, `json.loads` fails, and no repair shape covers it. Stored fields for both rows: question_en "", all four lists empty, n_facts 0; the TF prompt for these qids equals the A_r2 prompt (layout A).

**R4 teacher entity reading** (title-level, D6 G4 way: gold required titles found after `norm()` in the teacher's parsed entities / total required titles) and the A3.3 restatement-level reading (gold titles whose normalised string occurs in the normalised teacher question_en):

| reading | subset | recomputed | summary | match |
|---|---|---|---|---|
| R4 v2 (a) | D6-71 | 72/200 = 36.00% | 72/200 = 36.0% | yes |
| R4 v2 (b) | all 200 | 217/636 = 34.12% | 217/636 = 34.12% | yes |
| restatement v2 (a) | D6-71 | 68/200 = 34.00% | 68/200 = 34.0% | yes |
| restatement v2 (b) | all 200 | 218/636 = 34.28% | 218/636 = 34.28% | yes |
| R4 v1 (a) | D6-71 | 68/200 = 34.00% | v1 summary 68/200 = 34.0% | yes |
| R4 v1 (b) | all 200 | 206/636 = 32.39% | v1 summary 206/636 = 32.39% | yes |

Reference (D6 G4, Qwen3-14B itself): 42/200 = 21.0%; flag at ≤ 30%. The D6-71 qids and gold titles were re-derived from `outputs/efbpt/d6/d6_armE1.jsonl` (71 qids) and the DEV200 evidence ids via `required_titles()`+`norm()`; they equal the summary's `R4_rows` lists on every row. **Per-qid R4 hit** (≥ 1 gold title among the teacher's entities): v1 154 yes / 46 no; v2 159 yes / 41 no. Changes v1 → v2: no→yes on 15 qids (`00e13b8d`, `16d7c424`, `17b8fa74`, `27368c21`, `46428dac`, `4e1b65e8`, `593e66cb`, `5e2d9519`, `660bfdc1`, `80ca095e`, `867a5425`, `8cfde6ee`, `a895b73b`, `ba04b5eb`, `d1dbb410`); yes→no on 10 qids (`01430717`, `20abc7ba`, `684d4c03`, `71ca1a8a`, `b747938f`, `bcd77b20`, `d8fd81e4`, `de62d386`, `e5306822`, `f2859b2c`); unchanged on 175. The restatement-level reading (34.28%) is essentially the same as the entity-level one (34.12%): when the teacher misreads a term it misreads it already in its English restatement.

---

## 3. Validity

**Verbatim from the student log (`logs/d10v2_student_FULL_110655.log` lines 1013–1024):**

```
====================================================================================
D10 v2 VALIDITY (freeze §5, fixed order; A3.4)
====================================================================================
V1 unparsed spread A_r2/TF/TF_wrong: {'A_r2': 4.0, 'TF': 0.5, 'TF_wrong': 6.5} -> 6.00 pp -> PASS for R1 (A3.4: spread fails only through TF_wrong; TF_wrong readings VOID)   (TF_noq unparsed 23.50%, reported; |TF - A_r2| = 3.50 pp)
V2 determinism probe 20/20 identical -> PASS
V3 leak guards: none fatal in either job -> PASS
V4 teacher parse 198/200 = 99.0% -> PASS
V5 verdict-filter removed 0 -> ok
V6 verbatim gold-fact substrings (reported, not a gate): A_r2 0 in 0 qid(s), TF 5 in 4 qid(s), TF_wrong 5 in 4 qid(s), TF_noq 5 in 4 qid(s)
V7 T_SEAL verified (3 files) -> PASS
V8 prompt-content check: 592 facts prompts -> PASS
V9 Latin fraction of injected facts 1.0000
```

Preflight lines (student log 31–48): `[V7] T_SEAL outputs/efbpt/d10v2/teacher/T_SEAL.json: 3 files verified (md5, bytes, lines) -> PASS`; `[A3.2] sealed v1 files verified by pinned MD5: t_direct.jsonl, t_direct_f.jsonl, derangement.json`; `[leak-guard] A_r2 200 prompts clean (200 question_en, 532 gold facts, 862 evidence ids checked)`; TF / TF_wrong / TF_noq `question_en and evidence ids clean; 5 verbatim gold-fact substring(s) in 4 prompt(s), reported as V6`; `[V8] prompt-content check: 592 facts prompts equal their sealed post-T2 facts -> PASS`; `[A3.1] teacher question_en / sub_questions (863 strings) absent from all 800 student prompts`; `[V9] Latin fraction of injected teacher facts 1.0000 (47140/47140 letters; D8 GEN 0.9972)`. Teacher log: `[leak-guard] T1v2/facts 200 prompts clean`, `T_START.leak_guard.fatal_hits` 0.

**Recomputed from the sealed files:**

| check | summary value | recomputed | status |
|---|---|---|---|
| V1 unparsed A_r2 / TF / TF_wrong | 4.0 / 0.5 / 6.5 → spread 6.00 pp (limit 5.0), pass false | 8 / 1 / 13 of 200 → 6.00 pp; \|TF − A_r2\| = 3.50 pp | **FAIL as a spread; PASS for R1 under A3.4** (fails only through TF_wrong; TF_wrong readings VOID) |
| V1 TF_noq unparsed (reported) | 23.5% | 47/200 = 23.5% | reported |
| V2 probe | 20/20 identical | S_START `V2_determinism_probe` identical 20, pass True | PASS |
| V3 leak guards | teacher 200 prompts, student 800 prompts, 0 fatal | gold question_en found in 0 of 800 sealed prompts (recomputed); teacher question_en/sub_questions (863 strings) found in 0 prompts | PASS |
| V4 teacher parse | 198/200 = 99.0% (min 190) | 198 `parse_ok` true | PASS |
| V5 verdict filter | 0 removed (max 100) | Σ `n_removed` = 0; all `removed` lists empty | ok |
| V6 gold-fact substrings | TF 5 in 4 qids, TF_wrong 5 in 4, TF_noq 5 in 4, A_r2 0 | see §7 (recomputed from `urbench_facts` needles: same 5 facts, same qids) | reported |
| V7 T_SEAL | 3 files | all 3 MD5/bytes/lines match now (§1) | PASS |
| V8 prompt content | 592 facts prompts | 592 fact-bearing prompts (TF 196 + TF_wrong 196 + TF_noq 200, the latter on all 200 layout-G prompts, 4 with an empty facts block); every prompt contains exactly its sealed post-T2 facts and `n_facts_given` equals the teacher `n_facts` (own qid, or donor for TF_wrong): all | PASS |
| V9 Latin fraction | 1.0000 (47140 letters, 683 facts) | 683 facts, 47140 alphabetic chars, 47140 Latin (§2) | reported |

Layouts recomputed: A_r2 {'A': 200}; TF {'B': 196, 'A': 4}; TF_wrong {'B': 196, 'A': 4}; TF_noq {'G': 200} (summary `layouts`: same). The 4 layout-A rows in TF are the 4 zero-fact teacher rows; in TF_wrong the 4 layout-A rows are the qids whose donor has 0 facts (`0fe3edf3`, `92c263f9`, `959fb200`, `9883d6c8`). TF_noq is layout G on all 200 rows, with an empty facts block on 4; A zero-fact TF_noq prompt is the Urdu instruction line followed by an empty `حقائق:` header and no question (the student still answers; all 4 such rows parsed).

Flags in `d10_summary.json`: DETERMINISM false, FORMAT false, TEACHER_ANSWERING false, FORMAT_EFFECT false, LEAK false, MECHANISM false, V1_VOID false, **TF_WRONG_VOID_A3_4 true**.

**V1 detail: unparsed rows per arm** (primary extractor; `pred_secondary == pred` on every row of every arm, so the secondary column is identical):

| arm | unparsed | of which truncated (hit 1024 tokens) | gold yes / no among unparsed |
|---|---|---|---|
| A_r2 | 8 | 3 | 3 / 5 |
| TF | 1 | 0 | 1 / 0 |
| TF_wrong | 13 | 1 | 8 / 5 |
| TF_noq | 47 | 0 | 23 / 24 |

**The 13 TF_wrong unparsed rows** (first 150 characters of the generation, newlines shown as ⏎; donor = `derangement[qid]`):

| # | qid | gold | donor qid | donor n_facts | truncated | A_r2 same qid | TF same qid | TF_wrong generation (first 150 chars) |
|---|---|---|---|---|---|---|---|---|
| 1 | `15885efe6f91724c16d9` | yes | `86ead3f1` | 2 | no | parsed → yes (right) | parsed → no (wrong) | آوڈی آر ایٹ وی-ٹین پلس (Audi A8 L) کے لیے صوتی رکاوٹ توڑنا ممکن ہے، لیکن اس کی صوتی کیفیت کی کچھ محدودیتیں ہوسکتی ہیں۔ یہ مسئلہ عام طور پر گاڑی کی سیٹ |
| 2 | `1d923b8a2778448a9716` | no | `c928aaf3` | 5 | no | parsed → no (right) | parsed → no (right) | کیکٹس کی مٹی ہمیشہ نم ہونی چاہیے۔ کیکٹس کی گیلری کی مٹی کو ہمیشہ نم رکھنا چاہیے تاکہ گیلری کی جڑیں ہائیڈریشن کے لیے پانی کی فراہمی یقینی بنائیں۔ تاہم، |
| 3 | `2a4e1d4524997b6263bf` | no | `7870b1ce` | 3 | no | parsed → no (right) | parsed → no (right) | اس سوال کا جواب دیا جا سکتا ہے کہ **ہریمیز** اور **آئرس (اساطیر)** کی بات کرنا ایک افسانوی یا تاریخی مفہوم ہے، جبکہ **یو پی ایس (UPS)** ایک واقعی میں  |
| 4 | `3295844627a9bd1b9135` | yes | `f2859b2c` | 3 | no | parsed → yes (right) | parsed → yes (right) | اس سوال کا جواب دیا جاتا ہے کہ ہیری پوٹر اور میٹرکس دونوں وارنر برادرز کے لیے بہت مہیا سرمایہ کاری تھیں، لیکن ان کی مہیا اور اثر اندازی کے لحاظ سے مخت |
| 5 | `63fb8f40ca6c2226dd01` | yes | `a2dfceff` | 4 | no | parsed → no (wrong) | unparsed | ہیلن پالمر کی موت کا ڈاکٹر سیوس پر نمایاں اثر ہوا۔ ہیلن پالمر ڈاکٹر سیوس کی اہلیہ تھیں اور ان کی موت نے ڈاکٹر سیوس کی زندگی اور کام کے انداز پر گہری ا |
| 6 | `64354cb6ee6d987d4f0f` | yes | `d8fd81e4` | 6 | no | parsed → no (wrong) | parsed → no (wrong) | اس سوال کا جواب یہ ہے کہ **برف پر مشتمل غذا انسان کو مار ڈالے گی**۔⏎⏎لیکن یہ بات سمجھنا ضروری ہے کہ یہ "برف پر مشتمل غذا" کس معنی میں ہے۔ اگر یہ مطلب  |
| 7 | `84a83f07a543ae3024c9` | yes | `7419826e` | 3 | no | parsed → no (wrong) | parsed → yes (right) | اس سوال کا جواب یہ ہے کہ **ہائی اسکول کے طالب علم کے لیے پیلا صفحات (yellow pages) کا استعمال غیر معمولی ہو سکتا ہے**۔⏎⏎**سبب:**⏎- **پیلا صفحات** ایک  |
| 8 | `8cfde6ee28d059a5aff6` | yes | `bfb0f6c5` | 4 | yes | parsed → yes (right) | parsed → no (wrong) | اس سوال کا جواب یہ ہے کہ **ویری لارج ٹیلیسکوپ تک پیدل جاتے ہوئے مرنا ممکن ہے**۔⏎⏎اس کی وجوہات یہ ہیں:⏎⏎1. **ویری لارج ٹیلیسکوپ** (Very Large Telescope |
| 9 | `b8df6ac181c4c5b280f9` | no | `e12a00a2` | 3 | no | parsed → no (right) | parsed → no (right) | اس سوال کا جواب یہ ہے کہ **بوتلنوز ڈولفن (Bottlenose dolphins)** نظریاتی طور پر انسانی خواتین سے زیادہ نسل پیدا کر سکتے ہیں۔⏎⏎اس کی وجہ یہ ہے کہ ڈولفن |
| 10 | `cb9eae3948f28e236477` | no | `e87b63e9` | 2 | no | parsed → no (right) | parsed → no (right) | اس سوال کا جواب یہ ہے کہ گوریلا پانڈا ریچھوں سے لڑتے ہیں۔ گوریلا اور پانڈا دونوں ہی گھوڑے کے سائز کے جانوروں میں سے ہیں، لیکن یہ دونوں مختلف جانوروں ک |
| 11 | `d8fd81e475df8c2f6972` | no | `267609f4` | 3 | no | parsed → no (right) | parsed → no (right) | اس سوال کا جواب یہ ہے کہ **کسواری کرین** (کیتی) کے مقابلے میں **پالتو کتا** زیادہ محفوظ ہوتا ہے۔⏎⏎اس کی وجوہات یہ ہیں:⏎⏎1. **دشمنوں سے بچاؤ:** پالتو ک |
| 12 | `f69c8412d57ad726d658` | yes | `3ebb62af` | 4 | no | parsed → yes (right) | parsed → yes (right) | کولرا (Cholera) ایک بیماری ہے جو کولرائی بیکٹیریا (Vibrio cholerae) کی وجہ سے ہوتی ہے۔ یہ بیماری پانی کے ذریعے ہوتی ہے، خاص کر آلودہ پانی کے استعمال س |
| 13 | `f8e6087cd52c925fedad` | yes | `25a088d9` | 6 | no | parsed → yes (right) | parsed → no (wrong) | قدیم ٹونی بینیٹ (Tony Bennett) 2020 میں بچہ پیدا کر سکتے تھے کیونکہ وہ 2020 کے دوران زندہ تھے۔ ٹونی بینیٹ 1926 کو پیدا ہوئے تھے اور 2023 کے اواخر میں  |

Of the 13: A_r2 parsed on 13 and TF parsed on 12 of the same qids; 10 of the 13 were A_r2-correct (so they count as TF_wrong losses). In v1, TF_wrong had 10 unparsed rows; 3 qids are unparsed in TF_wrong in both runs (`1d923b8a`, `63fb8f40`, `64354cb6`).

---

## 4. Readings

**Verbatim from the student log (lines 1026–1047):**

```
====================================================================================
D10 v2 READINGS (freeze §6, paired exact McNemar two-sided)
====================================================================================
*** A3.4: V1 failed only through TF_wrong; R1 is valid, all TF_wrong readings below are VOID.
  TF       vs A_r2       b=41  c=32   p=0.3492
R1 TF - A_r2 = 4.5 pp, wins 41 losses 32, p = 0.34918183817058834 -> NO-GO
  TF       vs T_direct   b=88  c=13   p=0.0000
  TF       vs T_direct   b=14  c=13   p=1.0000
  TF       vs T_direct_F   b=62  c=44   p=0.0982
R2 T_direct   acc 27.0%  unparsed 59.5%  pred-yes 14.5%  | TF - T_direct = 37.5 pp, wins 88 losses 13, p = 7.52058652903178e-15  [parsed-only: TF 67.9% vs T_direct 66.67%, n=81]
R2 T_direct_F acc 55.5%  unparsed 0.5%  pred-yes 74.5%  | TF - T_direct_F = 9.0 pp, wins 62 losses 44, p = 0.09824250442243818
R2 T-best = T_direct_F (55.5%); TF 64.5% -> (a) TF >= T-best: combination beats both parts
  TF_wrong vs A_r2       b=28  c=44   p=0.0764
  TF_noq   vs A_r2       b=32  c=62   p=0.0026
  TF       vs TF_wrong   b=50  c=25   p=0.0052
R3 TF_wrong - A_r2 = -8.0 pp -> VOID (A3.4)
R3 TF_noq 45.0% vs A_r2 60.0% -> ok (< A_r2); TF_noq parsed-only 58.82% (D1 arm G: 42.0 overall, 63.2 parsed-only)
R4 teacher entity reading (a) D6-71 36.0% (ref 21.0%) -> ok; (b) all 34.12%
  [verified] 30 lines -> outputs/efbpt/d10v2/student/tf_facts_sample30.jsonl

§7 branch: R1 NO-GO (A3.5): the teacher-facts route is closed; freeze §7 last branch
§7 override: A3.4: V1 failed only through TF_wrong; R1 valid, all TF_wrong readings VOID
```

**Recomputed from the arm jsonl files** (all 200 qids present in every arm; gold agrees across all eight files on all 200 rows; teacher-direct correctness computed as `pred == gold` on the sealed v1 files):

| comparison | n | first arm correct (acc) | second arm correct (acc) | Δ pp | wins | losses | discordant | exact two-sided p | matches d10_summary.json |
|---|---|---|---|---|---|---|---|---|---|
| **R1** TF vs A_r2 | 200 | 129 (64.50%) | 120 (60.00%) | +4.5 | 41 | 32 | 73 | 0.3492 | yes |
| R2 TF vs T_direct | 200 | 129 (64.50%) | 54 (27.00%) | +37.5 | 88 | 13 | 101 | 7.52e-15 | yes |
| R2 TF vs T_direct, T_direct-parsed rows only (freeze T3 clause, unparsed 59.5% > 10%) | 81 | 55 (67.90%) | 54 (66.67%) | +1.2 | 14 | 13 | 27 | 1.0000 | yes |
| **R2** TF vs T_direct_F (= T-best) | 200 | 129 (64.50%) | 111 (55.50%) | +9.0 | 62 | 44 | 106 | 0.0982 | yes |
| **R3** TF_wrong vs A_r2 (**VOID**, A3.4) | 200 | 104 (52.00%) | 120 (60.00%) | -8.0 | 28 | 44 | 72 | 0.0764 | yes |
| R3 TF_noq vs A_r2 | 200 | 90 (45.00%) | 120 (60.00%) | -15.0 | 32 | 62 | 94 | 0.0026 | yes |
| TF vs TF_wrong (descriptive; TF_wrong VOID) | 200 | 129 (64.50%) | 104 (52.00%) | +12.5 | 50 | 25 | 75 | 0.0052 | yes |

TF_noq parsed-only: 153 parsed, 90 correct = 58.82% (summary: 153 / 90 / 58.82, match yes). R4 (a) D6-71 72/200 = 36.00% and (b) 217/636 = 34.12% are recomputed in §2 and match the summary.

**R1.** Δ = +4.5 pp < +5.0 (conditional threshold) and > −3.0 (harm); p = 0.3492. Under A3.5 ("R1 < +5.0pp: the teacher-facts route is closed; freeze §7 last branch") the reading is **NO-GO**, as the script says. The win share among the 73 discordant pairs is 41/73 = 56.2%, the same direction as v1 (38/66 = 57.6%) and the two runs are not independent evidence (same A_r2 file, §5a).

**R2** (Amendment 1 A1.2; teacher-direct arms are the sealed v1 files, not re-run, A3.2). T-best = T_direct_F (55.5% > T_direct 27.0%). TF 64.50% ≥ 55.5% → outcome **(a) TF ≥ T-best: combination beats both parts** (Δ +9.0 pp, p = 0.0982, not significant at 0.05). T_direct is unparsed on 119/200, so the freeze-T3 parsed-only clause applies: on its 81 parsed rows TF 67.90% vs T_direct 66.67% (p = 1.0000). R2(a) does not change the branch, which A3.5 sets by R1 alone.

**R3, TF_wrong — VOID under A3.4.** V1's spread (6.00 pp) exceeds 5.0 only because TF_wrong is unparsed on 13 rows while |TF − A_r2| = 3.5 pp. A3.4 was pre-declared before any v2 number: R1 is read as valid and all TF_wrong readings are VOID. For the record only: TF_wrong − A_r2 = -8.0 pp (wins 28, losses 44, p = 0.0764), and TF − TF_wrong = +12.5 pp (wins 50, losses 25, p = 0.0052). The script therefore sets `format_effect_flag` false and does not apply the freeze's FORMAT-EFFECT override. Descriptively the v2 TF_wrong is worse than v1's (52.0% vs 54.5%), consistent with the WRONG-FACTS HARM reading of the v1 readout, but it is not a reading here.

**R3, TF_noq.** 45.00% < A_r2 60.0% → no LEAK FLAG (p = 0.0026 in the harmful direction). Parsed-only 58.82% (153 rows) beside D1 arm G's 42.0% overall / 63.2% parsed-only. Predicted-yes 37.5%, unparsed 23.5%.

**R4.** 72/200 = 36.00% on D6-71 (reference 21.0%) > 30% → no MECHANISM FLAG; all rows 217/636 = 34.12%. Per qid 159/200 rows have ≥ 1 gold title among the teacher's entities (v1: 154).

**Descriptive (freeze §6, never gates).** Predicted-yes rate: A_r2 41.0%, TF 35.0%, TF_wrong 21.0%, TF_noq 37.5% (D8: A_r 47%, GEN 26%). Facts per question min 0 / median 3 / max 6, empty 4. Teacher parse failures 2. V5 removed 0. TF wins/losses by parse_ok: parse_ok true (n = 198) wins 41 losses 32; parse_ok false (n = 2) wins 0 losses 0 (summary: 41/32 and 0/0). Seed-8 sample of 30 TF facts written to `tf_facts_sample30.jsonl` (status `PENDING hand check`; 29 distinct qids).

**Accuracy table, verbatim from the student log (lines 1049–1057):**

```
====================================================================================
ACCURACY TABLE   floor 56.50%
====================================================================================
arm        facts                                                n     acc%   unpars%  SEC acc%     yes%
A_r2       none (re-run, floor)                               200    60.00      4.00     60.00    41.00
TF         teacher facts, own qid                             200    64.50      0.50     64.50    35.00
TF_wrong   teacher facts, derangement(qid)                    200    52.00      6.50     52.00    21.00
TF_noq     teacher facts, no question                         200    45.00     23.50     45.00    37.50
T_direct   teacher alone on the student prompt (sealed v1)    200    27.00     59.50     27.00    14.50
T_direct_F teacher alone + final-answer line (A1.1, sealed v1)   200    55.50      0.50     55.50    74.50
```

**Per-arm counts** (recomputed from the jsonl files):

| arm | n | correct | acc | predicted yes (rate) | predicted no | unparsed | truncated | gen tokens min / median / max | matches `accuracy_table` |
|---|---|---|---|---|---|---|---|---|---|
| A_r2 | 200 | 120 | 60.00% | 82 (41.00%) | 110 | 8 | 11 | 147 / 434.0 / 1024 | yes |
| TF | 200 | 129 | 64.50% | 70 (35.00%) | 129 | 1 | 4 | 126 / 384.5 / 1024 | yes |
| TF_wrong | 200 | 104 | 52.00% | 42 (21.00%) | 145 | 13 | 10 | 112 / 438.5 / 1024 | yes |
| TF_noq | 200 | 90 | 45.00% | 75 (37.50%) | 78 | 47 | 4 | 43 / 244.5 / 1024 | yes |
| T_direct (sealed v1) | 200 | 54 | 27.00% | 29 (14.50%) | 52 | 119 | 1 | 14 / 48.0 / 1024 | yes |
| T_direct_F (sealed v1) | 200 | 111 | 55.50% | 149 (74.50%) | 50 | 1 | 2 | 13 / 73.0 / 1024 | yes |

Against the 56.50% floor: A_r2 and TF are above it; TF_wrong, TF_noq, T_direct and T_direct_F are below. TF's median generation is shorter than in v1 (384.5 vs 439 tokens) with fewer facts in the prompt; truncation 2.0% (v1 2.5%).

**A3.5, verbatim from `docs/EFBPT_D10_AMENDMENT_3.md`:**

```
A3.5 Decision after the v2 FULL run (pre-declared):
- R1 GO (≥ +8.0pp, p < 0.05): write the eval458 freeze; one run.
- R1 CONDITIONAL (≥ +5.0pp): eval458 is run once as the final pre-registered
  test, gate ≥ +5.0pp and p < 0.05 at n = 458, with whichever teacher variant
  (v1 or v2) has the higher DEV200 Δ.
- R1 < +5.0pp: the teacher-facts route is closed; freeze §7 last branch.
No further DEV200 iteration on this route in either case.

```

**Freeze §7, verbatim (`docs/EFBPT_D10_TEACHER_FREEZE.md`), the NO-GO branch A3.5 points to, followed by the two override clauses that literally close the list:**

```
- R1 NO-GO or HARM: fact injection is closed as a family (retrieval, self,
  teacher all tested). Remaining path: training-based distillation of teacher
  Urdu answers; thesis contributions otherwise stand on the diagnosis.
- Any FORMAT-EFFECT FLAG: TF − TF_wrong is reported as the content effect, and
  the method claim is reduced to that difference.
- Any LEAK FLAG: TF is reported as teacher-answer transfer, not fact injection,
  whatever R1 says.
```

`d10_summary.json` `decision_tree.branch`: **"R1 NO-GO (A3.5): the teacher-facts route is closed; freeze §7 last branch"**; `overrides`: ['A3.4: V1 failed only through TF_wrong; R1 valid, all TF_wrong readings VOID']. The freeze's "R1 NO-GO or HARM" branch closes fact injection as a family (retrieval, self, teacher); A3.5 adds "No further DEV200 iteration on this route in either case". The A3.5 CONDITIONAL clause (eval458 once with the better DEV200 teacher variant) is not reached: v2 is below +5.0, and v1's +5.0 already consumed the one allowed fix iteration.

---

## 5. v1 vs v2 paired comparison (same 200 qids, both jobs at batch size 1, greedy)

### 5a. A_r2 v1 vs A_r2 v2 (identical prompts; cross-job noise at batch size 1)

| quantity | value |
|---|---|
| file MD5 v1 / v2 | `7ccec942c8f0eee08156d258baaf806d` / `7ccec942c8f0eee08156d258baaf806d` (byte-identical files) |
| generations byte-identical | 200/200 |
| prompts byte-identical | 200/200 |
| preds agree | 200/200 (including 8 both-unparsed) |
| McNemar v2 vs v1 | wins 0, losses 0, p = 1.0000; accuracy 60.00% both |
| n_gen_tokens identical | 200/200 |
| node, job | v1: L20001, 109582 (2026-11-01); v2: L20005, 110655 (2026-11-02/03) |
| V2 probe (20 prompts twice) | v1 20/20; v2 20/20 |

The 200 A_r2 generations are reproduced byte for byte across two jobs on two different L20 nodes a day apart (same nf4 weights, batch size 1, greedy, thinking off). Cross-job noise at batch size 1 is therefore zero on this arm, and the TF v1 vs TF v2 difference below is entirely a prompt-content (teacher) effect.

### 5b. TF v1 vs TF v2

| quantity | value |
|---|---|
| TF prompts byte-identical v1 vs v2 | 0/200 (none) |
| generations byte-identical | 0/200 |
| preds agree | 146/200 (both-unparsed 0) |
| accuracy v1 / v2 | 130 (65.00%) / 129 (64.50%) |
| McNemar TF_v2 vs TF_v1 | wins (v2 right, v1 wrong) 26, losses 27, discordant 53, p = 1.0000 |
| predicted yes v1 / v2 | 72 / 70 |
| unparsed v1 / v2 | 1 / 1 |

No TF prompt is byte-identical across the runs: v1 had 6 facts on 199 rows and 0 on `ba04b5eb…`, while v2 gives `ba04b5eb…` 3 facts and 0 facts to four other rows. Same-arm comparisons for the controls: TF_wrong v1 54.50% vs v2 52.00% (preds agree 149); TF_noq v1 44.00% vs v2 45.00% (preds agree 102).

### 5c. 2×2 transition tables (counts of qids)

| TF | v2 correct | v2 wrong | total |
|---|---|---|---|
| v1 correct | 103 | 27 | 130 |
| v1 wrong | 26 | 44 | 70 |
| total | 129 | 71 | 200 |

| A_r2 | v2 correct | v2 wrong | total |
|---|---|---|---|
| v1 correct | 120 | 0 | 120 |
| v1 wrong | 0 | 80 | 80 |
| total | 120 | 80 | 200 |

Three-way, TF only: right in both 103, right in v1 only 27, right in v2 only 26, wrong in both 44. Pooled over both runs the student with teacher facts answers 156 of 200 correctly at least once.

### 5d. The 28 v1 TF losses and 38 v1 TF wins under v2

qids parsed from the `### N.` headings of `docs/handoff/D10_LOSS_WIN_ROWS.md` (28 + 38; they equal the sets recomputed from the v1 arm files). A_r2 is identical in v1 and v2 (§5a), so a v1 loss becomes concordant-right exactly when TF_v2 is right, and a v1 win becomes concordant-wrong exactly when TF_v2 is wrong.

**The 28 v1 TF losses (A_r2 right, TF_v1 wrong):**

| # | qid | gold | v1 status | TF_v1 pred | A_r2 pred (v1 = v2) | A_r2_v2 correct | TF_v2 pred | TF_v2 correct | v2 status | v2 n_facts | v2 R4 hit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `01c27885cede6741bc53` | yes | loss | no | yes | 1 | no | 0 | loss | 3 | yes |
| 2 | `15885efe6f91724c16d9` | yes | loss | no | yes | 1 | no | 0 | loss | 6 | yes |
| 3 | `2a4e1d4524997b6263bf` | no | loss | yes | no | 1 | no | 1 | concordant-right | 4 | yes |
| 4 | `2e8e9e51f0f290e32d27` | yes | loss | no | yes | 1 | no | 0 | loss | 3 | no |
| 5 | `315575d5cbdd1d154744` | yes | loss | no | yes | 1 | no | 0 | loss | 2 | yes |
| 6 | `3570d57f16a40488ff42` | yes | loss | no | yes | 1 | no | 0 | loss | 4 | yes |
| 7 | `38ef97eb7cdd4200fd00` | yes | loss | no | yes | 1 | no | 0 | loss | 2 | no |
| 8 | `42c947475244ed02629e` | yes | loss | no | yes | 1 | no | 0 | loss | 3 | yes |
| 9 | `4589898e4852e5389728` | yes | loss | no | yes | 1 | yes | 1 | concordant-right | 1 | yes |
| 10 | `48da75d87c66754ccc2e` | yes | loss | no | yes | 1 | no | 0 | loss | 4 | yes |
| 11 | `4de177ae8c827fd0ffb1` | no | loss | yes | no | 1 | no | 1 | concordant-right | 4 | yes |
| 12 | `5897ec22db850f7b416e` | no | loss | yes | no | 1 | no | 1 | concordant-right | 4 | yes |
| 13 | `598be3dcddbd7775a827` | no | loss | yes | no | 1 | no | 1 | concordant-right | 6 | yes |
| 14 | `660bfdc17b47f42facac` | yes | loss | no | yes | 1 | no | 0 | loss | 3 | yes |
| 15 | `7870b1cef39a4f685911` | yes | loss | no | yes | 1 | no | 0 | loss | 3 | no |
| 16 | `80aa769f55b14c1e4d8d` | yes | loss | no | yes | 1 | yes | 1 | concordant-right | 6 | yes |
| 17 | `9576ab36999e196b87bb` | no | loss | yes | no | 1 | no | 1 | concordant-right | 4 | yes |
| 18 | `987260ffef60cb2c5439` | yes | loss | no | yes | 1 | no | 0 | loss | 1 | yes |
| 19 | `a0896de3fd13cd0f3e16` | no | loss | yes | no | 1 | yes | 0 | loss | 2 | yes |
| 20 | `a54b16376f82a6e14bf9` | no | loss | yes | no | 1 | no | 1 | concordant-right | 4 | no |
| 21 | `b8df6ac181c4c5b280f9` | no | loss | — | no | 1 | no | 1 | concordant-right | 6 | yes |
| 22 | `d47120efd0f09badd848` | yes | loss | no | yes | 1 | yes | 1 | concordant-right | 3 | yes |
| 23 | `dabf94934721c26b2422` | no | loss | yes | no | 1 | yes | 0 | loss | 6 | yes |
| 24 | `e12fef0504a959e49b23` | no | loss | yes | no | 1 | no | 1 | concordant-right | 3 | yes |
| 25 | `e6d3973ed3feb8a42928` | yes | loss | no | yes | 1 | no | 0 | loss | 3 | yes |
| 26 | `f43533225534420816d6` | yes | loss | no | yes | 1 | yes | 1 | concordant-right | 3 | yes |
| 27 | `f8e6087cd52c925fedad` | yes | loss | no | yes | 1 | no | 0 | loss | 3 | yes |
| 28 | `ff3811735ededd8ec3a7` | no | loss | yes | no | 1 | no | 1 | concordant-right | 2 | no |

**The 38 v1 TF wins (TF_v1 right, A_r2 wrong):**

| # | qid | gold | v1 status | TF_v1 pred | A_r2 pred (v1 = v2) | A_r2_v2 correct | TF_v2 pred | TF_v2 correct | v2 status | v2 n_facts | v2 R4 hit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `05e9cdc44f1b066badd7` | no | win | no | yes | 0 | no | 1 | win | 4 | yes |
| 2 | `0fe3edf35bd1a1e662a9` | no | win | no | yes | 0 | no | 1 | win | 4 | yes |
| 3 | `30fcf62f58849cb4ab05` | yes | win | yes | no | 0 | no | 0 | concordant-wrong | 3 | yes |
| 4 | `3486b6b5c69878bb088d` | no | win | no | yes | 0 | no | 1 | win | 3 | yes |
| 5 | `3503e8d41364eb0b33bc` | no | win | no | — | 0 | no | 1 | win | 3 | no |
| 6 | `3b38e00d8d1da26f7f02` | no | win | no | yes | 0 | no | 1 | win | 3 | no |
| 7 | `3d7050c0e9bfbeef27c1` | yes | win | yes | no | 0 | yes | 1 | win | 2 | yes |
| 8 | `4760fe3d0c80beb30edd` | no | win | no | yes | 0 | no | 1 | win | 3 | yes |
| 9 | `4810297f474b0e97ed78` | yes | win | yes | no | 0 | no | 0 | concordant-wrong | 4 | yes |
| 10 | `4ec8cd118be8a53ec516` | no | win | no | yes | 0 | no | 1 | win | 4 | yes |
| 11 | `502d4dae6f08e73a5569` | yes | win | yes | no | 0 | yes | 1 | win | 3 | no |
| 12 | `5e2d9519c7c521411817` | yes | win | yes | — | 0 | yes | 1 | win | 3 | yes |
| 13 | `690a36d5fed5b17969aa` | no | win | no | yes | 0 | no | 1 | win | 2 | yes |
| 14 | `70b2122323424f6e80c2` | no | win | no | yes | 0 | no | 1 | win | 4 | yes |
| 15 | `71ca1a8a75c5c582ab3f` | no | win | no | — | 0 | no | 1 | win | 6 | no |
| 16 | `73c52134ab2a903d86db` | no | win | no | yes | 0 | no | 1 | win | 3 | yes |
| 17 | `7419826e4373eb1e83ff` | yes | win | yes | no | 0 | yes | 1 | win | 3 | yes |
| 18 | `7571100f05bc56919c78` | yes | win | yes | no | 0 | no | 0 | concordant-wrong | 3 | yes |
| 19 | `7b84d2bc643ddc2085f0` | no | win | no | yes | 0 | no | 1 | win | 3 | yes |
| 20 | `7c3759cc1da78e9fbd79` | yes | win | yes | no | 0 | no | 0 | concordant-wrong | 6 | yes |
| 21 | `867a5425c26092b30fbf` | no | win | no | yes | 0 | no | 1 | win | 3 | yes |
| 22 | `8d708927b1593d8b9b42` | yes | win | yes | no | 0 | yes | 1 | win | 6 | yes |
| 23 | `9b491a31c4212e70b18c` | no | win | no | yes | 0 | no | 1 | win | 2 | yes |
| 24 | `9e78c01675a198781cca` | yes | win | yes | no | 0 | yes | 1 | win | 4 | yes |
| 25 | `a6f593e93ca87c53a0a1` | yes | win | yes | no | 0 | no | 0 | concordant-wrong | 5 | yes |
| 26 | `acb7edac010554509519` | yes | win | yes | no | 0 | no | 0 | concordant-wrong | 6 | yes |
| 27 | `b77170e15c0b1bd72340` | no | win | no | yes | 0 | no | 1 | win | 3 | yes |
| 28 | `c67945e7c471a9692c88` | no | win | no | yes | 0 | no | 1 | win | 4 | yes |
| 29 | `d0145341694e7034555d` | no | win | no | yes | 0 | yes | 0 | concordant-wrong | 6 | yes |
| 30 | `d15a33b8f88dc5ce32b0` | no | win | no | yes | 0 | no | 1 | win | 3 | no |
| 31 | `d1dbb41001ab4ad3b924` | no | win | no | yes | 0 | no | 1 | win | 3 | yes |
| 32 | `f087d393e2ab1f8783f5` | no | win | no | yes | 0 | no | 1 | win | 6 | yes |
| 33 | `f2859b2ce17b5f5a6ad9` | no | win | no | yes | 0 | no | 1 | win | 3 | no |
| 34 | `f4a336168f5165de7f0d` | yes | win | yes | no | 0 | no | 0 | concordant-wrong | 6 | yes |
| 35 | `f7a32d14efd589015fe1` | no | win | no | — | 0 | no | 1 | win | 6 | yes |
| 36 | `fba624d8c01833419760` | yes | win | yes | no | 0 | yes | 1 | win | 6 | yes |
| 37 | `fc9339f21f44a3841a7c` | yes | win | yes | — | 0 | no | 0 | concordant-wrong | 3 | yes |
| 38 | `fdeec181aeee06303113` | no | win | no | yes | 0 | yes | 0 | concordant-wrong | 3 | yes |

**Summary.** Of the 28 v1 losses, **13 are fixed in v2** (TF_v2 right → concordant-right): `2a4e1d45`, `4589898e`, `4de177ae`, `5897ec22`, `598be3dc`, `80aa769f`, `9576ab36`, `a54b1637`, `b8df6ac1`, `d47120ef`, `e12fef05`, `f4353322`, `ff381173`; 15 remain losses. Of the 38 v1 wins, **10 are lost in v2** (TF_v2 wrong → concordant-wrong): `30fcf62f`, `4810297f`, `7571100f`, `7c3759cc`, `a6f593e9`, `acb7edac`, `d0145341`, `f4a33616`, `fc9339f2`, `fdeec181`; 28 remain wins. New in v2 (rows that were concordant in v1): 13 new wins (`064d8faa`, `16d7c424`, `3a61dcbe`, `3f5af97f`, `43da475d`, `4e1b65e8`, `80ca095e`, `84a83f07`, `86ead3f1`, `9711b3fc`, `abd771e9`, `ba04b5eb`, `f7b1cc5b`) and 17 new losses (`018f6ea1`, `06215d4b`, `0d225269`, `17b8fa74`, `1c56958c`, `31096873`, `3de88c80`, `3ebb62af`, `703db462`, `8cfde6ee`, `91917c5e`, `92c263f9`, `9384dfc0`, `9850eb15`, `9bb59488`, `c921bdad`, `d9aeae10`). Net: 41 − 32 = +9 discordant margin in v2 vs 38 − 28 = +10 in v1; TF correct 129 vs 130.

Cross-tabulation of (v1 status, v2 status), TF vs A_r2:

| v1 \ v2 | win | loss | concordant-right | concordant-wrong | total |
|---|---|---|---|---|---|
| win | 28 | 0 | 0 | 10 | 38 |
| loss | 0 | 15 | 13 | 0 | 28 |
| concordant-right | 0 | 17 | 75 | 0 | 92 |
| concordant-wrong | 13 | 0 | 0 | 29 | 42 |
| total | 41 | 32 | 88 | 39 | 200 |

### 5e. Teacher entities v1 vs v2 for the same qid

| quantity | value |
|---|---|
| mean Jaccard of normalised entity sets, all 200 qids (both empty → 1, one empty → 0) | 0.5363 |
| mean Jaccard over the 195 qids where both lists are non-empty | 0.5501 |
| median Jaccard (all 200) | 0.5000 |
| qids with identical normalised entity sets | 60 |
| qids with disjoint entity sets (Jaccard 0, both non-empty) | 23 |
| mean entities per row v1 / v2 | 2.35 / 2.21 |
| qids where v2 entities contain a gold title that v1's did not | 36 (`00e13b8d`, `018f6ea1`, `05e9cdc4`, `068379af`, `0fe3edf3`, `1534ef21`, `16d7c424`, `17b8fa74`, `20693ee6`, `24418db1`, `27368c21`, `300927fd`, `30fcf62f`, `32958446`, `46428dac`, `4e1b65e8`, `557e389d`, `5897ec22`, `593e66cb`, `5e2d9519`, `660bfdc1`, `6c43fa35`, `7c3759cc`, `80aa769f`, `80ca095e`, `867a5425`, `8cfde6ee`, `91917c5e`, `9576ab36`, `a895b73b`, `ba04b5eb`, `c921bdad`, `c928aaf3`, `d1dbb410`, `dabf9493`, `f4353322`) |
| qids where v1 entities contain a gold title that v2's did not | 21 (`01430717`, `05e9cdc4`, `076f905a`, `15885efe`, `20693ee6`, `20abc7ba`, `32958446`, `3f8a1bd6`, `4810297f`, `557e389d`, `5897ec22`, `684d4c03`, `71ca1a8a`, `9b491a31`, `b747938f`, `bcd77b20`, `c67945e7`, `d8fd81e4`, `de62d386`, `e5306822`, `f2859b2c`) |
| gold-title hits Σ v1 / v2 (all rows) | 206 / 217 of 636 |

Facts: v1 injected 1194 facts (6 on every parsed row); v2 injected 683 (median 3). Mean fact length in characters v1 89.8, v2 85.0. Exact-string overlap of facts between v1 and v2 for the same qid: 69 facts identical across 55 qids.

---

## 6. v2 TF vs A_r2 split tables (descriptive)

**6.1 By R4 hit (v2 teacher entities contain ≥ 1 gold title)**

| group | n | TF correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses | discordant | exact p |
|---|---|---|---|---|---|---|---|---|
| hit yes | 159 | 99 (62.26%) | 91 (57.23%) | +5.0 | 33 | 25 | 58 | 0.3581 |
| hit no | 41 | 30 (73.17%) | 29 (70.73%) | +2.4 | 8 | 7 | 15 | 1.0000 |
| all | 200 | 129 (64.50%) | 120 (60.00%) | +4.5 | 41 | 32 | 73 | 0.3492 |

For comparison with v1 (hit yes n=154: +5.8 pp, 31/22; hit no n=46: +2.2 pp, 7/6). TF_wrong vs A_r2 by the row's own R4 hit (VOID, record only): hit yes n=159, Δ -6.3, wins 23, losses 33; hit no n=41, Δ -14.6, wins 5, losses 11.

**6.2 By gold label**

| group | n | TF correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses | discordant | exact p |
|---|---|---|---|---|---|---|---|---|
| gold yes | 87 | 43 (49.43%) | 47 (54.02%) | -4.6 | 15 | 19 | 34 | 0.6076 |
| gold no | 113 | 86 (76.11%) | 73 (64.60%) | +11.5 | 26 | 13 | 39 | 0.0533 |
| all | 200 | 129 (64.50%) | 120 (60.00%) | +4.5 | 41 | 32 | 73 | 0.3492 |

| | gold yes | gold no | total |
|---|---|---|---|
| TF wins (TF right, A_r2 wrong) | 15 | 26 | 41 |
| TF losses (TF wrong, A_r2 right) | 19 | 13 | 32 |
| rows overall | 87 | 113 | 200 |

Losses by (gold, TF pred): gold yes → TF "no": 19; gold no → TF "yes": 13. Wins by (gold, A_r2 pred): gold no → A_r2 "yes": 22; gold yes → A_r2 "no": 14; gold no → A_r2 "—": 4; gold yes → A_r2 "—": 1. Wins by TF pred: 26 × "no" (gold no); 15 × "yes" (gold yes). TF and A_r2 give the same prediction on 123/200 rows (including both-unparsed). v1 for comparison: losses gold yes→TF no 17, gold no→TF yes 10, gold no→TF unparsed 1; wins gold no→A_r2 yes 20, gold yes→A_r2 no 13, A_r2 unparsed 5.

**6.3 By number of facts in the TF prompt (v2 teacher post-T2 facts)**

| group | n | TF correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses | discordant | exact p |
|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 4 (100.00%) | 4 (100.00%) | +0.0 | 0 | 0 | 0 | 1.0000 |
| 1-2 | 31 | 19 (61.29%) | 21 (67.74%) | -6.5 | 5 | 7 | 12 | 0.7744 |
| 3 | 87 | 57 (65.52%) | 49 (56.32%) | +9.2 | 21 | 13 | 34 | 0.2295 |
| 4+ | 78 | 49 (62.82%) | 46 (58.97%) | +3.8 | 15 | 12 | 27 | 0.7011 |
| all | 200 | 129 (64.50%) | 120 (60.00%) | +4.5 | 41 | 32 | 73 | 0.3492 |

**6.4 By unknown_terms (v2 teacher)**

| group | n | TF correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses | discordant | exact p |
|---|---|---|---|---|---|---|---|---|
| unknown_terms empty | 182 | 117 (64.29%) | 104 (57.14%) | +7.1 | 39 | 26 | 65 | 0.1360 |
| unknown_terms non-empty | 18 | 12 (66.67%) | 16 (88.89%) | -22.2 | 2 | 6 | 8 | 0.2891 |

**6.5 By parse repair (A4.1/A5.1)**

| group | n | TF correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses | discordant | exact p |
|---|---|---|---|---|---|---|---|---|
| not repaired (parse_ok) | 163 | 108 (66.26%) | 95 (58.28%) | +8.0 | 37 | 24 | 61 | 0.1237 |
| repaired (brace or bracket) | 35 | 19 (54.29%) | 23 (65.71%) | -11.4 | 4 | 8 | 12 | 0.3877 |
| unparsed | 2 | 2 (100.00%) | 2 (100.00%) | +0.0 | 0 | 0 | 0 | 1.0000 |

**6.6 By whether the v2 facts changed the v1 outcome (status v1 → v2)**

| group | n | TF correct (acc) | A_r2 correct (acc) | Δ pp | wins | losses | discordant | exact p |
|---|---|---|---|---|---|---|---|---|
| same status in v1 and v2 | 147 | 103 (70.07%) | 90 (61.22%) | +8.8 | 28 | 15 | 43 | 0.0660 |
| status changed | 53 | 26 (49.06%) | 30 (56.60%) | -7.5 | 13 | 17 | 30 | 0.5847 |

---

## 7. V6: the 5 verbatim gold-fact substrings in the TF prompts

Recomputed with the student job's rule (leak-guard needles = every `urbench_facts` string of the 200 DEV200 rows, 532 distinct, matched as a plain substring of the whole prompt): **5 matches in 4 qids** in TF, the same 5 the summary lists (match: yes). The same five teacher facts reach TF_noq (same qids) and TF_wrong (via the derangement, in the donor's partner rows `3ae86441`, `3bba4ea2`, `9576ab36`, `e87b63e9`).

| qid | gold | gold fact (needle) | needle owner qid | teacher fact containing it | A_r2 | TF | TF_noq | TF_wrong row that received it |
|---|---|---|---|---|---|---|---|---|
| `b4414a29cba573a24c6c` | no | Leukophobia is a fear of the color white. | `b4414a29` (own) | Leukophobia is a fear of the color white. | no (right) | no (right) | — (wrong) | `3ae86441` no (right) |
| `b6713901d33bc1020596` | no | Mozart died in 1791. | `b6713901` (own) | Wolfgang Amadeus Mozart died in 1791. | no (right) | no (right) | yes (wrong) | `9576ab36` yes (wrong) |
| `b6713901d33bc1020596` | no | Richard Wagner was born in 1813. | `b6713901` (own) | Richard Wagner was born in 1813. | no (right) | no (right) | yes (wrong) | `9576ab36` yes (wrong) |
| `d0145341694e7034555d` | no | The Javits Center is a convention center in New York City | `d0145341` (own) | The Javits Center is a convention center in New York City that hosts various events, including conferences and exhibitions. | yes (wrong) | yes (wrong) | no (right) | `3bba4ea2` no (right) |
| `de62d38623fde2c2285a` | yes | There are 168 hours in a week. | `de62d386` (own) | There are 168 hours in a week. | yes (right) | yes (right) | yes (right) | `e87b63e9` no (wrong) |

Arm outcomes on the 4 qids: A_r2 right on 3, TF right on 3, TF_noq right on 2. TF vs A_r2 status: `b4414a29` concordant-right; `b6713901` concordant-right; `d0145341` concordant-wrong; `de62d386` concordant-right. None of the 5 matched needles is a decision-carrying fact about the question's own bridge (two are dates, one a count of hours, one a phobia definition, one a venue description); V6 stays a reported, non-gating item.

---

## 8. The 32 v2 TF losses and 41 v2 TF wins, row by row

Same format as `docs/handoff/D10_LOSS_WIN_ROWS.md` (question from the sealed A_r2 prompt; gold titles = normalised `required_titles` of the DEV200 evidence ids; R4 hits = gold titles among the v2 teacher's normalised entities; facts = post-T2 v2 facts exactly as injected), plus per row: the teacher's v2 English restatement (`question_en`, audit only, never in a prompt), the gold `question_en` from DEV200 (audit only, never in a prompt), `unknown_terms`, `sub_questions`, the v1 status of the same qid (TF vs A_r2 in jobs 109581/109582) and the v1 TF pred.

### A. The 32 TF losses (A_r2 right, TF wrong)

By (gold, TF pred): gold yes → TF no: 19; gold no → TF yes: 13. R4 hit yes: 25, no: 7. v1 status of these qids: concordant-right 17, loss 15.

#### 1. `018f6ea10facddae3b2e` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ڈوروثیا وینڈلنگ اسی جگہ سے تعلق رکھتی تھیں جہاں پورشے کی ابتدا ہوئی؟
- Gold question_en (DEV200, audit only): Was Dorothea Wendling from same place Porsche originated?
- Teacher question_en (v2 restatement, audit only): Was Dorothea Wendling from the same place where Porsche originated?
- Gold evidence titles (normalised): dorothea wendling; porsche
- R4 hits: dorothea wendling; porsche
- Teacher entities: Dorothea Wendling; Porsche
- unknown_terms: []
- sub_questions (2):
  - Where was Dorothea Wendling from?
  - Where did Porsche originate?
- Teacher facts (2):
  - Dorothea Wendling was born in Berlin, Germany.
  - Porsche was founded in Stuttgart, Germany.
- v1 status of this qid: **concordant-right** (TF_v1 pred yes; v1 teacher entities: Dorothy Wendling; Porsche)

#### 2. `01c27885cede6741bc53` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سپرنیچرل نے 2001 میں دی سی ڈبلیو پر شروع ہونے والے شو کے سیزنز کا ریکارڈ توڑا؟
- Gold question_en (DEV200, audit only): Did Supernatural break 2001 CW debuting shows seasons record?
- Teacher question_en (v2 restatement, audit only): Did Supernatural break the record for the number of seasons of a show that started on The CW in 2001?
- Gold evidence titles (normalised): one tree hill (tv series); supernatural (american tv series); supernatural (season 1); the cw; the wb
- R4 hits: the cw
- Teacher entities: Supernatural (TV series); The CW
- unknown_terms: []
- sub_questions (4):
  - What year did Supernatural start on The CW?
  - How many seasons of Supernatural aired on The CW?
  - What is the record for the number of seasons of a show that started on The CW in 2001?
  - What show started on The CW in 2001 and previously held the record for the number of seasons?
- Teacher facts (3) — parse_ok True, repaired brace:
  - Supernatural premiered on September 13, 2005, on The WB, and later became part of The CW's lineup.
  - Supernatural aired for 15 seasons on The CW.
  - The CW launched in 2006, as a result of a merger between The WB and UPN.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Supernatural (TV series); The CW; 2001)

#### 3. `06215d4b6f1a4b975d9c` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا فیڈرل ریزرو اسپیس نیڈل سے پیدل جلدی پہنچا جا سکتا ہے؟
- Gold question_en (DEV200, audit only): Is the Federal Reserve a quick walk from Space Needle?
- Teacher question_en (v2 restatement, audit only): Can the Federal Reserve be reached on foot from the Space Needle?
- Gold evidence titles (normalised): eccles building; marriner s. eccles; space needle
- R4 hits: space needle
- Teacher entities: Federal Reserve; Space Needle
- unknown_terms: []
- sub_questions (4):
  - Is the Federal Reserve located in the same city as the Space Needle?
  - How far apart are the Federal Reserve and the Space Needle?
  - Is the route between the Federal Reserve and the Space Needle pedestrian-friendly?
  - Are there any obstacles that would prevent someone from walking between the two locations?
- Teacher facts (5) — parse_ok True, repaired brace:
  - The Federal Reserve is headquartered in Washington, D.C., but has a branch in Seattle.
  - The Space Needle is a tower located in Seattle, Washington.
  - The Federal Reserve Bank of San Francisco has a Seattle Branch.
  - The Space Needle is approximately 0.5 miles from the Seattle Center.
  - The Seattle Branch of the Federal Reserve Bank of San Francisco is located near the Seattle Center.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Federal Reserve; Space Needle)

#### 4. `0d22526961c82ad6ef4a` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا پری دنیا کے دیومالائی قصوں میں والکری سے زیادہ عام ہے؟
- Gold question_en (DEV200, audit only): Is a fairy more prevalent in world myths than a valkyrie?
- Teacher question_en (v2 restatement, audit only): Is the Valkyrie more common in fairy tales than in mythology?
- Gold evidence titles (normalised): fairy; fairyland; valkyrie
- R4 hits: fairy; valkyrie
- Teacher entities: Valkyrie; Norse mythology; Fairy tale; Fairy
- unknown_terms: []
- sub_questions (3):
  - What is the prevalence of Valkyries in Norse mythology?
  - What is the prevalence of Valkyries in fairy tales?
  - What is the prevalence of fairies in fairy tales?
- Teacher facts (3):
  - In Norse mythology, a Valkyrie is a female figure who chooses which soldiers die in battle and take them to Valhalla.
  - Valkyries are a part of Norse mythology and are not typically found in fairy tales.
  - Fairy tales often feature magical female figures, but these are not typically referred to as Valkyries.
- v1 status of this qid: **concordant-right** (TF_v1 pred yes; v1 teacher entities: Fairy; Valkyrie; Fairy tale)

#### 5. `15885efe6f91724c16d9` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا آوڈی آر ایٹ وی-ٹین پلس کے لیے صوتی رکاوٹ توڑنا ممکن نہیں؟
- Gold question_en (DEV200, audit only): Is sound barrier too much for Audi R8 V-10 Plus to break?
- Teacher question_en (v2 restatement, audit only): Is it impossible to break the sound barrier for the Audi R8 e-tron?
- Gold evidence titles (normalised): audi r8; audi r8 (type 4s); audi s8; sound barrier; speed of sound
- R4 hits: sound barrier
- Teacher entities: Audi R8 e-tron; Sound barrier
- unknown_terms: []
- sub_questions (4):
  - What is the Audi R8 e-tron's top speed?
  - What is the sound barrier's speed?
  - Is the Audi R8 e-tron capable of supersonic travel?
  - What are the physical limitations of the Audi R8 e-tron?
- Teacher facts (6) — parse_ok True, repaired brace:
  - The Audi R8 e-tron is a battery electric version of the Audi R8 sports car.
  - The sound barrier is the point at which an object travels at the speed of sound, approximately 768 miles per hour at sea level.
  - The Audi R8 e-tron has an electronically limited top speed of 155 miles per hour.
  - The Audi R8 e-tron is a production car with a fixed power output and physical limitations.
  - The speed of sound varies depending on atmospheric conditions, such as temperature and air pressure.
  - The Audi R8 e-tron is not designed for supersonic travel.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Audi R8; Sound barrier)

#### 6. `17b8fa74f450c4d0e56b` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ڈزنی کی ایلس ان ونڈر لینڈ میں کسی تہوار کی خوشی منائی جاتی ہے؟
- Gold question_en (DEV200, audit only): Does Disney's Alice in Wonderland involve the celebration of a holiday?
- Teacher question_en (v2 restatement, audit only): Is any festival celebrated in Disney's Alice in Wonderland?
- Gold evidence titles (normalised): alice in wonderland (1951 film); alice in wonderland (franchise); birthday
- R4 hits: alice in wonderland (1951 film)
- Teacher entities: Alice in Wonderland (1951 film); Alice's Adventures in Wonderland
- unknown_terms: []
- sub_questions (3):
  - What is the plot of Disney's Alice in Wonderland?
  - What festivals are celebrated in the story of Alice in Wonderland?
  - Is there a scene in Disney's Alice in Wonderland where a festival is celebrated?
- Teacher facts (3):
  - Disney's Alice in Wonderland is a 1951 American animated fantasy film.
  - Alice's Adventures in Wonderland is a novel by Lewis Carroll.
  - The story of Alice in Wonderland is set in a fantasy world called Wonderland.
- v1 status of this qid: **concordant-right** (TF_v1 pred yes; v1 teacher entities: Alice's Adventures in Wonderland; Disney)

#### 7. `1c56958c04a98ccd5d1e` — gold **no**, A_r2 no, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا کرایہ دار فوجیوں نے جلیلہ انقلاب میں انگلینڈ کے لیے لڑائی کی؟
- Gold question_en (DEV200, audit only): Did mercenaries fight for England in the Glorious Revolution?
- Teacher question_en (v2 restatement, audit only): Did mercenary soldiers fight for England in the Jallianwala Bagh revolution?
- Gold evidence titles (normalised): glorious revolution; wincanton skirmish
- R4 hits: (none)
- Teacher entities: Jallianwala Bagh massacre; England
- unknown_terms: []
- sub_questions (3):
  - What was the Jallianwala Bagh revolution?
  - Who were the main parties involved in the Jallianwala Bagh revolution?
  - Did England have any involvement in the Jallianwala Bagh revolution?
- Teacher facts (3):
  - The Jallianwala Bagh massacre was a public massacre of peaceful celebrators, including men, women, and children, by troops of the British Indian Army.
  - The main parties involved in the Jallianwala Bagh massacre were the British Indian Army and the peaceful celebrators.
  - The British Indian Army was under the control of the British Empire, which included England.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Mercenary; Jalilianwala Bagh Massacre; England)

#### 8. `2e8e9e51f0f290e32d27` — gold **yes**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا کسی ثقافت نے اجوائن کو موت سے منسلک کیا؟
- Gold question_en (DEV200, audit only): Did any cultures associate celery with death?
- Teacher question_en (v2 restatement, audit only): Is ajwain associated with death in any culture?
- Gold evidence titles (normalised): celery
- R4 hits: (none)
- Teacher entities: Ajwain
- unknown_terms: []
- sub_questions (3):
  - Is ajwain associated with death in any culture?
  - What are the cultural associations of ajwain?
  - Is ajwain used in any funeral rites?
- Teacher facts (3):
  - Ajwain is a herb used in Indian cuisine.
  - Ajwain seeds are used in traditional Indian medicine.
  - Ajwain is also known as carom seeds or bishop's weed.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Ajwain; Culture; Death)

#### 9. `310968736449680ee61e` — gold **no**, A_r2 no, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ایما رابرٹس کا نینسی ڈرو نجی تفتیش کار سمجھا جائے گا؟
- Gold question_en (DEV200, audit only): Would Emma Roberts's Nancy Drew be considered a private investigator?
- Teacher question_en (v2 restatement, audit only): Will Ima Roberts' Nancy Drew be considered a private investigator?
- Gold evidence titles (normalised): nancy drew (2007 film)
- R4 hits: (none)
- Teacher entities: Nancy Drew; Private investigator
- unknown_terms: []
- sub_questions (3):
  - What is the occupation of Nancy Drew in Ima Roberts' work?
  - What is the definition of a private investigator?
  - Is Nancy Drew portrayed as a private investigator in Ima Roberts' work?
- Teacher facts (2):
  - Nancy Drew is a fictional character and the main protagonist in a series of mystery novels.
  - A private investigator is a person who conducts investigations, usually for a private citizen or business.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Ima Roberts; Nancy Drew; Private investigator)

#### 10. `315575d5cbdd1d154744` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا کرٹ کوبائن کو پروجیکٹ سیمی کولن سے فائدہ ہوتا؟
- Gold question_en (DEV200, audit only): Would Kurt Cobain have benefited from Project Semicolon?
- Teacher question_en (v2 restatement, audit only): Would Kurt Cobain have benefited from Project Semi-Colon?
- Gold evidence titles (normalised): kurt cobain; project semicolon; suicide of kurt cobain
- R4 hits: kurt cobain
- Teacher entities: Kurt Cobain; Project Semi-Colon
- unknown_terms: Project Semi-Colon
- sub_questions (3):
  - What was Kurt Cobain's mental health condition?
  - What was the purpose of Project Semi-Colon?
  - How does Project Semi-Colon support individuals with mental health conditions?
- Teacher facts (2):
  - Kurt Cobain struggled with depression and anxiety throughout his life.
  - Kurt Cobain died by suicide in 1994.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Kurt Cobain; Project Sami Collen)

#### 11. `3570d57f16a40488ff42` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سلیکون کے سوٹ جودو کو مشکل بناتے ہیں؟
- Gold question_en (DEV200, audit only): Do silicone suits make judo difficult?
- Teacher question_en (v2 restatement, audit only): Do silicon suits make judo difficult?
- Gold evidence titles (normalised): judo; keikogi; leopold's maneuvers; silicone; silicone rubber
- R4 hits: judo
- Teacher entities: Silicon; Judo
- unknown_terms: []
- sub_questions (4):
  - What is silicon used for in clothing?
  - What are the properties of silicon that could affect physical activities?
  - How does judo require flexibility and movement?
  - Can silicon suits restrict movement?
- Teacher facts (4):
  - Silicon is sometimes used in clothing for its water-repellent and insulating properties.
  - Silicon is a flexible and durable material.
  - Judo is a combat sport that requires flexibility, strength, and movement.
  - Judo practitioners wear judogi, a type of uniform designed for flexibility and comfort.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Silicon; Judo)

#### 12. `38ef97eb7cdd4200fd00` — gold **yes**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا سنہری بالوں والی سبز آنکھوں والی سارہ پیکسٹن کو لاطینی سمجھا جاتا ہے؟
- Gold question_en (DEV200, audit only): Is blonde hair green eyed Sara Paxton considered a Latino?
- Teacher question_en (v2 restatement, audit only): Is Sarah Paxton with golden hair and green eyes considered Latina?
- Gold evidence titles (normalised): latin america; latino (demonym); mexico; sara paxton
- R4 hits: (none)
- Teacher entities: Sarah Paxton
- unknown_terms: []
- sub_questions (2):
  - What is Sarah Paxton's ethnicity?
  - What are the characteristics that define a Latina?
- Teacher facts (2):
  - Sarah Paxton is an American actress.
  - A Latina is a woman from a Latin American country or of Latin American descent.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Sarah Paxton; Latin Americans)

#### 13. `3de88c80f0914eb08caf` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا پتوں کی بناوٹ ان کے رنگ بدلنے سے آزاد رہتی ہے؟
- Gold question_en (DEV200, audit only): Does the texture of leaves remain the same independent of their coloring changing?
- Teacher question_en (v2 restatement, audit only): Does the structure of leaves remain independent of their color change?
- Gold evidence titles (normalised): anthocyanin; autumn leaf color; leaf
- R4 hits: leaf
- Teacher entities: Leaf; Leaf structure; Leaf color
- unknown_terms: []
- sub_questions (3):
  - What determines the structure of leaves?
  - What causes leaves to change color?
  - Is the structure of leaves related to their color?
- Teacher facts (3):
  - The structure of leaves is determined by the arrangement of cells and tissues within the leaf.
  - Leaves change color due to the breakdown of chlorophyll and the production of other pigments such as carotenoids and anthocyanins.
  - The structure of leaves is influenced by factors such as light, temperature, and water availability.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Leaf; Color)

#### 14. `3ebb62af96f836d2d4d6` — gold **no**, A_r2 no, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا فلمیں ہمیشہ چالاکوں کو ناکام دکھاتی ہیں؟
- Gold question_en (DEV200, audit only): Do movies always show nerds as the losers?
- Teacher question_en (v2 restatement, audit only): Do movies always depict con artists as unsuccessful?
- Gold evidence titles (normalised): facebook; superbad (film); the social network
- R4 hits: (none)
- Teacher entities: Con artist; Film
- unknown_terms: []
- sub_questions (4):
  - Do movies typically portray con artists as failing in their schemes?
  - Are there any notable movies that depict con artists as successful?
  - Do movies often show con artists facing consequences for their actions?
  - Do some movies portray con artists as sympathetic characters?
- Teacher facts (4) — parse_ok True, repaired brace:
  - Con artists are often portrayed as charismatic characters in movies.
  - Some movies, such as 'The Sting' and 'Catch Me If You Can', depict con artists as successful in their schemes.
  - Many movies show con artists facing consequences for their actions, such as being caught by the authorities or facing revenge from their victims.
  - The portrayal of con artists in movies can vary widely, with some films depicting them as sympathetic characters and others as villainous.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Film; Con artist)

#### 15. `42c947475244ed02629e` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سیپی کو گٹار بنانے میں استعمال کیا جا سکتا ہے؟
- Gold question_en (DEV200, audit only): Can oysters be used in guitar manufacturing?
- Teacher question_en (v2 restatement, audit only): Can sepiolite be used to make guitars?
- Gold evidence titles (normalised): guitar; inlay (guitar); nacre; oyster
- R4 hits: guitar
- Teacher entities: Sepiolite; Guitar
- unknown_terms: []
- sub_questions (3):
  - What is sepiolite?
  - What materials are commonly used to make guitars?
  - Are there any examples of sepiolite being used in musical instrument construction?
- Teacher facts (3):
  - Sepiolite is a type of clay mineral.
  - Guitars are typically made from wood, metal, and plastic materials.
  - There is no widely known use of sepiolite in the construction of guitars.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Capybara; Guitar)

#### 16. `48da75d87c66754ccc2e` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا پرل ہاربر پر حملے کے دوران امریکی ہاؤس اسپیکر کے طور پر خدمات انجام دینے والی واحد خاتون زندہ تھیں؟
- Gold question_en (DEV200, audit only): Was only woman to serve as U.S. Speaker of the House alive during the attack on Pearl Harbor?
- Teacher question_en (v2 restatement, audit only): Was the only woman serving as US House Speaker during the Pearl Harbor attack still alive?
- Gold evidence titles (normalised): attack on pearl harbor; nancy pelosi; speaker of the united states house of representatives
- R4 hits: speaker of the united states house of representatives
- Teacher entities: Pearl Harbor; United States House of Representatives; Speaker of the United States House of Representatives
- unknown_terms: []
- sub_questions (4):
  - Who was the US House Speaker during the Pearl Harbor attack?
  - Was the US House Speaker during the Pearl Harbor attack a woman?
  - When did the US House Speaker during the Pearl Harbor attack die?
  - Did the Pearl Harbor attack occur during the lifetime of the US House Speaker at that time?
- Teacher facts (4):
  - The attack on Pearl Harbor occurred on December 7, 1941.
  - Sam Rayburn was the Speaker of the United States House of Representatives from 1940 to 1947.
  - Sam Rayburn was a man.
  - Sam Rayburn died on November 16, 1961.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Pearl Harbor; Speaker of the United States House of Representatives)

#### 17. `660bfdc17b47f42facac` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا مووس کو ارینڈیل کے سلطنت کے قریب کام کے لیے استعمال کیا جاتا ہے؟
- Gold question_en (DEV200, audit only): Are moose used for work near the kingdom of Arendelle?
- Teacher question_en (v2 restatement, audit only): Is moose used for work near the kingdom of Arendelle?
- Gold evidence titles (normalised): arendelle: world of frozen; elsa (frozen); frozen (2013 film); moose
- R4 hits: moose
- Teacher entities: Moose; Arendelle
- unknown_terms: Arendelle
- sub_questions (4):
  - What is the geographical location of Arendelle?
  - What is the typical habitat of moose?
  - What are moose typically used for?
  - Are moose found in the geographical location of Arendelle?
- Teacher facts (3):
  - Moose are native to northern regions of North America, Europe, and Asia.
  - Moose are typically found in northern forests with abundant vegetation and access to water.
  - Moose are often used for their meat, hides, and antlers.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Mouse; Arendelle)

#### 18. `703db4629ff46d82a205` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا کین اس سال کی پرائمریز میں جو بائیڈن کو چیلنج کر سکتا ہے؟
- Gold question_en (DEV200, audit only): Can Kane challenge Joe Biden in this year's primaries?
- Teacher question_en (v2 restatement, audit only): Can Cain challenge Biden in this year's primaries?
- Gold evidence titles (normalised): joe biden; kane (wrestler); primary election
- R4 hits: joe biden
- Teacher entities: Joe Biden; Democratic Party presidential primaries
- unknown_terms: Cain
- sub_questions (3):
  - Who is Cain?
  - Is Cain eligible to run in the primaries?
  - What are the requirements to challenge Biden in the primaries?
- Teacher facts (3):
  - Joe Biden is the 46th President of the United States.
  - The Democratic Party presidential primaries are the process by which the Democratic Party selects its nominee for the presidency.
  - To run in the Democratic Party presidential primaries, a candidate must meet the eligibility requirements set by the Democratic National Committee.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Joe Biden; United States presidential primary)

#### 19. `7870b1cef39a4f685911` — gold **yes**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا آدم سینڈلر کو کول سپروس اور ایک سکوبا آدمک کے حوالے سے کوئی بات سمجھ آئے گی؟
- Gold question_en (DEV200, audit only): Would Adam Sandler get a reference to Cole Spouse and a scuba man doll?
- Teacher question_en (v2 restatement, audit only): Would Adam Sandler understand a reference to Coldplay and a scuba diver?
- Gold evidence titles (normalised): big daddy (1999 film)
- R4 hits: (none)
- Teacher entities: Adam Sandler; Coldplay; Scuba diving
- unknown_terms: []
- sub_questions (3):
  - Is Adam Sandler familiar with Coldplay?
  - Is Adam Sandler familiar with scuba diving?
  - Is a reference to Coldplay and a scuba diver a common or obscure reference?
- Teacher facts (3):
  - Adam Sandler is an American actor and comedian.
  - Coldplay is a British rock band.
  - Scuba diving is a mode of underwater diving that involves the use of scuba equipment.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Adam Sandler; Cold Sore; Scuba Diving)

#### 20. `8cfde6ee28d059a5aff6` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ویری لارج ٹیلیسکوپ تک پیدل جاتے ہوئے مرنا ممکن ہے؟
- Gold question_en (DEV200, audit only): Is it possible to get killed walking to the Very Large Telescope?
- Teacher question_en (v2 restatement, audit only): Is it possible to die walking to the Very Large Telescope?
- Gold evidence titles (normalised): arabian desert; atacama desert; desert; evan tanner; heat stroke; very large telescope
- R4 hits: very large telescope
- Teacher entities: Very Large Telescope
- unknown_terms: []
- sub_questions (4):
  - What is the location of the Very Large Telescope?
  - What are the environmental conditions at the location of the Very Large Telescope?
  - What are the physical demands of walking to the Very Large Telescope?
  - Can a person die from the physical demands of walking in the environmental conditions at the location of the Very Large Telescope?
- Teacher facts (4):
  - The Very Large Telescope is located on Cerro Paranal in the Atacama Desert of northern Chile.
  - Cerro Paranal is at an elevation of 2,635 metres (8,645 ft) above sea level.
  - The Atacama Desert is one of the driest places on Earth.
  - Walking at high elevations can cause altitude sickness in some individuals.
- v1 status of this qid: **concordant-right** (TF_v1 pred yes; v1 teacher entities: Atacama Large Millimeter/submillimeter Array; Walking; Death)

#### 21. `91917c5e30dbf8552b34` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا کوئی ویگن روایتی پایلا کھائے گا؟
- Gold question_en (DEV200, audit only): Would a vegan eat a traditional Paella dish?
- Teacher question_en (v2 restatement, audit only): Will a vegan eat traditional paella?
- Gold evidence titles (normalised): paella; vegan outreach; veganism
- R4 hits: paella; veganism
- Teacher entities: Paella; Veganism
- unknown_terms: []
- sub_questions (3):
  - What is the typical main ingredient of traditional paella?
  - What is the definition of a vegan diet?
  - Are there any variations of paella that are vegan-friendly?
- Teacher facts (3):
  - Traditional paella typically includes ingredients such as seafood, chorizo, and chicken.
  - A vegan diet is a type of diet that excludes all animal products, including eggs, dairy, and even honey.
  - Paella is a traditional Spanish dish that originated in the Valencia region.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Veganism; Traditional paya)

#### 22. `92c263f997f09dbb74a8` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا CHE101 میں کسی کو مایا اینجلو کی کتاب کی ضرورت ہوگی؟
- Gold question_en (DEV200, audit only): Would someone in CHE101 require a Maya Angelou book?
- Teacher question_en (v2 restatement, audit only): Will anyone in CHE101 need Maya Angelou's book?
- Gold evidence titles (normalised): general chemistry; maya angelou
- R4 hits: maya angelou
- Teacher entities: Maya Angelou
- unknown_terms: CHE101
- sub_questions (4):
  - What is CHE101?
  - What is the curriculum of CHE101?
  - Who is Maya Angelou?
  - What books did Maya Angelou write?
- Teacher facts (2):
  - Maya Angelou was an American poet, memoirist, and civil rights activist.
  - Maya Angelou wrote seven autobiographies and several books of poetry.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: CHE101; Maya Angelou)

#### 23. `9384dfc0aabc73026925` — gold **yes**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا جسمانی اور علامتی دل نمایاں طور پر مختلف نظر آتے ہیں؟
- Gold question_en (DEV200, audit only): Do anatomical and symbolic hearts look remarkably different?
- Teacher question_en (v2 restatement, audit only): Do physical and symbolic hearts appear distinctly different?
- Gold evidence titles (normalised): heart; heart symbol
- R4 hits: (none)
- Teacher entities: Human Heart; Symbolism of the Heart
- unknown_terms: []
- sub_questions (4):
  - What is the physical appearance of the human heart?
  - What are common symbolic representations of the heart?
  - Do symbolic hearts have a standardized appearance?
  - How does the physical heart differ from symbolic hearts?
- Teacher facts (4):
  - The human heart is a muscular organ that pumps blood throughout the body.
  - The human heart is typically depicted as a red, hollow, muscular organ with a rounded base and a tapering apex.
  - Symbolic hearts are often represented as a stylized, two-dimensional shape with a rounded base and a pointed apex.
  - Symbolic hearts can be found in various forms of art, literature, and culture, often representing love, passion, or emotion.
- v1 status of this qid: **concordant-right** (TF_v1 pred yes; v1 teacher entities: Physical heart; Symbolic heart)

#### 24. `9850eb1534206facd067` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا جدید مرکزی عمل کاری یونٹ کا سرکٹ چپ گھر کی چابی پر فٹ ہو سکتا ہے؟
- Gold question_en (DEV200, audit only): Would a modern central processing unit circuit chip fit on a housekey?
- Teacher question_en (v2 restatement, audit only): Can a modern central processing unit circuit chip fit on a house key?
- Gold evidence titles (normalised): central processing unit; integrated circuit; lock and key
- R4 hits: central processing unit; integrated circuit
- Teacher entities: Central processing unit; Circuit; Integrated circuit; House key
- unknown_terms: []
- sub_questions (4):
  - What is the typical size of a modern central processing unit circuit chip?
  - What is the typical size of a house key?
  - What is the smallest possible size of a central processing unit circuit chip?
  - Are there any central processing unit circuit chips that are specifically designed to be small?
- Teacher facts (6) — parse_ok True, repaired brace:
  - A central processing unit (CPU) is the primary component of a computer that performs most of the processing inside a computer.
  - The first microprocessor, the Intel 4004, was released by Intel in 1971 and was the size of a fingernail.
  - A typical house key is around 2-3 inches (5-7.5 cm) in length.
  - Some CPUs are designed for specific applications, such as smartphones or embedded systems, and are smaller than those used in desktop computers.
  - The smallest CPUs are usually found in embedded systems, such as traffic lights or microwave ovens.
  - Some CPUs are designed to be extremely small, such as the ARM Cortex-M0+, which is used in some wearable devices and is smaller than a grain of rice.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Central processing unit; Integrated circuit; Microchip)

#### 25. `987260ffef60cb2c5439` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا پرل ہاربر شارک دیوی کا خیالی گھر ہے؟
- Gold question_en (DEV200, audit only): Is Pearl Harbor the mythical home of a shark goddess?
- Teacher question_en (v2 restatement, audit only): Is Pearl Harbor the fictional home of Shark Girl?
- Gold evidence titles (normalised): pearl harbor
- R4 hits: pearl harbor
- Teacher entities: Pearl Harbor
- unknown_terms: Shark Girl
- sub_questions (2):
  - What is the fictional home of Shark Girl?
  - Is Pearl Harbor associated with Shark Girl?
- Teacher facts (1):
  - Pearl Harbor is a lagoon harbor on the island of Oahu, Hawaii, United States.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Pearl Harbor; Shark Girl)

#### 26. `9bb59488a227a5f5cd62` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا آپ جی بی پر گیم آف تھرونز کے ہر ایچ ڈی قسط کو محفوظ کر سکتے ہیں؟
- Gold question_en (DEV200, audit only): Can you save every HD episode of Game of Thrones on Samsung Galaxy A10e?
- Teacher question_en (v2 restatement, audit only): Can you save every HD episode of Game of Thrones on a GB?
- Gold evidence titles (normalised): game of thrones; game of thrones (season 1); high-definition video; samsung galaxy; samsung galaxy a10; the iron throne (game of thrones)
- R4 hits: game of thrones
- Teacher entities: Game of Thrones; Gigabyte
- unknown_terms: []
- sub_questions (4):
  - What is the storage capacity of a GB?
  - What is the file size of an HD episode of Game of Thrones?
  - How many episodes of Game of Thrones are there in total?
  - What is the total file size of all HD episodes of Game of Thrones?
- Teacher facts (4) — parse_ok True, repaired brace:
  - Game of Thrones is an American fantasy drama television series.
  - A gigabyte (GB) is a unit of digital information or computer storage.
  - The average file size of an HD episode is around 4-6 GB.
  - There are 73 episodes of Game of Thrones in total.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Game of Thrones; GB; HD video)

#### 27. `a0896de3fd13cd0f3e16` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا اٹلانٹک کوڈ ویجی مائٹ سینڈوچ میں پایا جاتا ہے؟
- Gold question_en (DEV200, audit only): Is Atlantic cod found in a vegemite sandwich?
- Teacher question_en (v2 restatement, audit only): Is Atlantic cod found in a veggie might sandwich?
- Gold evidence titles (normalised): animal; atlantic cod; fish; vegemite; yeast extract
- R4 hits: atlantic cod
- Teacher entities: Atlantic Cod
- unknown_terms: veggie might sandwich
- sub_questions (3):
  - What is Atlantic cod?
  - What is a veggie might sandwich?
  - What are the typical ingredients of a veggie might sandwich?
- Teacher facts (2):
  - Atlantic cod is a well-known fish species native to the North Atlantic.
  - Atlantic cod is a mild-flavored fish often used in food products.
- v1 status of this qid: **loss** (TF_v1 pred yes; v1 teacher entities: Atlantic cod; Cod as food; Sandwich)

#### 28. `c921bdade889def26cf4` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا آئسوپروپائل الکحل میں نمک کی حل پذیری کم ہے؟
- Gold question_en (DEV200, audit only): In isopropyl alcohol, is the solubility of salt low?
- Teacher question_en (v2 restatement, audit only): Is the solubility of salt lower in isopropyl alcohol?
- Gold evidence titles (normalised): isopropyl alcohol; salt; solubility
- R4 hits: isopropyl alcohol; salt; solubility
- Teacher entities: Isopropyl alcohol; Salt; Solubility
- unknown_terms: []
- sub_questions (4):
  - What is the chemical composition of isopropyl alcohol?
  - What is the chemical composition of salt?
  - How does the chemical composition of a solvent affect the solubility of a solute?
  - What is the solubility of salt in isopropyl alcohol?
- Teacher facts (4):
  - Isopropyl alcohol is a compound with the chemical formula C3H8O.
  - Salt is a compound with the chemical formula NaCl.
  - The solubility of a solute in a solvent is affected by the chemical composition of the solvent, including its polarity and intermolecular forces.
  - Isopropyl alcohol is a polar solvent.
- v1 status of this qid: **concordant-right** (TF_v1 pred yes; v1 teacher entities: Isopropyl alcohol; Solubility; Sodium chloride)

#### 29. `d9aeae10998c093c0cc9` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ٹویوٹا سپرا وی لاگ بنا سکتی ہے؟
- Gold question_en (DEV200, audit only): Can a Toyota Supra make a vlog?
- Teacher question_en (v2 restatement, audit only): Can the Toyota Supra be made into a wagon?
- Gold evidence titles (normalised): toyota supra; vlog
- R4 hits: toyota supra
- Teacher entities: Toyota Supra; Wagon
- unknown_terms: []
- sub_questions (4):
  - What is the design of the Toyota Supra?
  - Can the Toyota Supra be modified?
  - What is the typical body style of a wagon?
  - Are there any examples of sports cars being converted into wagons?
- Teacher facts (4) — parse_ok True, repaired brace:
  - The Toyota Supra is a sports car with a front-engine, rear-wheel-drive layout.
  - The Toyota Supra has a two-door coupé body style.
  - A wagon is a car body style with a roof that extends to the rear of the vehicle, often with a flat floor and a rear door that swings upwards.
  - Some sports cars have been converted into wagons by aftermarket companies or enthusiasts.
- v1 status of this qid: **concordant-right** (TF_v1 pred no; v1 teacher entities: Toyota Supra; Turbocharger)

#### 30. `dabf94934721c26b2422` — gold **no**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا سیاہ دم والے خرگوش یورپی جنگلی بلی سے ڈرتے ہیں؟
- Gold question_en (DEV200, audit only): Do black-tailed jackrabbits fear the European wildcat?
- Teacher question_en (v2 restatement, audit only): Do black-tailed jackrabbits fear European wildcats?
- Gold evidence titles (normalised): black-tailed jackrabbit; european wildcat
- R4 hits: black-tailed jackrabbit; european wildcat
- Teacher entities: Black-tailed jackrabbit; European wildcat; Habitat; Predation; Fear response
- unknown_terms: []
- sub_questions (4):
  - Do black-tailed jackrabbits and European wildcats share a common habitat?
  - Do European wildcats prey on black-tailed jackrabbits?
  - Do black-tailed jackrabbits have natural fear responses to predators?
  - What are the typical behaviors of European wildcats when encountering prey?
- Teacher facts (6) — parse_ok True, repaired brace:
  - Black-tailed jackrabbits are native to North America.
  - European wildcats are native to Europe and Asia.
  - European wildcats are carnivorous mammals that primarily prey on small mammals.
  - Black-tailed jackrabbits have a strong instinct to flee when sensing predators.
  - European wildcats are solitary and primarily nocturnal animals.
  - Black-tailed jackrabbits inhabit deserts and grasslands in North America.
- v1 status of this qid: **loss** (TF_v1 pred yes; v1 teacher entities: European Rabbit; European Wildcat)

#### 31. `e6d3973ed3feb8a42928` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا افریقہائزڈ مکھیاں کثیر الثقافتی سمجھی جا سکتی ہیں؟
- Gold question_en (DEV200, audit only): Can Africanized bees be considered multicultural?
- Teacher question_en (v2 restatement, audit only): Can Africanized bees be considered multicultural?
- Gold evidence titles (normalised): african bee; africanized bee; honey bee; multiculturalism; multiracial people; western honey bee
- R4 hits: africanized bee; multiculturalism
- Teacher entities: Multiculturalism; Africanized bee
- unknown_terms: []
- sub_questions (3):
  - What is the definition of multiculturalism?
  - What are the characteristics of Africanized bees?
  - Do Africanized bees exhibit cultural behaviors?
- Teacher facts (3):
  - Multiculturalism is the coexistence of diverse cultures in a single society.
  - Africanized bees are a hybrid species of bee created by cross-breeding the European honey bee with the African honey bee.
  - Africanized bees are known for their aggressive behavior and ability to adapt to different environments.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Africanized bee; Multiculturalism)

#### 32. `f8e6087cd52c925fedad` — gold **yes**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا قدیم ٹونی بینیٹ 2020 میں بچہ پیدا کر سکتے تھے؟
- Gold question_en (DEV200, audit only): Could ancient Tony Bennett have a baby in 2020?
- Teacher question_en (v2 restatement, audit only): Could ancient Tony Bennett have fathered a child in 2020?
- Gold evidence titles (normalised): ramjit raghav; tony bennett
- R4 hits: tony bennett
- Teacher entities: Tony Bennett
- unknown_terms: []
- sub_questions (3):
  - When was Tony Bennett born?
  - What is the typical age range for men to father children?
  - Is Tony Bennett still alive in 2020?
- Teacher facts (3):
  - Tony Bennett was born on August 3, 1926.
  - Men can father children at any age, but fertility declines with age.
  - Tony Bennett was still alive in 2020.
- v1 status of this qid: **loss** (TF_v1 pred no; v1 teacher entities: Tony Bennett; 2020)

### B. The 41 TF wins (TF right, A_r2 wrong)

By (gold, A_r2 pred): gold no → A_r2 yes: 22; gold yes → A_r2 no: 14; gold no → A_r2 —: 4; gold yes → A_r2 —: 1. R4 hit yes: 33, no: 8. v1 status of these qids: win 28, concordant-wrong 13.

#### 1. `05e9cdc44f1b066badd7` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا خواتین میں مونچھیں ہونا عام بات ہے؟
- Gold question_en (DEV200, audit only): Is it common for women to have moustaches?
- Teacher question_en (v2 restatement, audit only): Is it common for women to have mustaches?
- Gold evidence titles (normalised): beard; facial hair; moustache
- R4 hits: facial hair
- Teacher entities: Facial hair; Hirsutism; Mustache
- unknown_terms: []
- sub_questions (4):
  - What is the typical facial hair pattern in women?
  - What is the prevalence of hirsutism in women?
  - What is the definition of a mustache?
  - How common is it for women to have visible facial hair?
- Teacher facts (4):
  - Facial hair is a normal part of human development, but its pattern and density vary between individuals and sexes.
  - Hirsutism is a condition characterized by excessive growth of terminal hair in a male pattern in women, affecting approximately 5-10% of women of reproductive age.
  - A mustache is a growth of hair on the upper lip, typically found in men but also present in some women.
  - Visible facial hair is present in approximately 40% of women, particularly after menopause.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Women; Moustache)

#### 2. `064d8faaa8f164b55270` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا اپالو 15 کے عملے کو ایک پہیے والی سائیکل چلانے میں دشواری ہوگی؟
- Gold question_en (DEV200, audit only): Would the crew of Apollo 15 have difficulty riding a unicycle?
- Teacher question_en (v2 restatement, audit only): Would the Apollo 15 crew have difficulty riding a unicycle?
- Gold evidence titles (normalised): apollo 15; unicycle
- R4 hits: apollo 15; unicycle
- Teacher entities: Apollo 15; Unicycle
- unknown_terms: []
- sub_questions (3):
  - What was the physical condition of the Apollo 15 crew?
  - What skills are required to ride a unicycle?
  - Did the Apollo 15 crew have experience with unicycles?
- Teacher facts (3):
  - The Apollo 15 crew consisted of David Scott, James Irwin, and Alfred Worden.
  - Riding a unicycle requires balance, coordination, and leg strength.
  - Astronauts undergo rigorous physical training to prepare for space missions.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred no; v1 teacher entities: Apollo 15; Unicycle)

#### 3. `0fe3edf35bd1a1e662a9` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا گینڈالف فرضی طور پر چارمد کے بارباس کے لیے ایک زبردست دشمن ہے؟
- Gold question_en (DEV200, audit only): Is Gandalf hypothetically a formidable foe for Charmed's Barbas?
- Teacher question_en (v2 restatement, audit only): Is Gandalf a great fictional enemy of Charmed's Barbas?
- Gold evidence titles (normalised): barbas (charmed); gandalf; marbas
- R4 hits: barbas (charmed); gandalf
- Teacher entities: Gandalf; Charmed; Barbas (Charmed)
- unknown_terms: []
- sub_questions (4):
  - Is Gandalf a fictional character?
  - Is Barbas a fictional character from Charmed?
  - What are the characteristics of Gandalf?
  - What are the characteristics of Barbas?
- Teacher facts (4):
  - Gandalf is a fictional character in J. R. R. Tolkien's Middle-earth legendarium.
  - Barbas is a fictional character in the American television series Charmed.
  - Gandalf is a powerful wizard in J. R. R. Tolkien's Middle-earth legendarium.
  - Barbas is a demon in the American television series Charmed.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Gandalf; Barbossa; Pirates of the Caribbean)

#### 4. `16d7c42444ef13ef4643` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا کیٹ گوسلین کے گھرانے میں بینڈی ٹیم بنانے کے لیے کافی افراد ہیں؟
- Gold question_en (DEV200, audit only): Can Kate Gosselin's household fill out a Bandy team?
- Teacher question_en (v2 restatement, audit only): Are there enough people in the Gosling family to form a bandy team?
- Gold evidence titles (normalised): bandy; kate gosselin; kate plus 8
- R4 hits: bandy
- Teacher entities: Bandy; Ryan Gosling
- unknown_terms: []
- sub_questions (3):
  - How many people are typically in a bandy team?
  - Who are the members of the Gosling family?
  - How many people are required to form a bandy team?
- Teacher facts (3):
  - Bandy is a team winter sport played on ice, in which skaters use sticks to direct a ball into the opposing team's goal.
  - A standard bandy team has 11 players on the ice for each team.
  - Ryan Gosling is a Canadian actor and musician.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred yes; v1 teacher entities: Cat Goslin; Bandie)

#### 5. `3486b6b5c69878bb088d` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ایریزونا میں کیلیفورنیا کے مقابلے میں زیادہ گوبھی اگتی ہے؟
- Gold question_en (DEV200, audit only): Do more Cauliflower grow in Arizona than California?
- Teacher question_en (v2 restatement, audit only): Does Arizona grow more cabbage than California?
- Gold evidence titles (normalised): arizona; california; cauliflower; climate of the united states
- R4 hits: arizona; california
- Teacher entities: Arizona; California; Cabbage
- unknown_terms: []
- sub_questions (3):
  - What is the total cabbage production in Arizona?
  - What is the total cabbage production in California?
  - How do cabbage production levels compare between different states in the United States?
- Teacher facts (3):
  - Arizona is a state in the southwestern United States.
  - California is a state on the West Coast of the United States.
  - Cabbage is a dense-green, leafy vegetable.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Arizona; California; Cabbage)

#### 6. `3503e8d41364eb0b33bc` — gold **no**, A_r2 —, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ایک ذمہ دار بارٹینڈر میلی بوبی براؤن کے لیے مشروب بنائے گا؟
- Gold question_en (DEV200, audit only): Would a responsible bartender make a drink for Millie Bobby Brown?
- Teacher question_en (v2 restatement, audit only): Would a responsible bartender serve a drink to Miley Cyrus?
- Gold evidence titles (normalised): legal drinking age; millie bobby brown; national minimum drinking age act
- R4 hits: (none)
- Teacher entities: Miley Cyrus; Bartender; Alcohol
- unknown_terms: []
- sub_questions (3):
  - Is Miley Cyrus a minor?
  - Is Miley Cyrus intoxicated?
  - Is Miley Cyrus a person who can legally be served alcohol in a bar?
- Teacher facts (3):
  - Miley Cyrus was born on November 23, 1992.
  - In the United States, the minimum drinking age is 21 years old.
  - A bartender is typically responsible for serving drinks to patrons in a bar.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Milly Bobby Brown; Bartender)

#### 7. `3a61dcbed7e358b3aee8` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا 200 مرد قطار میں گیزا کے اہرام مصر اعظم کی بنیاد کو ڈھانپ سکتے ہیں؟
- Gold question_en (DEV200, audit only): Can 200 men end to end cover Great Pyramid of Giza's base?
- Teacher question_en (v2 restatement, audit only): Can 200 men in a line cover the base of the Great Pyramid of Giza?
- Gold evidence titles (normalised): dinka people; foot (unit); great pyramid of giza; human height
- R4 hits: great pyramid of giza
- Teacher entities: Great Pyramid of Giza; Man
- unknown_terms: []
- sub_questions (4):
  - What is the base area of the Great Pyramid of Giza?
  - What is the average area occupied by a man in a line?
  - How many men are needed to cover the base area of the Great Pyramid of Giza?
  - What is the relationship between the number of men and the base area of the Great Pyramid of Giza?
- Teacher facts (4) — parse_ok True, repaired brace:
  - The base of the Great Pyramid of Giza is a square, each side being 230.4 meters long.
  - The base area of the Great Pyramid of Giza is 52,900 square meters.
  - The average area occupied by a man in a line is approximately 0.1 square meters.
  - The Great Pyramid of Giza is located in modern-day Egypt.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred no; v1 teacher entities: Great Pyramid of Giza; Egypt)

#### 8. `3b38e00d8d1da26f7f02` — gold **no**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ماریو تیز دوڑنے کے لیے مشروم استعمال کرتا ہے؟
- Gold question_en (DEV200, audit only): Does Mario use mushrooms to run faster?
- Teacher question_en (v2 restatement, audit only): Does Mario use mushrooms to run fast?
- Gold evidence titles (normalised): goomba; super mario
- R4 hits: (none)
- Teacher entities: Mario; Mushroom (Mario)
- unknown_terms: []
- sub_questions (3):
  - What is the effect of mushrooms on Mario in the Mario video game series?
  - Is running fast a power-up effect in the Mario video game series?
  - Are mushrooms a power-up in the Mario video game series?
- Teacher facts (3):
  - In the Mario video game series, mushrooms are a power-up that makes Mario grow bigger and allows him to take an extra hit.
  - The power-up effect of mushrooms in the Mario video game series includes increased size and health, but not necessarily speed.
  - Mushrooms are a recurring power-up in the Mario video game series.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Mario; Super Mushroom)

#### 9. `3d7050c0e9bfbeef27c1` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا میلانیا ٹرمپ کا پیشہ اولگا کیوریلنکو کے پیشے کے برابر تھا؟
- Gold question_en (DEV200, audit only): Did Melania Trump have same profession as Olga Kurylenko?
- Teacher question_en (v2 restatement, audit only): Was Melania Trump's profession the same as Olga Kurylenko's profession?
- Gold evidence titles (normalised): melania trump; olga kurylenko; olga storozhenko
- R4 hits: melania trump; olga kurylenko
- Teacher entities: Melania Trump; Olga Kurylenko
- unknown_terms: []
- sub_questions (2):
  - What was Melania Trump's profession?
  - What was Olga Kurylenko's profession?
- Teacher facts (2):
  - Melania Trump is a former model.
  - Olga Kurylenko is a model and actress.
- v1 status of this qid: **win** (TF_v1 pred yes; v1 teacher entities: Melania Trump; Olga Kurylenko)

#### 10. `3f5af97fdff20028347b` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا فرانسیسی انقلاب ٹیلیویژن پر دکھایا گیا تھا؟
- Gold question_en (DEV200, audit only): Was the French Revolution televised?
- Teacher question_en (v2 restatement, audit only): Was the French Revolution shown on television?
- Gold evidence titles (normalised): french revolution; television
- R4 hits: french revolution; television
- Teacher entities: French Revolution; Television
- unknown_terms: []
- sub_questions (3):
  - When did the French Revolution occur?
  - When was television first used for broadcasting?
  - Was the French Revolution recorded or broadcast on television at the time?
- Teacher facts (3):
  - The French Revolution began in 1789 and lasted for more than a decade.
  - The first public demonstration of a television system was held by John Logie Baird in 1926.
  - Regular high-definition television broadcasts did not begin until the late 20th century.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred yes; v1 teacher entities: French Revolution; Television)

#### 11. `43da475d5c2db929c458` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ایفل ٹاور آرکٹک سمندر کے سب سے گہرے مقام پر مکمل طور پر ڈوب سکتا ہے؟
- Gold question_en (DEV200, audit only): Could the Eiffel Tower be completely submerged at the Arctic Ocean's deepest point?
- Teacher question_en (v2 restatement, audit only): Can the Eiffel Tower be fully submerged at the Arctic Ocean's deepest point?
- Gold evidence titles (normalised): eiffel tower; fram strait; molloy deep
- R4 hits: eiffel tower
- Teacher entities: Eiffel Tower; Arctic Ocean
- unknown_terms: []
- sub_questions (3):
  - What is the height of the Eiffel Tower?
  - What is the depth of the Arctic Ocean's deepest point?
  - Is the Eiffel Tower's height less than the Arctic Ocean's deepest point?
- Teacher facts (2):
  - The Eiffel Tower is 330 meters tall.
  - The Arctic Ocean's deepest point is in the Molloy Deep, with a depth of approximately 5,669 meters.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred no; v1 teacher entities: Eiffel Tower; Arctic Ocean)

#### 12. `4760fe3d0c80beb30edd` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): اگر کوئی شخص ویگن ہو تو کیا وہ شہد کھائے گا؟
- Gold question_en (DEV200, audit only): If someone is a vegan, would they eat honey?
- Teacher question_en (v2 restatement, audit only): If a person is vegan, will they eat honey?
- Gold evidence titles (normalised): honey; veganism
- R4 hits: honey; veganism
- Teacher entities: Veganism; Honey
- unknown_terms: []
- sub_questions (3):
  - What is the definition of veganism?
  - Is honey an animal product?
  - Do vegans consume animal products?
- Teacher facts (3):
  - Veganism is a lifestyle and dietary choice that seeks to exclude, as far as possible, all forms of exploitation of animals for food, clothing, and other purposes.
  - Honey is a sweet, viscous fluid produced by bees from the nectar of plants.
  - Vegans do not consume or use products that come from animals, including eggs, dairy products, and even honey.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Veganism; Honey)

#### 13. `4e1b65e81ec09397b26e` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا بغیر کسی سازوسامان کے دیوقامت اسکویڈ کو اس کے قدرتی مسکن میں پکڑنا ناممکن ہے؟
- Gold question_en (DEV200, audit only): Is capturing giant squid in natural habitat impossible with no gear?
- Teacher question_en (v2 restatement, audit only): Is it impossible to catch a giant squid in its natural habitat without any equipment?
- Gold evidence titles (normalised): freediving; giant squid; scuba diving; underwater diving
- R4 hits: giant squid
- Teacher entities: Giant Squid; Deep-Sea Environment; Diving Equipment; Human Diver
- unknown_terms: []
- sub_questions (4):
  - What is the typical environment of a giant squid?
  - What are the common methods for catching giant squid?
  - What are the physical characteristics of a giant squid?
  - What are the limitations of human divers in deep water?
- Teacher facts (4):
  - Giant squid are found in the deep waters of the Atlantic, Pacific, and Indian Oceans.
  - Giant squid are typically caught using deep-sea submersibles or remote-operated vehicles (ROVs).
  - Giant squid can grow up to 13 meters in length and weigh up to 750 kilograms.
  - Human divers can experience nitrogen narcosis and oxygen toxicity at depths below 40 meters.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred no; v1 teacher entities: Colossal Squid; Habitat)

#### 14. `4ec8cd118be8a53ec516` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا الیگزینڈر ہیملٹن کو کوالاز کے بارے میں معلوم ہوتا؟
- Gold question_en (DEV200, audit only): Would Alexander Hamilton have known about koalas?
- Teacher question_en (v2 restatement, audit only): Did Alexander Hamilton know about cowpox?
- Gold evidence titles (normalised): alexander hamilton; koala
- R4 hits: alexander hamilton
- Teacher entities: Alexander Hamilton; Cowpox
- unknown_terms: []
- sub_questions (4):
  - What was Alexander Hamilton's profession?
  - When was cowpox first described in medical literature?
  - Was cowpox widely known in the 18th century?
  - Did Alexander Hamilton have any connection to the medical field?
- Teacher facts (4):
  - Alexander Hamilton was a statesman and one of the Founding Fathers of the United States.
  - Cowpox was first described in medical literature by English physician Edward Jenner in 1798.
  - Cowpox was not widely known in the 18th century until Edward Jenner's publication.
  - Edward Jenner's work on cowpox led to the development of the first vaccine against smallpox.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Alexander Hamilton; Cowpox)

#### 15. `502d4dae6f08e73a5569` — gold **yes**, A_r2 no, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ہوم ڈیپو سلیب پر چڑھانے کے سامان کے لیے ایک جامع دکان ہے؟
- Gold question_en (DEV200, audit only): Is Home Depot a one stop shop for crucifixion supplies?
- Teacher question_en (v2 restatement, audit only): Is Home Depot a comprehensive store for climbing gear?
- Gold evidence titles (normalised): crucifixion; descriptions in antiquity of the execution cross; one stop shop; the home depot
- R4 hits: (none)
- Teacher entities: Home Depot; Climbing
- unknown_terms: []
- sub_questions (3):
  - What products does Home Depot typically sell?
  - What types of climbing gear are commonly used?
  - Does Home Depot carry climbing gear?
- Teacher facts (3):
  - Home Depot is an American home improvement retailer.
  - Home Depot sells a wide range of products, including building materials, tools, and gardening supplies.
  - Climbing gear includes equipment such as ropes, harnesses, and carabiners.
- v1 status of this qid: **win** (TF_v1 pred yes; v1 teacher entities: Home Depot; Sleek)

#### 16. `5e2d9519c7c521411817` — gold **yes**, A_r2 —, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ٹی ایم این ٹی رنگ بھرنے کی کتاب میں پیزا ہوگا؟
- Gold question_en (DEV200, audit only): Would a TMNT coloring book have pizza in it?
- Teacher question_en (v2 restatement, audit only): Will TMNT coloring book have pizza?
- Gold evidence titles (normalised): teenage mutant ninja turtles
- R4 hits: teenage mutant ninja turtles
- Teacher entities: Teenage Mutant Ninja Turtles; Coloring book; Pizza
- unknown_terms: []
- sub_questions (3):
  - What is the main theme of TMNT coloring books?
  - Is pizza a recurring element in the TMNT franchise?
  - Do TMNT coloring books typically include recurring elements from the franchise?
- Teacher facts (3):
  - Teenage Mutant Ninja Turtles is a franchise created by Kevin Eastman and Peter Laird.
  - Pizza is a favorite food of the Teenage Mutant Ninja Turtles characters.
  - Teenage Mutant Ninja Turtles coloring books often feature characters and scenes from the franchise.
- v1 status of this qid: **win** (TF_v1 pred yes; v1 teacher entities: TMNT; Coloring book; Pizza)

#### 17. `690a36d5fed5b17969aa` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا جیک ڈیمپسی نے موجودہ WBC ہیوی ویٹ چیمپئن سے مقابلہ کیا تھا؟
- Gold question_en (DEV200, audit only): Did Jack Dempsey fight the current WBC heavyweight champion?
- Teacher question_en (v2 restatement, audit only): Did Jack Dempsey fight the current WBC heavyweight champion?
- Gold evidence titles (normalised): deontay wilder; jack dempsey; tyson fury
- R4 hits: jack dempsey
- Teacher entities: Jack Dempsey; World Boxing Council
- unknown_terms: []
- sub_questions (3):
  - Who is the current WBC heavyweight champion?
  - Did Jack Dempsey ever fight the current WBC heavyweight champion?
  - Is the current WBC heavyweight champion a boxer from the same era as Jack Dempsey?
- Teacher facts (2):
  - Jack Dempsey was an American professional boxer who competed from 1914 to 1927.
  - The World Boxing Council (WBC) is one of the four major organizations which sanction professional boxing bouts.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Jack Dempsey; World Boxing Council; Heavyweight)

#### 18. `70b2122323424f6e80c2` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا لیبرون جیمز فرضی طور پر یوری گاگرین کی طرف اوپر دیکھے گا؟
- Gold question_en (DEV200, audit only): Would LeBron James hypothetically glance upwards at Yuri Gagarin?
- Teacher question_en (v2 restatement, audit only): Would LeBron James look up to Yuri Gagarin hypothetically?
- Gold evidence titles (normalised): lebron james; yuri gagarin
- R4 hits: lebron james; yuri gagarin
- Teacher entities: LeBron James; Yuri Gagarin
- unknown_terms: []
- sub_questions (4):
  - What is LeBron James' profession?
  - What is Yuri Gagarin known for?
  - Do people in LeBron James' profession typically look up to pioneers in Yuri Gagarin's field?
  - Is Yuri Gagarin considered a pioneer in his field?
- Teacher facts (4):
  - LeBron James is a professional basketball player.
  - Yuri Gagarin was a Soviet pilot and cosmonaut.
  - Yuri Gagarin is widely regarded as a pioneer in space exploration.
  - Yuri Gagarin became the first human to journey into outer space in 1961.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: LeBron James; Yuri Gagarin)

#### 19. `71ca1a8a75c5c582ab3f` — gold **no**, A_r2 —, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا زیادہ تر ہائی اسکول کے ہیڈ کوچز این سی ایس یو کے ہیڈ کوچ جتنا کماتے ہیں؟
- Gold question_en (DEV200, audit only): Do most high school head coaches make as much as the Head Coach at NCSU?
- Teacher question_en (v2 restatement, audit only): Do most high school head coaches earn as much as the head coach of the NCSU?
- Gold evidence titles (normalised): head coach; nc state wolfpack; nc state wolfpack football
- R4 hits: (none)
- Teacher entities: High school; North Carolina State University
- unknown_terms: []
- sub_questions (4):
  - What is the average salary of high school head coaches?
  - What is the salary of the head coach of the NCSU?
  - How do the salaries of high school head coaches compare to that of the head coach of the NCSU?
  - What is the typical salary range for high school head coaches?
- Teacher facts (6) — parse_ok True, repaired brace:
  - The average salary of a high school head coach in the United States varies by state and school district.
  - The salary of the head coach of the North Carolina State University (NCSU) football team is around $3 million per year.
  - High school head coaches are typically paid by their school districts.
  - The salary of the head coach of the NCSU is determined by the university's athletic department.
  - The typical salary range for high school head coaches is between $30,000 and $70,000 per year.
  - High school head coaches often have additional income from summer camps, clinics, and other sports-related activities.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: High school; Head coach; NCAA)

#### 20. `73c52134ab2a903d86db` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا لیموں دودھ کے ذائقے کو بہتر بناتا ہے؟
- Gold question_en (DEV200, audit only): Does Lemon enhance the flavor of milk?
- Teacher question_en (v2 restatement, audit only): Does lemon improve the taste of milk?
- Gold evidence titles (normalised): curdling; lemon; soured milk
- R4 hits: lemon
- Teacher entities: Lemon; Milk; pH
- unknown_terms: []
- sub_questions (3):
  - What is the effect of lemon juice on milk?
  - How does acidity affect the taste of milk?
  - What is the pH level of lemon juice?
- Teacher facts (3):
  - Lemon juice is acidic in nature.
  - The acidity of lemon juice can cause milk to curdle.
  - The pH level of lemon juice is typically around 2.0.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Lemon; Milk)

#### 21. `7419826e4373eb1e83ff` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا کتا گرے سیل سے پہلے گھنٹی کی آواز پر ردعمل ظاہر کرے گا؟
- Gold question_en (DEV200, audit only): Would a dog respond to bell before Grey seal?
- Teacher question_en (v2 restatement, audit only): Will a dog react to the sound of a bell before a grey seal?
- Gold evidence titles (normalised): dog; dog anatomy; grey seal; hearing range; hertz; pinniped
- R4 hits: dog; grey seal
- Teacher entities: Dog; Grey seal; Bell
- unknown_terms: []
- sub_questions (3):
  - What is the typical reaction of a dog to a bell sound?
  - What is the typical reaction of a grey seal to a bell sound?
  - How do dogs and grey seals typically perceive auditory stimuli?
- Teacher facts (3):
  - Dogs are known to be able to hear sounds within the range of 50 Hz to 45,000 Hz.
  - Grey seals are able to hear sounds within the range of 50 Hz to 32,000 Hz.
  - Dogs are often trained using positive reinforcement techniques that involve the use of a bell sound as a stimulus.
- v1 status of this qid: **win** (TF_v1 pred yes; v1 teacher entities: Dog; Grey seal; Bell)

#### 22. `7b84d2bc643ddc2085f0` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا نوح کی کشتی مفروضاتی طور پر سیلاب زدہ لنکن سرنگ سے گزر سکتی ہے؟
- Gold question_en (DEV200, audit only): WIll Noah's Ark hypothetically sail through flooded Lincoln Tunnel?
- Teacher question_en (v2 restatement, audit only): Could Noah's Ark hypothetically pass through the flooded Lincoln Tunnel?
- Gold evidence titles (normalised): cubit; lincoln tunnel; noah's ark
- R4 hits: lincoln tunnel; noah's ark
- Teacher entities: Noah's Ark; Lincoln Tunnel
- unknown_terms: []
- sub_questions (3):
  - What were the dimensions of Noah's Ark?
  - What are the dimensions of the Lincoln Tunnel?
  - How high was the flood in the Lincoln Tunnel?
- Teacher facts (3):
  - According to the biblical account, Noah's Ark was 300 cubits long, 50 cubits wide, and 30 cubits high.
  - The Lincoln Tunnel has a maximum height of 13 feet 1 inch (4 meters) and a maximum width of 31 feet 6 inches (9.6 meters).
  - The Lincoln Tunnel is approximately 1.5 miles (2.4 kilometers) long.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Noah's Ark; Lincoln Tunnel)

#### 23. `80ca095e38ab73b908ff` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا شمالی فر سیل چھ سالہ بچے کے لیے اچھا پالتو جانور ہے؟
- Gold question_en (DEV200, audit only): Does Northern fur seal make good pet for six year old?
- Teacher question_en (v2 restatement, audit only): Is the Northern Furse Seal a good pet for a six-year-old child?
- Gold evidence titles (normalised): northern fur seal
- R4 hits: northern fur seal
- Teacher entities: Northern Fur Seal; Child
- unknown_terms: []
- sub_questions (3):
  - What are the characteristics of Northern Furse Seals?
  - What are the needs of a six-year-old child in terms of pets?
  - Can Northern Furse Seals be domesticated?
- Teacher facts (3):
  - Northern Fur Seals are carnivorous mammals that belong to the family Otariidae.
  - Northern Fur Seals are wild animals that live in large colonies in the ocean.
  - Children at the age of six need pets that are safe, easy to care for, and can provide emotional support.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred yes; v1 teacher entities: Norwegian Forest Cat; Child)

#### 24. `84a83f07a543ae3024c9` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ایک ہائی اسکول کے طالب علم کے لیے پیلا صفحات استعمال کرنا غیر معمولی ہوگا؟
- Gold question_en (DEV200, audit only): Would it be uncommon for a high schooler to use the yellow pages?
- Teacher question_en (v2 restatement, audit only): Would it be unusual for a high school student to use yellow pages?
- Gold evidence titles (normalised): adolescence; digital native; high school (north america); information age; yellow pages
- R4 hits: yellow pages
- Teacher entities: High school; Yellow pages
- unknown_terms: []
- sub_questions (4):
  - What is the typical age range of high school students?
  - What are yellow pages typically used for?
  - How common is it for people in that age range to use yellow pages?
  - Are there alternative methods that high school students typically use to find information?
- Teacher facts (4):
  - High school students are typically between the ages of 14 and 18.
  - Yellow pages are telephone directories that list businesses organized by category.
  - The use of yellow pages has declined significantly since the late 1990s with the advent of online directories and search engines.
  - Most high school students have access to the internet and use online search engines to find information.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred no; v1 teacher entities: High school; Yellow Pages)

#### 25. `867a5425c26092b30fbf` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ڈزنی کے سات بونے اصل ہیں؟
- Gold question_en (DEV200, audit only): Are Disney's seven dwarves the original ones?
- Teacher question_en (v2 restatement, audit only): Are Disney's Seven Dwarfs real?
- Gold evidence titles (normalised): seven dwarfs; snow white and the seven dwarfs (1937 film)
- R4 hits: seven dwarfs
- Teacher entities: Seven Dwarfs; Walt Disney
- unknown_terms: []
- sub_questions (3):
  - What are the Seven Dwarfs?
  - What is the origin of the Seven Dwarfs?
  - Are the Seven Dwarfs based on real people?
- Teacher facts (3):
  - The Seven Dwarfs are the main characters in the 1937 Disney animated film Snow White and the Seven Dwarfs.
  - The Seven Dwarfs were created by Walt Disney and his team of animators and writers.
  - The Seven Dwarfs are fictional characters and not based on real people.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: The Walt Disney Company; Snow White and the Seven Dwarfs)

#### 26. `86ead3f15417204affc2` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ڈسٹن ہافمین بنی اسرائیل میں سے ہیں؟
- Gold question_en (DEV200, audit only): Is Dustin Hoffman one of the B'nei Yisrael?
- Teacher question_en (v2 restatement, audit only): Is Dustin Hoffman of Jewish descent?
- Gold evidence titles (normalised): dustin hoffman; indian jews in israel; israelites
- R4 hits: dustin hoffman
- Teacher entities: Dustin Hoffman; Jewish descent
- unknown_terms: []
- sub_questions (2):
  - What is Dustin Hoffman's ancestry?
  - What is the definition of Jewish descent?
- Teacher facts (2):
  - Dustin Hoffman was born to a Jewish family.
  - Jewish descent refers to having ancestry from the Jewish people.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred no; v1 teacher entities: Dustin Hoffman; Jewish people)

#### 27. `8d708927b1593d8b9b42` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ووڈرو ولسن پلسسی بمقابلہ فرگوسن کے فیصلے کی حمایت کریں گے؟
- Gold question_en (DEV200, audit only): Would Woodrow Wilson support Plessy v. Ferguson decision?
- Teacher question_en (v2 restatement, audit only): Would Woodrow Wilson support the decision in Plessy v. Ferguson?
- Gold evidence titles (normalised): plessy v. ferguson; woodrow wilson
- R4 hits: plessy v. ferguson; woodrow wilson
- Teacher entities: Woodrow Wilson; Plessy v. Ferguson; Racial segregation; Civil rights
- unknown_terms: []
- sub_questions (4):
  - What were Woodrow Wilson's views on racial segregation?
  - What was the decision in Plessy v. Ferguson?
  - What were the implications of the decision in Plessy v. Ferguson?
  - What was Woodrow Wilson's stance on civil rights?
- Teacher facts (6) — parse_ok True, repaired brace:
  - Woodrow Wilson was a supporter of racial segregation.
  - The decision in Plessy v. Ferguson upheld the constitutionality of racial segregation laws.
  - Plessy v. Ferguson was a landmark Supreme Court case in the United States.
  - Woodrow Wilson was the 28th President of the United States.
  - Plessy v. Ferguson was decided in 1896.
  - Woodrow Wilson served as President from 1913 to 1921.
- v1 status of this qid: **win** (TF_v1 pred yes; v1 teacher entities: Woodrow Wilson; Plessy v. Ferguson)

#### 28. `9711b3fc634e6f4f471d` — gold **no**, A_r2 —, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا دوسری جنگ عظیم ایلن ٹورنگ کے بغیر ویسی ہی ہوتی؟
- Gold question_en (DEV200, audit only): Would World War II have been the same without Alan Turing?
- Teacher question_en (v2 restatement, audit only): Would World War II have been the same without Alan Turing?
- Gold evidence titles (normalised): alan turing; cryptanalysis of the enigma; enigma machine
- R4 hits: alan turing
- Teacher entities: Alan Turing; World War II; Allied Powers
- unknown_terms: []
- sub_questions (3):
  - What contributions did Alan Turing make to the Allied war effort?
  - How significant were these contributions to the outcome of World War II?
  - What alternatives were available to the Allies for codebreaking and intelligence gathering?
- Teacher facts (5):
  - Alan Turing was a British mathematician, computer scientist, and logician who made significant contributions to codebreaking during World War II.
  - Turing worked at the Government Code and Cypher School at Bletchley Park, where he helped crack the German Enigma code.
  - The Allies relied heavily on intelligence gathered from codebreaking to inform their strategic decisions during the war.
  - The Enigma code was a complex encryption system used by the German military to send secret messages.
  - The Allies had limited success with codebreaking before Turing's contributions, and his work significantly improved their ability to gather intelligence.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred yes; v1 teacher entities: World War II; Alan Turing)

#### 29. `9b491a31c4212e70b18c` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا جین آسٹن کو وسطی بچے کے سنڈروم کا سامنا تھا؟
- Gold question_en (DEV200, audit only): Did Jane Austen suffer from middle child syndrome?
- Teacher question_en (v2 restatement, audit only): Did Jane Austen face Middle Child Syndrome?
- Gold evidence titles (normalised): jane austen; middle child syndrome; timeline of jane austen
- R4 hits: jane austen
- Teacher entities: Jane Austen
- unknown_terms: Middle Child Syndrome
- sub_questions (4):
  - What is Middle Child Syndrome?
  - What are the characteristics of Middle Child Syndrome?
  - What is known about Jane Austen's birth order?
  - How did Jane Austen's family dynamics influence her life?
- Teacher facts (2):
  - Jane Austen was born on December 16, 1775, in Steventon, Hampshire, England.
  - Jane Austen was the second daughter and seventh child of Reverend George Austen and Cassandra Austen.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Jane Austen; Middle child syndrome)

#### 30. `9e78c01675a198781cca` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا مائیکل بلومبرگ مائکرونیشیا کا قرضہ ایک دہائی کے لیے ادا کر سکتے ہیں؟
- Gold question_en (DEV200, audit only): Can Michael Bloomberg fund the debt of Micronesia for a decade?
- Teacher question_en (v2 restatement, audit only): Can Michael Bloomberg pay off Micronesia's debt in a decade?
- Gold evidence titles (normalised): economy of the federated states of micronesia; michael bloomberg; micronesia
- R4 hits: michael bloomberg
- Teacher entities: Michael Bloomberg; Federated States of Micronesia; Net worth; Debt; Annual budget
- unknown_terms: []
- sub_questions (4):
  - What is Michael Bloomberg's net worth?
  - What is Micronesia's current debt?
  - How does debt repayment work in a decade?
  - What are the annual expenses of Micronesia?
- Teacher facts (4) — parse_ok True, repaired brace:
  - Michael Bloomberg's net worth is estimated to be over $70 billion.
  - The Federated States of Micronesia has a current debt of around $93 million.
  - Debt repayment can be calculated using various methods, including the debt snowball method and the debt avalanche method.
  - The annual budget of the Federated States of Micronesia is around $100 million.
- v1 status of this qid: **win** (TF_v1 pred yes; v1 teacher entities: Michael Bloomberg; Federated States of Micronesia; Decade)

#### 31. `abd771e9a20b107e7e56` — gold **yes**, A_r2 no, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا نکولا ٹیسلا کے نام پر متعدد برانڈز بنے ہیں؟
- Gold question_en (DEV200, audit only): Has Nikola Tesla's name inspired multiple brands?
- Teacher question_en (v2 restatement, audit only): Are there multiple brands named after Nikola Tesla?
- Gold evidence titles (normalised): nikola motor company; tesla; tesla electric light and manufacturing; tesla, inc.
- R4 hits: (none)
- Teacher entities: Nikola Tesla
- unknown_terms: []
- sub_questions (3):
  - What is Nikola Tesla known for?
  - Are there any companies or brands named after Nikola Tesla?
  - What are the names of the companies or brands named after Nikola Tesla?
- Teacher facts (3):
  - Nikola Tesla was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist best known for his contributions to the development of the modern alternating current (AC) electricity system.
  - Tesla, Inc. is an American electric vehicle and clean energy company named after Nikola Tesla.
  - Tesla Motors was the former name of Tesla, Inc.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred no; v1 teacher entities: Nikola Tesla; Brand)

#### 32. `b77170e15c0b1bd72340` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا دورین کے پھل خوشبو کے لحاظ سے خوشگوار ہیں؟
- Gold question_en (DEV200, audit only): Are Durian fruits an olfactory delight?
- Teacher question_en (v2 restatement, audit only): Are durian fruits pleasant in terms of smell?
- Gold evidence titles (normalised): durian
- R4 hits: durian
- Teacher entities: Durian
- unknown_terms: []
- sub_questions (3):
  - What is the typical smell of durian fruits?
  - Is the smell of durian fruits generally considered pleasant?
  - What are the characteristics of durian fruits?
- Teacher facts (3):
  - Durian is a tropical fruit native to Southeast Asia.
  - Durian is known for its distinctive odor and spiky exterior.
  - The smell of durian is often compared to rotten eggs, gasoline, or sewage.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Durian; Fruit)

#### 33. `ba04b5ebc2edd682c4dd` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا دی میٹرکس ایک آزاد فلم ہے؟
- Gold question_en (DEV200, audit only): Is the Matrix a standalone movie?
- Teacher question_en (v2 restatement, audit only): Is The Matrix an independent film?
- Gold evidence titles (normalised): the matrix; the matrix (franchise)
- R4 hits: the matrix
- Teacher entities: The Matrix; Independent film
- unknown_terms: []
- sub_questions (3):
  - What is the definition of an independent film?
  - Who produced The Matrix?
  - Who distributed The Matrix?
- Teacher facts (3):
  - An independent film is a film produced outside the major film studio system.
  - The Matrix was produced by Joel Silver and the Wachowskis.
  - The Matrix was distributed by Warner Bros. Pictures.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred yes; v1 teacher entities: (none))

#### 34. `c67945e7c471a9692c88` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا برفانی الو سونوران میں زندہ رہ سکتا ہے؟
- Gold question_en (DEV200, audit only): Could a snowy owl survive in the Sonoran?
- Teacher question_en (v2 restatement, audit only): Can the Snowy Owl live in Sonora?
- Gold evidence titles (normalised): arctic; climate of the arctic; snowy owl; sonoran desert
- R4 hits: snowy owl
- Teacher entities: Snowy Owl; Sonora
- unknown_terms: []
- sub_questions (4):
  - What is the typical habitat of the Snowy Owl?
  - What is the geographical location of Sonora?
  - What are the climate and geography of Sonora like?
  - Can the Snowy Owl adapt to different climates and geographies?
- Teacher facts (4):
  - The Snowy Owl is native to the Arctic regions.
  - Sonora is a state located in the northwest of Mexico.
  - Sonora has a diverse geography, including deserts, mountains, and coastal areas.
  - The Snowy Owl is adapted to life in the Arctic tundra.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Snowy Owl; Sonoran Desert)

#### 35. `d15a33b8f88dc5ce32b0` — gold **no**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا ڈھولک نوازوں کو اضافی تاروں کی ضرورت ہوتی ہے؟
- Gold question_en (DEV200, audit only): Do drummers need spare strings?
- Teacher question_en (v2 restatement, audit only): Do dholki players need additional strings?
- Gold evidence titles (normalised): drum; drummer; string instrument
- R4 hits: (none)
- Teacher entities: Dholki
- unknown_terms: []
- sub_questions (3):
  - What is the typical string configuration of a dholki?
  - Do dholki players often modify their instruments?
  - What is the purpose of adding extra strings to a musical instrument?
- Teacher facts (3):
  - A dholki is a two-headed hand drum from the Indian subcontinent.
  - The typical string configuration of a dholki is not standardized.
  - Some musicians modify their instruments to suit their playing style or musical genre.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Dholak; Stringed instrument)

#### 36. `d1dbb41001ab4ad3b924` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا نیلے ہونٹ معمول کی بات ہے؟
- Gold question_en (DEV200, audit only): Are blue lips normal?
- Teacher question_en (v2 restatement, audit only): Is having blue lips normal?
- Gold evidence titles (normalised): cyanosis; oxygen saturation (medicine)
- R4 hits: cyanosis
- Teacher entities: Human lips; Cyanosis
- unknown_terms: []
- sub_questions (3):
  - What is the normal color of human lips?
  - What causes blue discoloration of the lips?
  - Is blue discoloration of the lips a common condition?
- Teacher facts (3):
  - The normal color of human lips varies from person to person, but is typically pink or reddish-pink.
  - Cyanosis is a condition characterized by a bluish discoloration of the skin and mucous membranes, including the lips.
  - Cyanosis is usually caused by inadequate oxygenation of the blood.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Blue lips; Normality)

#### 37. `f087d393e2ab1f8783f5` — gold **no**, A_r2 yes, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا 1980 کے امریکہ کے صدارتی انتخابات کے نتائج کو فوٹو فِنِش کہا جا سکتا ہے؟
- Gold question_en (DEV200, audit only): Can 1980 United States presidential election result be considered a photo finish?
- Teacher question_en (v2 restatement, audit only): Can the results of the 1980 United States presidential election be described as a photo finish?
- Gold evidence titles (normalised): 1980 united states presidential election; history of the united states (1980–1991)
- R4 hits: 1980 united states presidential election
- Teacher entities: Photo finish; 1980 United States presidential election
- unknown_terms: []
- sub_questions (4):
  - What is the definition of a photo finish?
  - What were the results of the 1980 United States presidential election?
  - How close were the results of the 1980 United States presidential election?
  - What is the typical margin of victory in a photo finish?
- Teacher facts (6):
  - A photo finish is a term used to describe a very close finish in a competition, often requiring a photograph to determine the winner.
  - The 1980 United States presidential election was won by Republican candidate Ronald Reagan, who defeated incumbent President Jimmy Carter.
  - Ronald Reagan won the election by a margin of 489 to 49 in the Electoral College.
  - The popular vote margin was 43.9% to 41% in favor of Reagan.
  - A photo finish typically involves a margin of victory of less than 1% or a few percentage points.
  - The term photo finish originated in horse racing, where it was used to describe a close finish between two or more horses.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: 1980 United States presidential election; Photo finish)

#### 38. `f2859b2ce17b5f5a6ad9` — gold **no**, A_r2 yes, TF no, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا چنچیلا نسلِ فیلس کاتس چوہے کی قسم ہے؟
- Gold question_en (DEV200, audit only): Is a Chinchilla breed of felis catus a type of rodent?
- Teacher question_en (v2 restatement, audit only): Is Chinchilla a type of Felis catus mouse?
- Gold evidence titles (normalised): cat; chinchilla; persian cat; rodent
- R4 hits: (none)
- Teacher entities: Chinchilla (genus); Felis catus
- unknown_terms: Felis catus mouse
- sub_questions (3):
  - What is the scientific classification of Chinchilla?
  - What is the scientific classification of Felis catus?
  - Is Felis catus a type of mouse?
- Teacher facts (3):
  - Chinchilla is a genus of medium-sized rodents native to the Andes mountains in South America.
  - Felis catus is a domesticated mammal and a subspecies of the wildcat.
  - Felis catus is not a type of mouse, but rather a type of cat.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Chinchilla; Rodent; Felis catus)

#### 39. `f7a32d14efd589015fe1` — gold **no**, A_r2 —, TF no, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا ہرپیس سمپلیکس وائرس زہرہ پر پھیل سکتا ہے؟
- Gold question_en (DEV200, audit only): Can Herpes simplex virus spread on Venus?
- Teacher question_en (v2 restatement, audit only): Can the herpes simplex virus spread on Venus?
- Gold evidence titles (normalised): herpes simplex virus; sterilization (microbiology); venus; virus
- R4 hits: herpes simplex virus; venus
- Teacher entities: Herpes simplex virus; Venus
- unknown_terms: []
- sub_questions (4):
  - Can the herpes simplex virus survive in extreme temperatures?
  - Can the herpes simplex virus survive in a carbon dioxide-rich environment?
  - Is the environment on Venus suitable for the herpes simplex virus to spread?
  - What is the primary method of transmission of the herpes simplex virus?
- Teacher facts (6):
  - The herpes simplex virus is a human pathogen that can survive on surfaces for a short period of time.
  - The herpes simplex virus is typically spread through skin-to-skin contact with an infected person.
  - The surface temperature of Venus reaches as high as 462°C (863°F).
  - The atmosphere of Venus is composed mainly of carbon dioxide.
  - The herpes simplex virus requires a human host to replicate and survive.
  - The primary method of transmission of the herpes simplex virus is through direct contact with an infected person's lesions or bodily fluids.
- v1 status of this qid: **win** (TF_v1 pred no; v1 teacher entities: Herpes simplex virus; Venus)

#### 40. `f7b1cc5b3fab95aa1be4` — gold **yes**, A_r2 no, TF yes, R4 hit no

- Question (Urdu, from the sealed A_r2 prompt): کیا چھاتی کے کینسر کا تعلق ربن سے ہے؟
- Gold question_en (DEV200, audit only): Is breast cancer associated with a ribbon?
- Teacher question_en (v2 restatement, audit only): Is there a relationship between breast cancer and ribbons?
- Gold evidence titles (normalised): awareness ribbon
- R4 hits: (none)
- Teacher entities: Breast cancer; Ribbons
- unknown_terms: []
- sub_questions (3):
  - What is the significance of ribbons in breast cancer awareness?
  - Is there a scientific link between ribbons and breast cancer?
  - Are ribbons used as a symbol for breast cancer?
- Teacher facts (3):
  - Breast cancer is a type of cancer that forms in the cells of the breasts.
  - Ribbons are often used as a symbol for breast cancer awareness, particularly the pink ribbon.
  - The pink ribbon is an international symbol of breast cancer awareness.
- v1 status of this qid: **concordant-wrong** (TF_v1 pred no; v1 teacher entities: Breast Cancer; Rib)

#### 41. `fba624d8c01833419760` — gold **yes**, A_r2 no, TF yes, R4 hit yes

- Question (Urdu, from the sealed A_r2 prompt): کیا اگست دنیا کے کسی حصے کے لیے سردیوں کا مہینہ ہے؟
- Gold question_en (DEV200, audit only): Is August a winter month for part of the world?
- Teacher question_en (v2 restatement, audit only): Is August a winter month in any part of the world?
- Gold evidence titles (normalised): august; summer; winter
- R4 hits: august
- Teacher entities: August; Season; Southern Hemisphere; Northern Hemisphere
- unknown_terms: []
- sub_questions (3):
  - In which months do the Southern Hemisphere and the Northern Hemisphere experience winter?
  - Is August a month of winter in the Southern Hemisphere?
  - Is August a month of winter in the Northern Hemisphere?
- Teacher facts (6):
  - The Southern Hemisphere experiences winter in June, July, and August.
  - The Northern Hemisphere experiences winter in December, January, and February.
  - August is a month of winter in Antarctica.
  - August is a month of winter in some parts of South America, such as Argentina and Chile.
  - August is a month of winter in some parts of Australia, such as New South Wales and Victoria.
  - August is a month of winter in some parts of New Zealand.
- v1 status of this qid: **win** (TF_v1 pred yes; v1 teacher entities: August; Season; Southern Hemisphere; Northern Hemisphere)

---

## 9. Full 200-row join

Columns: gold; pred / ✓ (1 = correct; — = unparsed, counted wrong) for A_r2_v1, TF_v1, A_r2_v2, TF_v2, TF_wrong_v2, TF_noq_v2; v2 teacher parse_ok (1/0), repaired (b = brace, k = bracket, · = none), n_facts, n_unknown; R4 hit v1 / v2 (y/n); v1 → v2 status of TF vs A_r2 (W = win, L = loss, R = concordant-right, X = concordant-wrong). Row order = file order.

| # | qid | gold | A_r2_v1 | TF_v1 | A_r2_v2 | TF_v2 | TF_wrong_v2 | TF_noq_v2 | parse_ok | rep | n_facts | n_unk | R4 v1 | R4 v2 | st v1→v2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `00e13b8d03d35929c049` | no | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | 1 | · | 2 | 0 | n | y | X→X |
| 2 | `01430717f590ad3f4e74` | no | no/1 | no/1 | no/1 | no/1 | no/1 | —/0 | 1 | · | 4 | 0 | y | n | R→R |
| 3 | `018f6ea10facddae3b2e` | yes | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | —/0 | 1 | · | 2 | 0 | y | y | R→L |
| 4 | `01c27885cede6741bc53` | yes | yes/1 | no/0 | yes/1 | no/0 | no/0 | —/0 | 1 | b | 3 | 0 | y | y | L→L |
| 5 | `05e9cdc44f1b066badd7` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | —/0 | 1 | · | 4 | 0 | y | y | W→W |
| 6 | `06215d4b6f1a4b975d9c` | no | no/1 | no/1 | no/1 | yes/0 | yes/0 | yes/0 | 1 | b | 5 | 0 | y | y | R→L |
| 7 | `064d8faaa8f164b55270` | yes | no/0 | no/0 | no/0 | yes/1 | no/0 | no/0 | 1 | · | 3 | 0 | y | y | X→W |
| 8 | `06724dc213e0dae715f5` | yes | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | 1 | · | 3 | 0 | y | y | R→R |
| 9 | `068379affae04debca2b` | yes | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | —/0 | 1 | · | 3 | 0 | y | y | R→R |
| 10 | `069c88860654c9c5df0b` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | n | n | R→R |
| 11 | `06ebd3ca1fe98d5796f7` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 12 | `06f7878425a995c2a633` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 13 | `076f3f72b7e6d17073eb` | no | no/1 | no/1 | no/1 | no/1 | no/1 | —/0 | 1 | b | 4 | 0 | y | y | R→R |
| 14 | `076f905ae36408b5ee69` | no | no/1 | no/1 | no/1 | no/1 | no/1 | —/0 | 1 | k | 2 | 0 | y | y | R→R |
| 15 | `0d22526961c82ad6ef4a` | yes | yes/1 | yes/1 | yes/1 | no/0 | no/0 | no/0 | 1 | · | 3 | 0 | y | y | R→L |
| 16 | `0fe3edf35bd1a1e662a9` | no | yes/0 | no/1 | yes/0 | no/1 | yes/0 | —/0 | 1 | · | 4 | 0 | y | y | W→W |
| 17 | `10325585370ba368ca9a` | no | yes/0 | yes/0 | yes/0 | yes/0 | no/1 | yes/0 | 1 | · | 2 | 0 | y | y | X→X |
| 18 | `1534ef2130cb48d6abb4` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | no/0 | 1 | · | 4 | 0 | y | y | X→X |
| 19 | `15885efe6f91724c16d9` | yes | yes/1 | no/0 | yes/1 | no/0 | —/0 | no/0 | 1 | b | 6 | 0 | y | y | L→L |
| 20 | `16c08e5f354b035c41a5` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 21 | `16d7c42444ef13ef4643` | no | yes/0 | yes/0 | yes/0 | no/1 | yes/0 | —/0 | 1 | · | 3 | 0 | n | y | X→W |
| 22 | `17b8fa74f450c4d0e56b` | yes | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | yes/1 | 1 | · | 3 | 0 | n | y | R→L |
| 23 | `1a4eb85c31188af515b0` | yes | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | 1 | · | 3 | 0 | y | y | R→R |
| 24 | `1c56958c04a98ccd5d1e` | no | no/1 | no/1 | no/1 | yes/0 | no/1 | yes/0 | 1 | · | 3 | 0 | n | n | R→L |
| 25 | `1d923b8a2778448a9716` | no | no/1 | no/1 | no/1 | no/1 | —/0 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 26 | `1eb8a833f2f5f1698b52` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 27 | `20693ee66ad7ca890e08` | no | no/1 | no/1 | no/1 | no/1 | no/1 | —/0 | 1 | · | 6 | 0 | y | y | R→R |
| 28 | `20abc7ba538a2fd1dd76` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 4 | 0 | y | n | R→R |
| 29 | `24418db1d18a7fdf79f3` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | no/0 | 1 | b | 6 | 0 | y | y | X→X |
| 30 | `24a3aff40ef7fd04e208` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | yes/1 | 1 | b | 4 | 0 | y | y | X→X |
| 31 | `25a088d9d2ce674e639a` | yes | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | 1 | b | 6 | 0 | n | n | R→R |
| 32 | `265dd54c248f8b048851` | no | no/1 | no/1 | no/1 | no/1 | yes/0 | yes/0 | 1 | · | 0 | 1 | n | n | R→R |
| 33 | `267609f49ccd3c0839c3` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 34 | `27368c21e50b6af694ab` | no | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | 1 | · | 4 | 0 | n | y | X→X |
| 35 | `2a4e1d4524997b6263bf` | no | no/1 | yes/0 | no/1 | no/1 | —/0 | yes/0 | 1 | · | 4 | 0 | y | y | L→R |
| 36 | `2aba85e0e138ad86035e` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 37 | `2bc94c6dfc79d7a86fa2` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 38 | `2c9f3c00412a4018380f` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | 1 | · | 0 | 1 | n | n | R→R |
| 39 | `2e508efbf0b72f1af2c2` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | no/0 | 1 | · | 4 | 0 | y | y | X→X |
| 40 | `2e8e9e51f0f290e32d27` | yes | yes/1 | no/0 | yes/1 | no/0 | yes/1 | —/0 | 1 | · | 3 | 0 | n | n | L→L |
| 41 | `300927fd135e8d94af16` | yes | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | 1 | b | 6 | 0 | y | y | R→R |
| 42 | `30fcf62f58849cb4ab05` | yes | no/0 | yes/1 | no/0 | no/0 | no/0 | no/0 | 1 | · | 3 | 0 | y | y | W→X |
| 43 | `310968736449680ee61e` | no | no/1 | no/1 | no/1 | yes/0 | no/1 | no/1 | 1 | · | 2 | 0 | n | n | R→L |
| 44 | `315575d5cbdd1d154744` | yes | yes/1 | no/0 | yes/1 | no/0 | no/0 | —/0 | 1 | · | 2 | 1 | y | y | L→L |
| 45 | `3157a0f428531ebed7b9` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | no/0 | 1 | · | 3 | 0 | y | y | R→R |
| 46 | `3295844627a9bd1b9135` | yes | yes/1 | yes/1 | yes/1 | yes/1 | —/0 | yes/1 | 1 | · | 4 | 0 | y | y | R→R |
| 47 | `3486b6b5c69878bb088d` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | W→W |
| 48 | `3503e8d41364eb0b33bc` | no | —/0 | no/1 | —/0 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | n | n | W→W |
| 49 | `3570d57f16a40488ff42` | yes | yes/1 | no/0 | yes/1 | no/0 | no/0 | —/0 | 1 | · | 4 | 0 | y | y | L→L |
| 50 | `35a5b27d84294ebf1fe0` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | —/0 | 1 | · | 2 | 0 | n | n | X→X |
| 51 | `38ef97eb7cdd4200fd00` | yes | yes/1 | no/0 | yes/1 | no/0 | no/0 | no/0 | 1 | · | 2 | 0 | n | n | L→L |
| 52 | `3a61dcbed7e358b3aee8` | yes | no/0 | no/0 | no/0 | yes/1 | no/0 | yes/1 | 1 | b | 4 | 0 | y | y | X→W |
| 53 | `3ae8644164c547560eda` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 54 | `3b1834a6dd331eb1d791` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 55 | `3b38e00d8d1da26f7f02` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | n | n | W→W |
| 56 | `3bba4ea27500361cb141` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 57 | `3cec996f26ec99d0a519` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 2 | 1 | y | y | R→R |
| 58 | `3d7050c0e9bfbeef27c1` | yes | no/0 | yes/1 | no/0 | yes/1 | no/0 | yes/1 | 1 | · | 2 | 0 | y | y | W→W |
| 59 | `3de88c80f0914eb08caf` | no | no/1 | no/1 | no/1 | yes/0 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→L |
| 60 | `3ebb62af96f836d2d4d6` | no | no/1 | no/1 | no/1 | yes/0 | no/1 | no/1 | 1 | b | 4 | 0 | n | n | R→L |
| 61 | `3f5af97fdff20028347b` | no | yes/0 | yes/0 | yes/0 | no/1 | no/1 | —/0 | 1 | · | 3 | 0 | y | y | X→W |
| 62 | `3f8a1bd6bf3a967cdeb6` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | b | 4 | 0 | y | y | R→R |
| 63 | `3fc11f59d64ea3b65136` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | —/0 | 1 | · | 3 | 0 | n | n | R→R |
| 64 | `4125c5f4c7a536916608` | yes | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | 1 | b | 4 | 0 | n | n | R→R |
| 65 | `42c947475244ed02629e` | yes | yes/1 | no/0 | yes/1 | no/0 | no/0 | no/0 | 1 | · | 3 | 0 | y | y | L→L |
| 66 | `43da475d5c2db929c458` | yes | no/0 | no/0 | no/0 | yes/1 | no/0 | yes/1 | 1 | · | 2 | 0 | y | y | X→W |
| 67 | `4589898e4852e5389728` | yes | yes/1 | no/0 | yes/1 | yes/1 | yes/1 | yes/1 | 1 | · | 1 | 1 | y | y | L→R |
| 68 | `46428dac0ceefa1ad73b` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | no/0 | 1 | · | 3 | 0 | n | y | X→X |
| 69 | `4760fe3d0c80beb30edd` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | W→W |
| 70 | `4773d577873ff9e04a88` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 71 | `4810297f474b0e97ed78` | yes | no/0 | yes/1 | no/0 | no/0 | no/0 | yes/1 | 1 | b | 4 | 0 | y | y | W→X |
| 72 | `48da75d87c66754ccc2e` | yes | yes/1 | no/0 | yes/1 | no/0 | no/0 | yes/1 | 1 | · | 4 | 0 | y | y | L→L |
| 73 | `4bbf4b169b4981ad5a34` | no | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | 1 | · | 3 | 0 | y | y | X→X |
| 74 | `4de177ae8c827fd0ffb1` | no | no/1 | yes/0 | no/1 | no/1 | yes/0 | —/0 | 1 | b | 4 | 0 | y | y | L→R |
| 75 | `4e1b65e81ec09397b26e` | yes | no/0 | no/0 | no/0 | yes/1 | no/0 | yes/1 | 1 | · | 4 | 0 | n | y | X→W |
| 76 | `4ec8cd118be8a53ec516` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | no/1 | 1 | · | 4 | 0 | y | y | W→W |
| 77 | `502d4dae6f08e73a5569` | yes | no/0 | yes/1 | no/0 | yes/1 | yes/1 | no/0 | 1 | · | 3 | 0 | n | n | W→W |
| 78 | `557e389d7efe47c5bc4d` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | yes/1 | 1 | · | 3 | 0 | y | y | X→X |
| 79 | `5897ec22db850f7b416e` | no | no/1 | yes/0 | no/1 | no/1 | no/1 | —/0 | 1 | · | 4 | 0 | y | y | L→R |
| 80 | `593e66cbdec8e3a15852` | no | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | —/0 | 1 | · | 2 | 0 | n | y | X→X |
| 81 | `598be3dcddbd7775a827` | no | no/1 | yes/0 | no/1 | no/1 | no/1 | no/1 | 1 | b | 6 | 0 | y | y | L→R |
| 82 | `5ab66b27a0caee886970` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 83 | `5e2d9519c7c521411817` | yes | —/0 | yes/1 | —/0 | yes/1 | no/0 | —/0 | 1 | · | 3 | 0 | n | y | W→W |
| 84 | `5f24cbd41798ed5c7608` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 2 | 0 | y | y | R→R |
| 85 | `63fb8f40ca6c2226dd01` | yes | no/0 | no/0 | no/0 | —/0 | —/0 | —/0 | 1 | · | 3 | 0 | n | n | X→X |
| 86 | `64354cb6ee6d987d4f0f` | yes | no/0 | no/0 | no/0 | no/0 | —/0 | no/0 | 1 | · | 3 | 0 | y | y | X→X |
| 87 | `660bfdc17b47f42facac` | yes | yes/1 | no/0 | yes/1 | no/0 | no/0 | yes/1 | 1 | · | 3 | 1 | n | y | L→L |
| 88 | `67079beff747978ed4c2` | no | yes/0 | yes/0 | yes/0 | yes/0 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | X→X |
| 89 | `684d4c03be354635b80f` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 1 | y | n | R→R |
| 90 | `690a36d5fed5b17969aa` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | yes/0 | 1 | · | 2 | 0 | y | y | W→W |
| 91 | `6c43fa359095fd0845f5` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 3 | 0 | y | y | R→R |
| 92 | `6f4c8a2789c0305c4f63` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | —/0 | 1 | · | 3 | 0 | y | y | R→R |
| 93 | `703db4629ff46d82a205` | no | no/1 | no/1 | no/1 | yes/0 | yes/0 | yes/0 | 1 | · | 3 | 1 | y | y | R→L |
| 94 | `70b2122323424f6e80c2` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | yes/0 | 1 | · | 4 | 0 | y | y | W→W |
| 95 | `71ca1a8a75c5c582ab3f` | no | —/0 | no/1 | —/0 | no/1 | no/1 | —/0 | 1 | b | 6 | 0 | y | n | W→W |
| 96 | `7281474f2760dce03f39` | no | no/1 | no/1 | no/1 | no/1 | yes/0 | —/0 | 1 | · | 3 | 0 | n | n | R→R |
| 97 | `73c52134ab2a903d86db` | no | yes/0 | no/1 | yes/0 | no/1 | yes/0 | yes/0 | 1 | · | 3 | 0 | y | y | W→W |
| 98 | `73ca2ef1da65b2a2ebe6` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | 1 | · | 3 | 0 | y | y | R→R |
| 99 | `7419826e4373eb1e83ff` | yes | no/0 | yes/1 | no/0 | yes/1 | no/0 | —/0 | 1 | · | 3 | 0 | y | y | W→W |
| 100 | `7571100f05bc56919c78` | yes | no/0 | yes/1 | no/0 | no/0 | no/0 | yes/1 | 1 | · | 3 | 0 | y | y | W→X |
| 101 | `7870b1cef39a4f685911` | yes | yes/1 | no/0 | yes/1 | no/0 | no/0 | no/0 | 1 | · | 3 | 0 | n | n | L→L |
| 102 | `7b84d2bc643ddc2085f0` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | —/0 | 1 | · | 3 | 0 | y | y | W→W |
| 103 | `7be51fdef30345de666e` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 3 | 0 | y | y | R→R |
| 104 | `7c3759cc1da78e9fbd79` | yes | no/0 | yes/1 | no/0 | no/0 | no/0 | no/0 | 1 | · | 6 | 0 | y | y | W→X |
| 105 | `7dc74fd9b62e514750c2` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | no/0 | 1 | · | 3 | 0 | y | y | X→X |
| 106 | `7ea958a52fc626ac5577` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | b | 4 | 0 | y | y | R→R |
| 107 | `7f139bf51273517efd4e` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | yes/1 | 1 | · | 3 | 0 | y | y | X→X |
| 108 | `7f4effbc97ab2b5fd4a7` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | —/0 | 1 | b | 4 | 0 | y | y | R→R |
| 109 | `806a13f37055bd21fddc` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 3 | 0 | y | y | R→R |
| 110 | `80aa769f55b14c1e4d8d` | yes | yes/1 | no/0 | yes/1 | yes/1 | no/0 | —/0 | 1 | · | 6 | 0 | y | y | L→R |
| 111 | `80ca095e38ab73b908ff` | no | yes/0 | yes/0 | yes/0 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | n | y | X→W |
| 112 | `84a83f07a543ae3024c9` | yes | no/0 | no/0 | no/0 | yes/1 | —/0 | no/0 | 1 | · | 4 | 0 | y | y | X→W |
| 113 | `867a5425c26092b30fbf` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | n | y | W→W |
| 114 | `86ead3f15417204affc2` | yes | no/0 | no/0 | no/0 | yes/1 | no/0 | yes/1 | 1 | · | 2 | 0 | y | y | X→W |
| 115 | `8cfde6ee28d059a5aff6` | yes | yes/1 | yes/1 | yes/1 | no/0 | —/0 | yes/1 | 1 | · | 4 | 0 | n | y | R→L |
| 116 | `8d06b619a7045ed02f51` | no | yes/0 | yes/0 | yes/0 | yes/0 | no/1 | —/0 | 1 | · | 3 | 0 | y | y | X→X |
| 117 | `8d3ddaee20ad48edc066` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | 1 | · | 5 | 0 | y | y | R→R |
| 118 | `8d708927b1593d8b9b42` | yes | no/0 | yes/1 | no/0 | yes/1 | no/0 | —/0 | 1 | b | 6 | 0 | y | y | W→W |
| 119 | `8e03ed9fce31618b095a` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | no/0 | 1 | · | 4 | 0 | n | n | X→X |
| 120 | `8eecf2ff459ef5a9b98a` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | —/0 | 1 | · | 3 | 0 | y | y | R→R |
| 121 | `9052063382d30870a2ed` | no | —/0 | yes/0 | —/0 | yes/0 | no/1 | —/0 | 1 | · | 6 | 0 | n | n | X→X |
| 122 | `90fc9a677ae0df6f2974` | no | no/1 | no/1 | no/1 | no/1 | yes/0 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 123 | `91917c5e30dbf8552b34` | no | no/1 | no/1 | no/1 | yes/0 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→L |
| 124 | `9247b5afb2eef4c9f95b` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | 1 | · | 4 | 0 | y | y | R→R |
| 125 | `92c263f997f09dbb74a8` | no | no/1 | no/1 | no/1 | yes/0 | no/1 | yes/0 | 1 | · | 2 | 1 | y | y | R→L |
| 126 | `9384dfc0aabc73026925` | yes | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | yes/1 | 1 | · | 4 | 0 | n | n | R→L |
| 127 | `9576ab36999e196b87bb` | no | no/1 | yes/0 | no/1 | no/1 | yes/0 | no/1 | 1 | · | 4 | 1 | y | y | L→R |
| 128 | `959fb200fccf056f00d7` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 2 | 1 | y | y | R→R |
| 129 | `9711b3fc634e6f4f471d` | no | —/0 | yes/0 | —/0 | no/1 | no/1 | no/1 | 1 | · | 5 | 0 | y | y | X→W |
| 130 | `9850eb1534206facd067` | no | no/1 | no/1 | no/1 | yes/0 | no/1 | yes/0 | 1 | b | 6 | 0 | y | y | R→L |
| 131 | `987260ffef60cb2c5439` | yes | yes/1 | no/0 | yes/1 | no/0 | no/0 | yes/1 | 1 | · | 1 | 1 | y | y | L→L |
| 132 | `9883d6c816d4855035ea` | yes | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | 1 | · | 2 | 0 | y | y | R→R |
| 133 | `9b491a31c4212e70b18c` | no | yes/0 | no/1 | yes/0 | no/1 | yes/0 | yes/0 | 1 | · | 2 | 1 | y | y | W→W |
| 134 | `9bb59488a227a5f5cd62` | no | no/1 | no/1 | no/1 | yes/0 | no/1 | yes/0 | 1 | b | 4 | 0 | y | y | R→L |
| 135 | `9cfa675eeebfbbdb2487` | no | yes/0 | yes/0 | yes/0 | yes/0 | no/1 | no/1 | 1 | b | 4 | 0 | y | y | X→X |
| 136 | `9e78c01675a198781cca` | yes | no/0 | yes/1 | no/0 | yes/1 | no/0 | yes/1 | 1 | b | 4 | 0 | y | y | W→W |
| 137 | `9fee16a5e703c2fdcd33` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 2 | 0 | n | n | R→R |
| 138 | `a0896de3fd13cd0f3e16` | no | no/1 | yes/0 | no/1 | yes/0 | no/1 | yes/0 | 1 | · | 2 | 1 | y | y | L→L |
| 139 | `a2dfceff60d5b18fc70d` | yes | —/0 | no/0 | —/0 | no/0 | no/0 | no/0 | 1 | b | 4 | 0 | y | y | X→X |
| 140 | `a406e06cd0164c1c6d15` | yes | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | —/0 | 1 | · | 3 | 0 | n | n | R→R |
| 141 | `a4de29904c45a2d17511` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | 1 | · | 3 | 0 | n | n | R→R |
| 142 | `a54b16376f82a6e14bf9` | no | no/1 | yes/0 | no/1 | no/1 | yes/0 | no/1 | 1 | · | 4 | 0 | n | n | L→R |
| 143 | `a6f593e93ca87c53a0a1` | yes | no/0 | yes/1 | no/0 | no/0 | no/0 | no/0 | 1 | b | 5 | 0 | y | y | W→X |
| 144 | `a895b73bdfb2397b2e27` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | —/0 | 1 | · | 6 | 0 | n | y | R→R |
| 145 | `a8bbf2e84d0649b1c52c` | no | no/1 | no/1 | no/1 | no/1 | no/1 | —/0 | 1 | b | 4 | 0 | n | n | R→R |
| 146 | `abd771e9a20b107e7e56` | yes | no/0 | no/0 | no/0 | yes/1 | no/0 | yes/1 | 1 | · | 3 | 0 | n | n | X→W |
| 147 | `acb7edac010554509519` | yes | no/0 | yes/1 | no/0 | no/0 | no/0 | no/0 | 1 | · | 6 | 0 | y | y | W→X |
| 148 | `ae0891799990442fbb0e` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | no/0 | 1 | b | 4 | 0 | y | y | X→X |
| 149 | `b2dddce981183d5a083d` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | no/0 | 1 | · | 3 | 0 | y | y | X→X |
| 150 | `b3443bf3c312c9d144e4` | no | no/1 | no/1 | no/1 | no/1 | no/1 | —/0 | 1 | · | 3 | 0 | y | y | R→R |
| 151 | `b4414a29cba573a24c6c` | no | no/1 | no/1 | no/1 | no/1 | no/1 | —/0 | 1 | b | 4 | 0 | y | y | R→R |
| 152 | `b64be915de0e671e8548` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 2 | 0 | y | y | R→R |
| 153 | `b6713901d33bc1020596` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 4 | 0 | y | y | R→R |
| 154 | `b747938f597b09e43603` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | 1 | · | 1 | 1 | y | n | R→R |
| 155 | `b77170e15c0b1bd72340` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | yes/0 | 1 | · | 3 | 0 | y | y | W→W |
| 156 | `b8df6ac181c4c5b280f9` | no | no/1 | —/0 | no/1 | no/1 | —/0 | —/0 | 1 | b | 6 | 0 | y | y | L→R |
| 157 | `ba04b5ebc2edd682c4dd` | no | yes/0 | yes/0 | yes/0 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | n | y | X→W |
| 158 | `bcd77b200a2b90ca2ecc` | yes | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | yes/1 | 0 | · | 0 | 0 | y | n | R→R |
| 159 | `bd734e6cb6674e25dbe8` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | b | 4 | 0 | y | y | R→R |
| 160 | `bfb0f6c573076a1a0634` | no | no/1 | no/1 | no/1 | no/1 | no/1 | —/0 | 1 | · | 4 | 0 | y | y | R→R |
| 161 | `c51fd5841358ded6eaec` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 4 | 0 | y | y | R→R |
| 162 | `c67945e7c471a9692c88` | no | yes/0 | no/1 | yes/0 | no/1 | yes/0 | no/1 | 1 | · | 4 | 0 | y | y | W→W |
| 163 | `c70d40b57fa67ea13ed6` | no | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | 1 | · | 3 | 0 | y | y | X→X |
| 164 | `c921bdade889def26cf4` | yes | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | —/0 | 1 | · | 4 | 0 | y | y | R→L |
| 165 | `c928aaf3842cb87e3642` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 5 | 1 | y | y | R→R |
| 166 | `cac049d9ec0b86a7f911` | no | yes/0 | yes/0 | yes/0 | yes/0 | no/1 | yes/0 | 1 | · | 4 | 0 | y | y | X→X |
| 167 | `cac8e6300a98cf6128af` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | b | 6 | 0 | n | n | R→R |
| 168 | `cb9eae3948f28e236477` | no | no/1 | no/1 | no/1 | no/1 | —/0 | no/1 | 1 | · | 4 | 0 | y | y | R→R |
| 169 | `d0145341694e7034555d` | no | yes/0 | no/1 | yes/0 | yes/0 | no/1 | no/1 | 1 | b | 6 | 0 | y | y | W→X |
| 170 | `d0183768701c74f966c5` | no | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | yes/0 | 1 | · | 3 | 0 | y | y | X→X |
| 171 | `d1142162a82aab83611e` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | R→R |
| 172 | `d15a33b8f88dc5ce32b0` | no | yes/0 | no/1 | yes/0 | no/1 | yes/0 | no/1 | 1 | · | 3 | 0 | n | n | W→W |
| 173 | `d1dbb41001ab4ad3b924` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | —/0 | 1 | · | 3 | 0 | n | y | W→W |
| 174 | `d47120efd0f09badd848` | yes | yes/1 | no/0 | yes/1 | yes/1 | no/0 | yes/1 | 1 | · | 3 | 0 | y | y | L→R |
| 175 | `d808a0c72dbe605309ab` | no | no/1 | no/1 | no/1 | no/1 | no/1 | no/1 | 1 | · | 2 | 0 | y | y | R→R |
| 176 | `d8fd81e475df8c2f6972` | no | no/1 | no/1 | no/1 | no/1 | —/0 | no/1 | 1 | · | 6 | 0 | y | n | R→R |
| 177 | `d9aeae10998c093c0cc9` | no | no/1 | no/1 | no/1 | yes/0 | no/1 | yes/0 | 1 | b | 4 | 0 | y | y | R→L |
| 178 | `dabf94934721c26b2422` | no | no/1 | yes/0 | no/1 | yes/0 | no/1 | —/0 | 1 | b | 6 | 0 | y | y | L→L |
| 179 | `dce3695a2aaa8d7dba5a` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 4 | 0 | y | y | R→R |
| 180 | `de62d38623fde2c2285a` | yes | yes/1 | yes/1 | yes/1 | yes/1 | no/0 | yes/1 | 1 | · | 4 | 0 | y | n | R→R |
| 181 | `e12a00a2c45fcb4f38e7` | no | no/1 | no/1 | no/1 | no/1 | no/1 | —/0 | 1 | · | 3 | 0 | y | y | R→R |
| 182 | `e12fef0504a959e49b23` | no | no/1 | yes/0 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 3 | 0 | y | y | L→R |
| 183 | `e26a5601edb90a738d56` | no | no/1 | no/1 | no/1 | no/1 | yes/0 | yes/0 | 1 | · | 2 | 1 | y | y | R→R |
| 184 | `e530682239bb5e59ebf0` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 0 | · | 0 | 0 | y | n | R→R |
| 185 | `e6d3973ed3feb8a42928` | yes | yes/1 | no/0 | yes/1 | no/0 | yes/1 | —/0 | 1 | · | 3 | 0 | y | y | L→L |
| 186 | `e87b63e92165b417d37f` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | yes/1 | 1 | · | 2 | 0 | y | y | X→X |
| 187 | `f07a6acf27e25dedbb8d` | no | no/1 | no/1 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 2 | 0 | y | y | R→R |
| 188 | `f087d393e2ab1f8783f5` | no | yes/0 | no/1 | yes/0 | no/1 | no/1 | no/1 | 1 | · | 6 | 0 | y | y | W→W |
| 189 | `f2859b2ce17b5f5a6ad9` | no | yes/0 | no/1 | yes/0 | no/1 | yes/0 | no/1 | 1 | · | 3 | 1 | y | n | W→W |
| 190 | `f43533225534420816d6` | yes | yes/1 | no/0 | yes/1 | yes/1 | yes/1 | no/0 | 1 | · | 3 | 0 | y | y | L→R |
| 191 | `f4a336168f5165de7f0d` | yes | no/0 | yes/1 | no/0 | no/0 | no/0 | —/0 | 1 | · | 6 | 0 | y | y | W→X |
| 192 | `f4a731f739533c52becd` | yes | no/0 | no/0 | no/0 | no/0 | no/0 | yes/1 | 1 | · | 3 | 0 | y | y | X→X |
| 193 | `f69c8412d57ad726d658` | yes | yes/1 | yes/1 | yes/1 | yes/1 | —/0 | —/0 | 1 | · | 2 | 0 | y | y | R→R |
| 194 | `f7a32d14efd589015fe1` | no | —/0 | no/1 | —/0 | no/1 | yes/0 | no/1 | 1 | · | 6 | 0 | y | y | W→W |
| 195 | `f7b1cc5b3fab95aa1be4` | yes | no/0 | no/0 | no/0 | yes/1 | no/0 | no/0 | 1 | · | 3 | 0 | n | n | X→W |
| 196 | `f8e6087cd52c925fedad` | yes | yes/1 | no/0 | yes/1 | no/0 | —/0 | yes/1 | 1 | · | 3 | 0 | y | y | L→L |
| 197 | `fba624d8c01833419760` | yes | no/0 | yes/1 | no/0 | yes/1 | yes/1 | —/0 | 1 | · | 6 | 0 | y | y | W→W |
| 198 | `fc9339f21f44a3841a7c` | yes | —/0 | yes/1 | —/0 | no/0 | no/0 | no/0 | 1 | · | 3 | 0 | y | y | W→X |
| 199 | `fdeec181aeee06303113` | no | yes/0 | no/1 | yes/0 | yes/0 | no/1 | no/1 | 1 | · | 3 | 0 | y | y | W→X |
| 200 | `ff3811735ededd8ec3a7` | no | no/1 | yes/0 | no/1 | no/1 | no/1 | yes/0 | 1 | · | 2 | 0 | n | n | L→R |

Column sums: A_r2_v1 ✓ 120, TF_v1 ✓ 130, A_r2_v2 ✓ 120, TF_v2 ✓ 129, TF_wrong_v2 ✓ 104, TF_noq_v2 ✓ 90; parse_ok 198; repaired 35; Σ n_facts 683; rows with unknown terms 18; R4 hit v1 154, v2 159.

---

## 10. eval458 pre-check (read-only; no prompt built, no model call)

Two files describe eval458. The freeze names `data/sdfr_splits/strategyqa_eval.jsonl`; the DEV200 loader's schema (`urbench_qid`, `question_ur`, `question_en`, boolean `answer`, `evidence_paragraph_ids`) is the one of `data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl`, whose `is_eval` flag marks the same 458 qids.

| check | `data/sdfr_splits/strategyqa_eval.jsonl` | `data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl` rows with `is_eval` true |
|---|---|---|
| n rows / unique qids | 458 / 458 | 458 / 458 |
| keys | `answer`, `decomposition`, `description`, `facts`, `qid`, `question`, `term` | `answer`, `description`, `evidence_paragraph_ids`, `has_paragraph_evidence`, `is_eval`, `match_method`, `official_decomposition`, `official_description`, `official_evidence`, `official_facts`, `official_qid`, `official_source`, `official_term`, `question_en`, `question_ur`, `term`, `urbench_decomposition`, `urbench_facts`, `urbench_qid`, `urdu_qid_original` |
| gold yes / no | 221 / 237 (Urdu `ہاں`/`نہیں`; other values 0) | 221 / 237 (boolean; other 0) |
| majority floor | 51.75% (no) | 51.75% (no) |
| overlap with DEV200 qids | 0 | 0 |
| same qid set in both files | yes | |
| rows with a non-empty Urdu question | 458 (key `question`; no `question_ur` key) | 458 (key `question_ur`) |
| rows with a non-empty `question_en` | 0 (no such key) | 458 |
| rows with ≥ 1 evidence paragraph id | 0 (no such key) | 457 (Σ ids 2041; rows with `has_paragraph_evidence` true 457) |
| rows with ≥ 1 gold `urbench_facts` (leak needles) | 458 (key `facts`, Urdu) | 458 |
| Urdu question identical across the two files (by qid) | 458/458 | |

Overlap with the DEV200 qids is 0 in both files (expected 0). Overlap of eval458 with the D6-71 subset: 0.

**Can the teacher/student loaders read it as-is? No; a path change is not enough.** Both v2 scripts call `load_rows()` from `eval/error_analysis_tests/efbpt/d4_extract_facts.py`, which reads `DEV_PATH` imported from `eval/error_analysis_tests/efbpt/efbpt_eval_dev200.py` (`DEV_PATH = "data/strategyqa_official/dev200_seed4242.jsonl"`, line 40) and dies unless the file has exactly 200 rows (`die("DEV200 has %d rows, expected 200")`, `d4_extract_facts.py` line 92; `efbpt_eval_dev200.py` also has `EXPECTED_ROWS = 200`, line 51, used by its own `load_dev`). The leak-guard needles come from the same `DEV_PATH` (`d8_reachable_ceiling.py` `load_leak_needles`, line 330), so an eval458 run must point it at the eval rows or the guard would check the wrong question_en / facts / evidence ids. On top of the path, the following constants and checks in the v2 scripts are DEV200-specific and would abort on 458 rows: `d10v2_teacher.py` — `if len(set(qids)) != 200: die("DEV200 qids are not unique")` (line 418), the derangement `n` 200 and the `V1_DERANGEMENT_PATH` / `V1_DERANGEMENT_MD5` equality check (lines 96–97; a 458-row derangement cannot equal the sealed 200-row file); `d10v2_student.py` — the same 200 check (line 321), `D6_E1_PATH` / `D6_N = 71` with `set(d6_qids) <= set(qids)` (R4 (a) is defined on the 71 D6 qids, all in DEV200, none in eval458), `V1_PINNED` t_direct / t_direct_f files (200 rows, DEV200 qids; the student dies if `t_facts.jsonl` qids differ from the loaded rows, line 363), and the `V2_PROBE_N_FULL = 20` probe over A_r2 prompts (fine at any n). The rows themselves are readable by the loader's field names only from the mapped file (`urbench_qid`, `question_ur`, `question_en`, boolean `answer`, `evidence_paragraph_ids`); `strategyqa_eval.jsonl` has `qid`, `question`, Urdu `answer` and no evidence ids, so `required_titles()` would return [] on every row and R4 would be undefined. Under A3.5 and the freeze §7 NO-GO branch no eval458 run of this route is scheduled; this pre-check is recorded for whichever route reaches eval458.

---

## 11. Open items carried forward

- D10 v2 V5 hand check of the 30 TF facts (`outputs/efbpt/d10v2/student/tf_facts_sample30.jsonl`, truth and relevance null); the v1 sample (30 facts) is also still PENDING.
- D8 V5 hand check of 30 Xp facts (still open from D8).
- A5.3 finding, confirmed on FULL: the teacher used `unknown_terms` on 18/200 rows, while its restatement names a different entity than the gold question on many more (restatement-level reading 34.3%; e.g. `00e13b8d` SnapCap → Snapdeal). Abstention engaged on 6 of the 32 v2 losses and on 2 of the 41 v2 wins.
- §7: fact injection is closed as a family (retrieval D2–D5, self D8, teacher D10 v1/v2). Remaining freeze path: training-based distillation of teacher Urdu answers.

