# D11-O READOUT — oracle-reading ceiling diagnostic (jobs 112301–112306)

Written 2026-11-07 from the SEALED files only (outputs/efbpt/d11o/teacher_v1,
teacher_v2, student) plus the three sealed D10/D10v2 reference files the
student job pinned. Nothing was re-run. Every number in §4–§6 was recomputed
from the arm jsonl files with an independent script (exact two-sided binomial
McNemar, scipy binomtest, same as d5_balanced_extract.mcnemar) and agrees with
d11o_summary.json to the last digit. Contract: docs/EFBPT_D11O_ORACLE_READING_FREEZE.md
(MD5 ebd638eb…) as amended by docs/EFBPT_D11O_AMENDMENT_1.md (137cfb38…).
D11-O is a diagnostic, not a method: its numbers decide the design of D11 and
are never claimed as a method result.

**Headline.** Oracle reading (teacher gets the gold English question, student
still reads only Urdu) lifts the v1 teacher arm from +5.0 pp (TF_v1, Urdu
input) to **+14.0 pp over the floor (TO1 74.0% vs A_r2 60.0%, 46W/18L,
p = 0.0006)**. The reading effect itself, same prompt English vs Urdu input, is
**+9.0 pp (TO1 vs TF_v1, 34W/16L, p = 0.015)**, i.e. 0.82 of the GF−TF gap.
TO1 is statistically indistinguishable from the gold-fact ceiling (TO1 vs GF
−2.0 pp, 20W/24L, p = 0.65). **TO2 is VOID as a reading arm**: the v2 teacher
parsed 8/200 rows; the other 192 student prompts are byte-identical to the
A_r2 floor prompts and produced byte-identical generations (§7). Freeze §6
branch: max(O2) = +14.0 ≥ +9.0 → **reading is the main wall → D11 = reading-robust
injection**.

---

## 1. Jobs (sacct, 2026-11-04, partition q_intel_share_L20, 1× L20, 8 CPU, 32G)

| JobID | JobName | State | Exit | Elapsed | Start | End | Node |
|---|---|---|---|---|---|---|---|
| 112301 | d11o_teacher_v1_TEST | COMPLETED | 0:0 | 00:06:17 | 14:27:21 | 14:33:38 | L20003 |
| 112302 | d11o_teacher_v2_TEST | COMPLETED | 0:0 | 00:01:55 | 14:33:38 | 14:35:33 | L20003 |
| 112303 | d11o_student_TEST | COMPLETED | 0:0 | 00:15:14 | 14:35:33 | 14:50:47 | L20003 |
| 112304 | d11o_teacher_v1_FULL | COMPLETED | 0:0 | 00:36:55 | 14:50:47 | 15:27:42 | L20003 |
| 112305 | d11o_teacher_v2_FULL | COMPLETED | 0:0 | 00:50:22 | 15:27:42 | 16:18:04 | L20002 |
| 112306 | d11o_student_FULL | COMPLETED | 0:0 | 05:23:15 | 16:18:04 | 21:41:19 | L20003 |

TEST-5 → FULL chained afterok as freeze §7 requires. Student FULL (3 arms ×
200 + 20-prompt V2' probe, batch size 1) took 5 h 23 min against the 08:00:00
limit. TEST outputs live under outputs/efbpt/d11o/_test/ and are not read here.

## 2. Seals — all MD5s recomputed 2026-11-07, all match

### 2.1 Sealed D11-O files

| File | bytes | lines | MD5 recomputed | = seal |
|---|---:|---:|---|---|
| teacher_v1/T_START.json | 8,902 | 318 | af8f98c63dbce8b4a7ba68eb1b07f7b8 | ✓ (T_SEAL v1, S_START, S_SEAL) |
| teacher_v1/t_facts.jsonl | 711,074 | 200 | ffdea77d13cb00e0d7b9fe9689663e78 | ✓ |
| teacher_v2/T_START.json | 9,020 | 321 | 922f18f0fb8651fb97cb9ba9315b772b | ✓ (T_SEAL v2, S_START, S_SEAL) |
| teacher_v2/t_facts.jsonl | 627,514 | 200 | 8ebe020ce603fd464dd28a4574a4972f | ✓ |
| student/S_START.json | 11,449 | 439 | c0db4f9ecb60560ccd434cfbe00af901 | ✓ (S_SEAL) |
| student/d11o_armGF.jsonl | 347,957 | 200 | 448fec4127bb8060c10422d9aa42699f | ✓ |
| student/d11o_armTO1.jsonl | 486,845 | 200 | 6548b408a68a8e94bb65b9ca6d023fdd | ✓ |
| student/d11o_armTO2.jsonl | 363,988 | 200 | 85d4b81e24ffbec239ed13417c10fc5c | ✓ |
| student/d11o_summary.json | 174,283 | 7,432 | e0efa04bcc0737f3eb4c8304193e836e | ✓ |

T_SEAL v1: job 112304, end 15:27:37, script_md5 338df12c…, parse_ok 198,
repaired 0, removed 0. T_SEAL v2: job 112305, end 16:18:02, same script_md5,
parse_ok 8, repaired 0, removed 0. S_SEAL: job 112306, end 21:41:16,
script_md5 d66c93e5…, A_r2_reused_sealed true. No d11o_armA_r2.jsonl exists
(V2' passed, so the floor was not re-generated — as the freeze specifies).

### 2.2 Pinned references and contracts (recomputed)

| Item | MD5 recomputed | pinned in |
|---|---|---|
| outputs/efbpt/d10/student/d10_armA_r2.jsonl (floor) | 7ccec942c8f0eee08156d258baaf806d | freeze §2, S_START, S_SEAL ✓ |
| outputs/efbpt/d10/student/d10_armTF.jsonl (TF_v1) | c54b7553baa5ffa77204bed1dab37250 | ✓ |
| outputs/efbpt/d10v2/student/d10_armTF.jsonl (TF_v2) | 1a3cdba1f26ccca4b010a00676197f0a | ✓ |
| docs/EFBPT_D11O_ORACLE_READING_FREEZE.md | ebd638eb4b4e48cde00202c8488df124 | all START/SEAL ✓ |
| docs/EFBPT_D11O_AMENDMENT_1.md | 137cfb384e5edde0a41590fef992dcac | S_START/S_SEAL ✓ |
| docs/EFBPT_D10_TEACHER_FREEZE.md | b3a822d8199034c7fe58d76cf5cc428f | all START/SEAL ✓ |
| eval/error_analysis_tests/efbpt/d11o_teacher.py | 338df12c1c8b2f3c362b1f2076244c3b | T_SEAL v1, v2 ✓ |
| eval/error_analysis_tests/efbpt/d11o_student.py | d66c93e563deb37a6cd0b5b59beb5aab | S_SEAL ✓ |

S_START records git head 30485547… (= commit "D11-O Amendment 1"), dirty
false; student model /mnt/home/user41/downloaded_models/Qwen/Qwen3-14B, nf4
double-quant bf16 compute, greedy, thinking off, batch size 1, answer_max_new
1024; Urdu instruction MD5 f3b58d766fe3ec2573ff4f24761cf0c9; torch 2.9.0,
transformers 4.57.6, bitsandbytes 0.49.1. Teacher: Meta-Llama-3.1-70B-Instruct-AWQ-INT4
under vLLM 0.13.0, max_seq_len 4096, seed 0 (logs 112304/112305).

## 3. V-block (from S_START / S_SEAL / the three FULL logs)

| Check | Result |
|---|---|
| V1 unparsed % | A_r2 4.0, GF 1.0, TO1 1.0, TO2 4.0, TF_v1 0.5, TF_v2 0.5 (reported; floor and TF are sealed files) |
| V2' determinism probe | **PASS** 20/20 A_r2 prompts regenerated at batch size 1 are byte-identical to the sealed D10 A_r2 generations → sealed A_r2 reused as the floor. (§7 adds an incidental 192/192 confirmation.) |
| V3 leak guards | none fatal in either job. Teacher prompts: question_en is the oracle input and was removed as a needle (200 strings); gold facts (532) and evidence ids (862) checked, 0 hits in TO1 and TO2 prompts. Student prompts: question_en and evidence ids 0 hits in all arms; gold facts PERMITTED in GF (532 substrings in 200/200 prompts, recorded) |
| V4 teacher parse | v1 198/200 = 99.0% ok (min 95%); **v2 8/200 = 4.0% FORMAT FLAG** |
| V5 verdict filter removed | v1 0, v2 0 |
| V6 verbatim gold-fact substrings in TO prompts (A1.1, reported) | TO1 1 (qid 703db4629ff46d82a205, "Joe Biden is a member of the Democratic Party"); TO2 0 |
| V7 teacher seals | both T_SEALs verified (4 files, md5 + bytes + lines) → PASS |
| V8 prompt content | 406 facts prompts equal their source facts (GF 200 gold, TO1 198, TO2 8 post-T2) → PASS; arm-A layout rows carry no facts header |
| V9 Latin fraction of injected facts | GF 0.9999 (532 facts), TO1 1.0000 (1,188 facts), TO2 1.0000 (28 facts) |
| Layouts | A_r2 A×200; GF B×200; TO1 B×198 + A×2 (two empty teacher rows); **TO2 A×192 + B×8** |
| GF prompt row-0 user MD5 | 6e7327ddd112948f39ef9ef77c993960 (D1 arm B byte for byte, per freeze §2) |

## 4. Accuracy table (n = 200, same qids, recomputed)

| arm | facts source | correct | acc % | unparsed % | trunc % | pred-yes % | acc₂ % | mean gen tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_r2 | none (sealed D10 floor) | 120 | 60.0 | 4.0 | 5.5 | 41.0 | 60.0 | 481 |
| GF | gold urbench_facts, layout B | 152 | **76.0** | 1.0 | 2.5 | 37.0 | 76.0 | 371 |
| TO1 | v1 teacher, English input | 148 | **74.0** | 1.0 | 7.0 | 32.0 | 74.0 | 492 |
| TO2 | v2 teacher, English input (8 parsed) | 122 | 61.0 | 4.0 | 5.5 | 40.0 | 61.0 | 474 |
| TF_v1 | v1 teacher, Urdu input (sealed D10) | 130 | 65.0 | 0.5 | 2.5 | 36.0 | 65.0 | 475 |
| TF_v2 | v2 teacher, Urdu input (sealed D10v2) | 129 | 64.5 | 0.5 | 2.0 | 35.0 | 64.5 | 409 |

Gold split: 87 yes / 113 no (pred-yes rates in §6 O6). acc₂ = secondary
(Devanagari) extractor; identical to primary in every arm.

## 5. Readings O1–O6 (paired exact McNemar, same 200 qids; descriptive, no GO gate)

### O1 — knowledge ceiling at batch size 1
GF − A_r2 = **+16.0 pp** (76.0 vs 60.0), wins 49 / losses 17 (66 discordant),
p = 0.000101. D1 at batch size 8 gave +20.5 (78.0 vs 57.5).

### O2 — perfect-reading teacher gain
| pair | Δ pp | W / L | discordant | p |
|---|---:|---|---:|---|
| **TO1 − A_r2** | **+14.0** (74.0 vs 60.0) | 46 / 18 | 64 | **0.000617** |
| TO2 − A_r2 (VOID, §7) | +1.0 (61.0 vs 60.0) | 2 / 0 | 2 | 0.5 |

Reference, Urdu input (sealed): TF_v1 − A_r2 = +5.0, 38/28, p = 0.268;
TF_v2 − A_r2 = +4.5, 41/32, p = 0.349.

### O3 — the reading effect itself (same prompt, English vs Urdu input)
| pair | Δ pp | W / L | discordant | p |
|---|---:|---|---:|---|
| **TO1 − TF_v1** | **+9.0** (74.0 vs 65.0) | 34 / 16 | 50 | **0.0153** |
| TO2 − TF_v2 (VOID) | −3.5 (61.0 vs 64.5) | 32 / 39 | 71 | 0.477 |

### O4 — reading share = (TO − TF) / (GF − TF)
| version | TO − TF | GF − TF | share |
|---|---:|---:|---:|
| v1 | +9.0 | +11.0 | **0.818** |
| v2 (VOID) | −3.5 | +11.5 | −0.304 |

### Paired TO1 vs GF and the other cross-arm pairs (not in the job log)
| pair | Δ pp | W / L | discordant | p |
|---|---:|---|---:|---|
| **TO1 − GF** | **−2.0** (74.0 vs 76.0) | 20 / 24 | 44 | **0.652** |
| TO2 − GF | −15.0 | 17 / 47 | 64 | 0.000227 |
| TO1 − TO2 | +13.0 | 44 / 18 | 62 | 0.00130 |
| TF_v1 − TF_v2 | +0.5 | 27 / 26 | 53 | 1.0 |

With perfect reading the v1 teacher's six facts are within noise of the gold
facts on this n (44 discordant rows, 20/24). The remaining knowledge/relevance
gap to GF is not resolvable at n = 200.

### O5 — teacher side
| | TO1 (v1) | TO2 (v2) |
|---|---|---|
| parse_ok / repaired / V5 removed | 198 / 0 / 0 | **8** / 0 / 0 |
| finish_reason | stop 198, length 2 | stop 198, length 2 |
| facts per row (post-T2) | min 0, median 6, max 6; 0×2, 6×198 | min 0, median 0, max 6; 0×192, 3×6, 4×1, 6×1 |
| gen tokens min / median / max | 109 / 156.5 / 512 | 117 / 206.5 / 768 |
| R4 (a) D6-71 titles in parsed entities | **90/200 = 45.0%** | 2/200 = 1.0% |
| R4 (b) all 200 rows (636 titles) | **268/636 = 42.1%** | 11/636 = 1.7% |
| R4 restatement-level (teacher question_en) | n/a (v1 has no restatement) | (a) 4/200 = 2.0%; (b) 10/636 = 1.6% |
| V6 gold-fact restatements in prompts | 1 | 0 |

R4 references: 14B D6 G4 21.0%; Urdu-input 70B teacher D10 v1 34.0%, D10v2
36.0% (on the D6-71). English input raises v1 entity reading 34 → 45 on the
same 71 qids. §8.3 shows why this cannot go much higher than ~41–45% under the
norm-substring rule.

### O6 — splits of each arm vs A_r2 by gold label (the "no"-bias check)
| arm | gold | n | acc arm | acc A_r2 | Δ pp | W / L | p | pred-yes arm | pred-yes A_r2 |
|---|---|---:|---:|---:|---:|---|---|---:|---:|
| GF | yes | 87 | 65.5 | 54.0 | +11.5 | 21/11 | 0.110 | 65.5 | 54.0 |
| GF | no | 113 | 84.1 | 64.6 | +19.5 | 28/6 | 0.000195 | 15.0 | 31.0 |
| TO1 | yes | 87 | 57.5 | 54.0 | +3.4 | 15/12 | 0.701 | 57.5 | 54.0 |
| TO1 | no | 113 | 86.7 | 64.6 | +22.1 | 31/6 | 0.0000413 | 12.4 | 31.0 |
| TO2 | yes | 87 | 54.0 | 54.0 | 0.0 | 0/0 | 1.0 | 54.0 | 54.0 |
| TO2 | no | 113 | 66.4 | 64.6 | +1.8 | 2/0 | 0.5 | 29.2 | 31.0 |

TO1's gain is concentrated on gold-no rows (+22.1 pp, pred-yes 31.0 → 12.4);
on gold-yes it is +3.4 pp and not significant. GF shows the same shape but
less extreme (+19.5 / +11.5). The facts mostly help the student reject false
premises; they help less at confirming true ones. (In TO1, the student says
"yes" on 57.5% of gold-yes rows, same as A_r2's 54.0 ± noise, while cutting
false "yes" on gold-no from 31.0% to 12.4%.)

## 6. Pre-declared design decision (freeze §6) — the decision line

**max(O2) = +14.0 pp ≥ +9.0 pp → reading is the main wall → D11 = reading-robust
injection (multi-reading agreement gate; title-grounded entity reading).**
Reading wall +9.0, knowledge wall +6.0, as frozen. eval458 stays reserved for
the final D11 method, run once. max(O2) is TO1's; TO2 contributes nothing to
the decision (it is void, §7) and the branch would be the same at any TO2 value.

## 7. TO2 is VOID as a reading arm (teacher parse 8/200)

### 7.1 What happened
The v2 (T1-v2) teacher, given the gold English question, wrote the JSON
`question_en` value **without quotation marks** in **192/192** failed rows.
In 148 of those it then also wrote `sub_questions` and/or `facts` list items
without quotes. Because parse fails → `facts = []` → the student prompt falls
back to layout A (no facts header), **192 of 200 TO2 student prompts are
byte-identical to the A_r2 floor prompts**, and all 192 generations are
byte-identical to the sealed A_r2 generations (0 differ; 0 preds differ). The
TO2 arm therefore re-measures the floor on 192 rows and carries 8 facts rows.
Its O2 (+1.0, 2W/0L), O3 (−3.5) and O4 (−0.30) are not readings of anything;
they are excluded from every conclusion. On the 8 parsed rows TO2 is 8/8
correct (A_r2 6/8, TF_v2 7/8, GF 8/8) — n = 8, not reportable.

### 7.2 Failure classification over all 192 failures (simple rules on `raw`)
Rules: *truncated* = finish_reason "length"; *prose* = text before the first
`{` or no `{`; *unquoted string* = `"question_en":` followed by a non-quote
character, or a list item line starting with a letter; *other* = none of these.

| primary shape | n | note |
|---|---:|---|
| **unquoted string** (JSON-shaped, value(s) without quotes) | **190** | all 190 start with `{`, finish "stop" |
| truncated (finish "length", 768 tokens) | 2 | both ALSO unquoted; one is a degenerate repetition loop (Appendix A, row 3157a0f4) |
| prose instead of JSON | 0 | |
| other | 0 | |

Flag counts (multi-label): unquoted `question_en` 192/192; unquoted list
items 148/192; truncated 2/192. Combos: `question_en` only 44; `question_en` +
list items 146; truncated + both 2. Repair probe (audit only, nothing enters a
prompt): quoting the `question_en` value alone makes 43/192 rows parse; 149
still fail on the unquoted list items.

Finish reasons: failed rows stop 190 / length 2; parsed rows stop 8 / 8.
n_gen_tokens, failed rows: min 117, median 208, mean 210.5, max 768;
histogram 100–199: 85, 200–299: 103, 300–399: 2, 700+: 2. Parsed rows: 136,
150, 151, 152, 163, 165, 188, 240. So the failures are not a length problem:
they are full, well-formed-looking replies with the quotes missing.

### 7.3 Why (mechanism, from the sealed texts)
- In **172/192** failed rows the unquoted `question_en` value is the input
  question **verbatim** (2 more equal after norm(); 18 differ by small edits,
  e.g. adding "the"). In all 8 parsed rows it is also verbatim — but quoted.
  The teacher is copying the English input, which appears unquoted in the
  prompt ("Question: {question_en}"), straight into the value slot.
- With Urdu input (D10v2, same T1-v2 prompt, same model, same decoding) the
  teacher had to *compose* an English sentence and quoted it: 198/200 parsed,
  only 1/200 raw texts has an unquoted `question_en` (qid e530682239bb5e59ebf0,
  one of D10v2's two parse failures), and 0/200 have unquoted list items.
- The two frozen prompt edits (freeze §3) changed only the language words and
  the question line; the output-format instruction (`"question_en": the
  question in English, one sentence, your best reading;`) shows the key in
  quotes but not the value. With a copyable English input this was enough for
  Llama-3.1-70B-AWQ greedy to drop the quotes, and the omission then propagates
  to the following lists in 148 rows. The D10v2 A4/A5 repairs (brace/bracket
  appends) do not address missing quotes and fired on 0 rows.
- The v1 prompt (two keys, no restatement) is unaffected: 198/200 parsed.

### 7.4 Consequence for D11
The oracle-reading number for the v2 (decomposition) prompt does not exist.
The decision in §6 rests on TO1 alone and does not need it. If a v2 oracle
number is ever wanted, the fix is a format change (e.g. a one-line JSON example
with a quoted value, or vLLM structured outputs as the QEA Llama backend already
uses) under a new freeze, not a parse repair of the sealed TO2 output.

## 8. Reading metrics on existing outputs (audit only; nothing enters a prompt)

### 8.1 (3a) The frozen L0 pair set — path and schema
There is no single materialised pair file. The 289 pair instances are built
deterministically by `load_pairs()` in eval/error_analysis_tests/efbpt/l0_part_bc.py:61-80
from the `GOLD` list in l0_part_a_census.py:27-29, classified by
`classify(span, title)` (TRANSLIT if translit similarity ≥ 0.70, COREF from the
explicit two-pair list of L0 Amendment 1, else SEMANTIC):

| source | rows | id field | entities field | pairs |
|---|---:|---|---|---:|
| data/strategyqa_official/efbpt/plan_a_gold_100.jsonl (MD5 c71356b4a98b4fb718a29cc7f5b98216) | 100 | qid | entities | 225 |
| data/strategyqa_official/efbpt/blind30_gold.jsonl (MD5 322c95a16bec2ad0f635e3c21696ae4d) | 30 | urbench_qid | question_entities | 64 |

Each entity entry is `{"canonical_title": <English Wikipedia title>, "urdu_span": <span in the Urdu question>}`
(stored as a Python-literal string in the jsonl; the loaders literal-eval it).
A pair = (qid, urdu_span) deduplicated, title non-empty. Rebuilt today: **289
pairs = 126 TRANSLIT + 161 SEMANTIC + 2 COREF, LINKABLE 287**, 130 distinct
qids — identical to outputs/efbpt/l0/l0_summary.json and experiments.md "EFBPT L0".
Question text: plan_a_gold_100 rows carry only question_ur; the gold
question_en for those qids is in data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl
(2,290 rows, MD5 bb982798279f5d519ddef01f5cb7ba8c, all 100 qids present);
blind30 question_en is in data/strategyqa_official/efbpt/blind30_rows.jsonl
(agrees with the mapped file on all 30).

**Important scope fact: none of the 130 L0 qids is a DEV200 qid** (overlap 0;
L0 was built from the Plan-A gold-100 and blind-30 cohorts). The teacher
restatements (D10v2 and TO2) exist only for DEV200. So M3 on the restatements
cannot be computed over the L0 pair set at all; §8.2 gives the L0 gold ceiling
and the DEV200 analogue side by side, with denominators.

### 8.2 (3b) M3 = share of LINKABLE pairs whose norm(title) is a substring of norm(reading of the question)
norm() = d3_oracle_retrieval.norm (lines 59-60): underscores → spaces, strip,
lower, collapse whitespace. Substring, no parenthetical stripping unless stated.

**(i) L0 pair set, gold question_en — the ceiling for a verbatim-title reader**

| set | hit / n | M3 |
|---|---:|---:|
| LINKABLE | 218 / 287 | **75.96%** |
| TRANSLIT | 108 / 126 | **85.71%** |
| SEMANTIC | 110 / 161 | **68.32%** |
| (paren-stripped titles) LINKABLE / TRANSLIT / SEMANTIC | 228/287, 113/126, 115/161 | 79.4 / 89.7 / 71.4% |
| by source: plan_a_gold_100 LINKABLE; blind30 LINKABLE | 168/224; 50/63 | 75.0%; 79.4% |

Even the gold English question contains the canonical title verbatim for only
¾ of linkable pairs; misses are disambiguators ("Vulcan (mythology)",
"Tiger Stadium (LSU)"), lemma/surface mismatches ("baptized" vs Baptism,
"movie" vs Film, "unemployed" vs Unemployment, "CEO" vs Chief executive officer)
and descriptive spans ("Lamborghini's fastest model" vs Lamborghini Veneno).
TRANSLIT pairs are 17 pp easier than SEMANTIC under this rule, as expected.

**(ii) Teacher restatements over the L0 pairs: not computable.** Coverage
of the 287 LINKABLE pairs by a D10v2 restatement: 0; by a TO2 parsed
restatement: 0 (no DEV200 qids in L0).

**(iii) DEV200 analogue (same rule, required gold titles from
evidence_paragraph_ids instead of L0 pairs; this is R4 computed on the reading
instead of on the entities):**

| reading of the question | titles covered | hit | M3-analogue | D6-71 subset |
|---|---:|---:|---:|---|
| gold question_en (ceiling) | 636 | 259 | **40.72%** | 81/200 = 40.5% |
| D10v2 teacher restatement from Urdu (198 parsed rows) | 628 | 218 | **34.71%** | 68/195 = 34.9% |
| gold question_en on the same 198 rows | 628 | 255 | 40.61% | 78/195 = 40.0% |
| TO2 restatement, 8 parsed rows | 23 | 10 | 43.5% (n = 8 rows) | 4/10 |
| TO2 restatement regex-extracted from raw, all 200 (audit; = input verbatim in 172) | 636 | 259 | 40.72% | 82/200 = 41.0% |

Reading from Urdu loses 6 pp of title presence against the gold English
question (255 → 218 on the same rows: 47 titles present in the gold question
are absent from the restatement, 10 appear only in the restatement). Only
19/198 D10v2 restatements equal the gold question after norm(). Typical losses
are entity misreadings — "Acela Express" → "MTR's Airport Express", "Abyssal
plain" → "Abyssinian meadows", "Glorious Revolution" → "Jallianwala Bagh
revolution", "Grapefruit" → "jackfruit", "Kate Gosselin" → "Gosling family",
"Hair" → "bal" (untranslated Urdu) — i.e. exactly the reading wall that O3
measures at +9.0 pp.

### 8.3 (3c) Why R4 caps near 45%: required titles present in the gold question itself
DEV200 has 636 required titles over 200 qids (per row: 1×16, 2×49, 3×66,
4×36, 5×21, 6×10, 7×1, 8×1). Under norm-substring, the **gold English question
contains 259/636 = 40.7%** of them (D6-71: 81/200 = 40.5%; 286/636 = 45.0%
with parentheticals stripped). The rest are evidence-side titles the question
never names (e.g. "LendingTree", "Payday loan" for the SnapCap question; "The
WB", "Supernatural (season 1)" for the Supernatural question) or titles that
appear only as inflected/common-noun surface forms. TO1's R4 = 268/636 =
42.1%: 229 of its 268 hits are titles that occur verbatim in the question, 39
are titles the teacher supplied without a verbatim cue. So an English-input
teacher already reads essentially everything a title-substring rule can
credit; R4 above ~45% would require predicting evidence titles, not reading.
R4 is a reading metric only up to that ceiling, and the D11 "title-grounded
entity reading" target should be stated against 40–45%, not 100%.

## 9. Resources for D11-R

**(a) Gemma-2-27B-it NF4 load (QEA code) and measured speed.**
eval/error_analysis_tests/efbpt/efbpt_qea_triage_v1.py: model spec at lines
82-84 (`MODELS["gemma"]`: path /mnt/home/user41/downloaded_models/AI-ModelScope/gemma-2-27b-it,
max_model_len 8192); backend `class HfBnbBackend` lines 995-1036; the load
itself lines 1008-1016:
```python
self.tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True, trust_remote_code=False)
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
self.model = AutoModelForCausalLM.from_pretrained(spec["path"], local_files_only=True, trust_remote_code=False,
                                                  quantization_config=quant, device_map={"": 0},
                                                  attn_implementation="eager", use_safetensors=True,
                                                  dtype=torch.bfloat16)
self.model.eval()
```
(explicit `dtype=torch.bfloat16` per QEA Amendment 2; the dtype census at
1017-1019 asserts bf16 on embed/lm_head/norm/compute; stop ids add
`<end_of_turn>`.) Measured, sequential greedy, one L20 (sealed records):

| run | calls | gen s | **s/call** | mean out tok | mean prompt tok | tok/s | load s | peak CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qea_review_calibration_v1/gemma_run | 102 | 678.4 | **6.65** (median 6.55, max 10.2) | 73.4 | 639 | 11.0 | 722.6 | 23.1 GB |
| qea_identification_calibration_v1/gemma_run | 81 | 278.9 | **3.44** (max 5.3) | 37.3 | 527 | 10.8 | 599.7 | 23.1 GB |
| qea_triage_v1_smoke_gemma_v2 (job 96350) | 2 | — | — | 70 | 406–448 | — | 533.1 | 23.1 GB |

So ≈ 11 output tokens/s at ~500–650 prompt tokens, plus 9–12 min load. A
200-row D11-R pass at ~150 output tokens would be ≈ 14 s/row ≈ 47 min + load;
at 512 tokens ≈ 47 s/row ≈ 2.6 h. (For comparison the Qwen3-14B nf4 student
in this job: 620 generations at ≤1024 tokens in 5 h 23 min ≈ 31 s/row.)

**(b) NLLB-200 on disk?** Not under /mnt/home/user41 (no `models--facebook--nllb*`
in ~/.cache/huggingface/hub; nothing under downloaded_models). A **complete
copy of facebook/nllb-200-3.3B exists on the shared filesystem at
/mnt/home/user49/junhaoliu/models/nllb-200-3.3B** (another user's directory,
group/world-readable, dated 2026-06-16): config.json, generation_config.json,
pytorch_model-0000{1,2,3}-of-00003.bin (6.93 + 8.55 + 2.10 GB = 17.6 GB),
pytorch_model.bin.index.json, sentencepiece.bpe.model, tokenizer.json,
tokenizer_config.json, special_tokens_map.json, README.md — the same 12-file
set as the HF repo, so it is loadable in place with `local_files_only=True`
(read access only; copy or pin by MD5 before any frozen use, since it is not
ours). No distilled-1.3B copy anywhere on this filesystem (also present in
that directory, for reference: mbart-large-50-many-to-many-mmt, mt0-xl).
HEAD requests (no download):
- https://huggingface.co/facebook/nllb-200-3.3B/resolve/main/config.json → 307 → **200**, x-repo-commit 1a07f7d195896b2114afcb79b7b57ab512e7b43e; repo files: config, generation_config, 3 × pytorch_model .bin + index, sentencepiece.bpe.model, tokenizer.json, tokenizer_config.json, special_tokens_map.json.
- https://huggingface.co/facebook/nllb-200-distilled-1.3B/resolve/main/config.json → 307 → **200**, x-repo-commit 7be3e24664b38ce1cac29b8aeed6911aa0cf0576.
Both repos are reachable from this node without authentication.

**(c) sacrebleu in urbench_eval?** **No** — `ModuleNotFoundError: No module
named 'sacrebleu'` under /mnt/home/user41/miniconda3/envs/urbench_eval/bin/python
(3.10.19). Present: sentencepiece 0.2.1, sentence-transformers 5.4.1,
transformers 4.57.6, torch 2.9.0, vllm 0.13.0, scipy 1.15.3. chrF would need
`pip install sacrebleu` (pure Python, no CUDA dependency) under a recorded
environment change.

## 10. What this readout does not claim
- No method result: TO1 is an oracle (gold English question) arm.
- The TO2 / v2-prompt oracle number does not exist (§7); nothing is inferred about the decomposition prompt's reading.
- n = 200, one seed, greedy; the TO1 − GF difference (−2.0, p = 0.65) is "not distinguishable", not "equal".
- M3 on teacher restatements over the L0 pairs is impossible (disjoint qids); the DEV200 analogue in §8.2(iii) is a different population and rule base (required evidence titles, not annotated question entities).

---
## Appendix A — full raw TO2 teacher text (v2 prompt, English input), 5 failed + 1 truncated-failed + 2 parsed rows, verbatim from teacher_v2/t_facts.jsonl `raw`

### FAIL 00e13b8d03d35929c049 — finish_reason stop, n_gen_tokens 125
Input question_en: Is SnapCap an example of a retail store?

```text
{
  "question_en": Is SnapCap an example of a retail store?,
  "unknown_terms": ["SnapCap"],
  "sub_questions": [
    "What is SnapCap?",
    "What is a retail store?",
    "Is SnapCap a business that sells goods or services directly to consumers?"
  ],
  "entities": [
    "Retail"
  ],
  "facts": [
    "A retail store is a business that sells goods or services directly to consumers.",
    "Retail is the process of selling consumer goods or services to customers through multiple channels of distribution to earn a profit."
  ]
}
```

### FAIL 01430717f590ad3f4e74 — finish_reason stop, n_gen_tokens 167
Input question_en: Can you buy spinal cord at Home Depot?

```text
{
  "question_en": Can you buy spinal cord at Home Depot?,
  "unknown_terms": [],
  "sub_questions": [
    What products does Home Depot sell?,
    Is spinal cord a product sold by retailers?,
    Is spinal cord a product used in construction or home improvement?
  ],
  "entities": [
    "Home Depot",
    "Spinal cord"
  ],
  "facts": [
    Home Depot is an American home improvement retailer.,
    Home Depot sells products for construction, home improvement, and gardening.,
    Spinal cord is a long, thin, tube-like structure made up of nervous tissue, which extends from the base of the brain down to the lower back.,
    Spinal cord is a part of the human body and is not a product used in construction or home improvement.
  ]
}
```

### FAIL 018f6ea10facddae3b2e — finish_reason stop, n_gen_tokens 133
Input question_en: Was Dorothea Wendling from same place Porsche originated?

```text
{
  "question_en": Was Dorothea Wendling from the same place Porsche originated?,
  "unknown_terms": [],
  "sub_questions": [
    "Where was Dorothea Wendling from?",
    "Where did Porsche originate?",
    "Is the place of origin of Dorothea Wendling the same as the place of origin of Porsche?"
  ],
  "entities": [
    "Dorothea Wendling",
    "Porsche"
  ],
  "facts": [
    "Dorothea Wendling was born in Stuttgart, Germany.",
    "Porsche was founded in Stuttgart, Germany."
  ]
}
```

### FAIL 01c27885cede6741bc53 — finish_reason stop, n_gen_tokens 145
Input question_en: Did Supernatural break 2001 CW debuting shows seasons record?

```text
{
  "question_en": Did Supernatural break the 2001 CW debuting shows seasons record?,
  "unknown_terms": [],
  "sub_questions": [
    "What is the record for the number of seasons of a show that debuted on the CW in 2001?",
    "How many seasons did Supernatural have?"
  ],
  "entities": [
    "Supernatural (TV series)",
    "The CW"
  ],
  "facts": [
    "Supernatural is an American fantasy horror television series that aired from 2005 to 2020.",
    "Supernatural had 15 seasons.",
    "The CW is an American English-language broadcast television network."
  ]
}
```

### FAIL 05e9cdc44f1b066badd7 — finish_reason stop, n_gen_tokens 247
Input question_en: Is it common for women to have moustaches?

```text
{
  "question_en": Is it common for women to have moustaches?,
  "unknown_terms": [],
  "sub_questions": [
    What percentage of women have visible facial hair?,
    What is the typical definition of a moustache?,
    Is facial hair in women usually visible or not?,
    What is the prevalence of hirsutism in women?
  ],
  "entities": [
    "Women",
    "Moustache",
    "Facial hair",
    "Hirsutism"
  ],
  "facts": [
    Women typically have fine vellus hair on their faces, which is usually not visible to the naked eye.,
    A moustache is typically defined as a growth of hair on the upper lip between the nose and the mouth.,
    Facial hair in women can be caused by a variety of factors, including hormonal imbalances and genetics.,
    Hirsutism is a condition characterized by excessive growth of terminal hair in a male pattern in women.,
    Hirsutism affects approximately 5-10% of women of reproductive age.,
    The prevalence of hirsutism varies depending on the population being studied and the criteria used to define it.
  ]
}
```

### FAIL-TRUNC 3157a0f428531ebed7b9 — finish_reason length, n_gen_tokens 768
Input question_en: Are any minor league baseball teams named after felines?

```text
{
  "question_en": Are any minor league baseball teams named after felines?,
  "unknown_terms": [],
  "sub_questions": [
    What are the names of minor league baseball teams?,
    What are felines?,
    Are any of the names of minor league baseball teams also names of felines?
  ],
  "entities": [
    "Minor league baseball",
    "Felines"
  ],
  "facts": [
    Minor league baseball is a hierarchy of professional baseball leagues below Major League Baseball.,
    Felines are a biological family of carnivorous mammals commonly known as cats.,
    The New Orleans Baby Cakes were renamed the New Orleans Baby Cakes to the New Orleans Zephyrs and then to the New Orleans Baby Cakes and are now known as the New Orleans Baby Cakes but were previously known as the New Orleans Zephyrs and before that the Denver Zephyrs but were previously known as the New Orleans Zephyrs and before that the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but were previously known as the New Orleans Zephyrs but were previously known as the Denver Zephyrs but
```

### OK 068379affae04debca2b — finish_reason stop, n_gen_tokens 151
Input question_en: Did the Nepalese Civil War take place near India?

```text
{
  "question_en": "Did the Nepalese Civil War take place near India?",
  "unknown_terms": [],
  "sub_questions": [
    "Where did the Nepalese Civil War take place?",
    "Where is India located?",
    "Is the location of the Nepalese Civil War near the location of India?"
  ],
  "entities": [
    "Nepalese Civil War",
    "India"
  ],
  "facts": [
    "The Nepalese Civil War took place in Nepal.",
    "India is a country located in South Asia.",
    "Nepal is a landlocked country located in South Asia, bordered by India to the east, west, and south."
  ]
}
```

### OK 20693ee66ad7ca890e08 — finish_reason stop, n_gen_tokens 240
Input question_en: Do the Egyptian pyramids look the same from outside as they did when new?

```text
{
  "question_en": "Do the Egyptian pyramids look the same from outside as they did when new?",
  "unknown_terms": [],
  "sub_questions": [
    "What was the original exterior appearance of the Egyptian pyramids?",
    "What changes have occurred to the exterior of the Egyptian pyramids over time?",
    "What is the current exterior appearance of the Egyptian pyramids?"
  ],
  "entities": [
    "Egyptian pyramids",
    "Great Pyramid of Giza"
  ],
  "facts": [
    "The original exterior of the Great Pyramid of Giza was covered in smooth white limestone.",
    "The Great Pyramid of Giza's limestone casing was highly reflective and could be seen from a distance.",
    "Over time, the limestone casing of the Great Pyramid of Giza was stripped and used for other building projects.",
    "The current exterior of the Great Pyramid of Giza is rough and composed of the original core stones.",
    "The Great Pyramid of Giza has undergone restoration and conservation efforts to protect its exterior.",
    "The exterior of the Great Pyramid of Giza is exposed to the elements and subject to erosion."
  ]
}
```
