# EFBPT D15 — READING EFFECT AND TEACHER FACTS ON NATIVE URDU QUESTIONS (FREEZE)

Status: FROZEN 2026-11-22, before any d15 script, job or output exists. Known at this point: the
row counts of the source file (section 11.1). No model has seen any row; the sample (N1) is
drawn only after this file is committed.

Goal: test the reading effect on NATIVE Urdu questions (never translated from English), with
controls for the translator, the instruction language and the subject type; and test the
teacher-facts method on the same set. D15 selects nothing and tunes nothing. No earlier verdict
changes whatever D15 shows; D11, D13 and D14 stand as they are. No arm is run again after it is
scored.

Sections N1 to N10 are the design as given. Section 11 holds the operational details that make
N1 to N10 executable; it changes no arm, prompt, model, decoding or decision rule. "D14 F3",
"D14 F4", "D14 F6", "D14 Amendment 1" and "D10 engine" mean the text and code of
docs/EFBPT_D14_MORE_TASKS_FREEZE.md, docs/EFBPT_D14_AMENDMENT_1.md,
docs/EFBPT_D10_TEACHER_FREEZE.md and eval/error_analysis_tests/efbpt/d14_common.py.

No existing file under docs/, no existing d1* ... d14* script or fair-baseline script and no
sealed file under outputs/ is edited, moved or deleted. New code is in new files with the
prefix d15_.

## N1. Rows and sample

Valid rows: correct_key in {A,B,C,D}; options has exactly the keys A-D, all non-empty;
question non-empty; exact duplicates (question + options) keep the lowest id. Sample: rows
sorted by id inside each subdomain, random.Random(1515) sample of 100 per subdomain (all rows
if fewer than 100). Expected n about 2,600. Write the id list to a committed file with its MD5.
The sample is equal per subject, so overall accuracy is not the dataset-weighted accuracy;
say so in the readout.
Groups, fixed from the subdomain NAMES only, before any run: URDU = name contains "urdu";
REGIONAL = name contains "islam" or "pakistan"; OTHER = the rest. If a name is unclear, put it
in OTHER and log it.

## N2. Prompts (4 options)

 UR = the CSQA fair prompt (cot_csqa_baseline_fair.py) built by code from its own strings with
   exactly these replacements, each asserted to occur exactly once: the Urdu word for "five"
   -> "چار"; "(A، B، C، D، E)" -> "(A، B، C، D)"; "A، B، C، D یا E" -> "A، B، C یا D";
   and the "E: {E}" line removed.
 EN instruction = "You are given a question and four options (A, B, C, D).\nFirst think
   briefly about which answer is correct.\nAt the end use only this form: Answer: X\nwhere X
   is only A, B, C or D."
 EN item block = "Question: {q}\nA: {A}\nB: {B}\nC: {C}\nD: {D}"; trailer
   "\n\nStep-by-step reasoning:\n\nAnswer:".
 UE = EN instruction + the item block with the Urdu text in it + the EN trailer.
 EN = EN instruction + an English item block + the EN trailer.
 Domain, subdomain and level are never put in a prompt.

## N3. Translations

(the gold key is never sent; same system message as D14 F3; a translation is
valid if it has the labels Question:, A:, B:, C:, D: in this order).
 R1  = qwen3.8-max, D14 F3 settings, one retry if invalid; refused twice = refused.
 TL  = Llama-3.1-70B-AWQ translating by itself (D10 engine, greedy, max_tokens 512).
 SL  = Qwen3-14B translating by itself (thinking off, greedy, max_tokens 512).
 An invalid local translation is not repeated (greedy); the row is reading_failed for the
 arms that use it. Report, per translation, the count of outputs that contain "correct",
 "answer is" or a mark on one option (reported, not a check).

## N4. Arms

 Student S = Qwen3-14B, engine and sampling of the CSQA fair script (as D14 CSQA):
  S_ur   UR.
  S_ro   UR + the D14 translation label + the R1 item block (D14 F4 layout).
  S_self as S_ro with the SL item block.
  S_tr   UR + facts block (D14 F4 rules and filters unchanged; facts from the teacher with
         the R1 item block).
 Teacher T = Llama-3.1-70B-AWQ, D14 engine (Amendment 1 arguments), max_tokens 1024:
  T_ur UR; T_ue UE; T_r1 EN with the R1 block; T_self EN with the TL block.
 Commercial M = qwen3.8-max, D14 settings, max_tokens 1024: M_ur UR; M_ue UE; M_r1 EN with R1.
 (M_r1 is self-translation for M, because R1 is the same model.)

## N5. Scoring

The CSQA fair extract_answer with the letter class limited to A-D, on the answer
text (after </think> for Qwen3-14B). Empty = unparsed = wrong. Truncated = reached max_tokens.

## N6. Analysis set

Analysis set A = sampled rows with a valid R1 translation and an answered (not refused)
call in M_ur, M_ue and M_r1. Every comparison is read on A. The count of dropped rows is
reported overall and per group, and the all-sampled-rows line is printed beside each test.
For arms that use TL or SL, a reading_failed row counts as wrong, and the line on rows with a
valid local translation is printed beside it.

## N7. Tests

(exact two-sided McNemar; b = first arm right and second wrong; deciding line by
the D14 F6 rule on set A; Holm at 0.05 inside each family; wording "helps" / "hurts" /
"no clear effect" as in D14 F6).
 H1 reading effect on native Urdu (3 tests): S_ro vs S_ur; T_r1 vs T_ue; M_r1 vs M_ue.
 H2 self-translation, no outside model (2 tests): S_self vs S_ur; T_self vs T_ue.
 H3 method (1 test): S_tr vs S_ur.
 Descriptive, no wording: T_ue vs T_ur and M_ue vs M_ur (instruction language); T_r1 vs T_ur
 and M_r1 vs M_ur (the D14 form); S_tr vs S_ro; S_self vs S_ro; T_self vs T_r1; every H1 and
 H2 pair inside URDU, REGIONAL and OTHER, and per domain, with b, c and raw p; a per-subdomain
 table of the deltas without p.
 Power, write it in: with about 2,600 rows, raw p < 0.05 needs about 1.3 to 2.2 pp and the
 first Holm step of H1 about 1.5 to 2.6 pp (discordant share 10% to 30%); inside a group of
 300 rows about 6 pp.

## N8. Predictions, before any run (set A)

S_ro - S_ur in [-2, +6]; T_r1 - T_ue in [-2, +8];
M_r1 - M_ue in [-4, +3]; S_self - S_ur in [-4, +4]; T_self - T_ue in [-3, +6];
S_tr - S_ur in [-2, +6]; for each H1 pair the delta in URDU is not above the delta in OTHER.

## N9. Reading of the outcome, fixed now

If at least 2 of the 3 H1 tests say "helps": the
reading effect holds on native Urdu text. If none says "helps": the effect measured on the
translated tasks (D13, D14) is not shown on native text and is reported as tied, at least in
part, to translated benchmarks. Otherwise: "mixed", with the three results stated one by
one. H2 is read the same way for the claim "the gain needs no stronger translator".
Whatever the outcome, D11, D13 and D14 stand as they are.

## N10. Checks

Preflight (row counts, sample MD5, the UR prompt replacements, the scorer on the
sealed CSQA fair output still gives 190/300 with the unrestricted class, no gold in any
prompt); TEST-5 = the first 5 sampled rows, every arm, never scored; one seal per arm;
nothing is scored before every arm is sealed; seconds and tokens per row logged.

## 11. Operational details, fixed now

Where a D14 section is cited (D14 9.x), its text applies with the D15 rows, arms and paths.

### 11.1 Source and pinned inputs

UrduMMLU (MBZUAI/UrduMMLU, CC BY 4.0), the one file of its test split, repository revision
`aae060a8564c261393de206c94a4557c850e7f2f`. Downloaded 2026-11-22 from
`https://hf-mirror.com/datasets/MBZUAI/UrduMMLU/resolve/aae060a8564c261393de206c94a4557c850e7f2f/<file>`
(huggingface.co gave no answer within 30 s; the mirror serves the same repository, and the
SHA-256 of urdummlu.json equals the LFS object id that the repository tree lists for this
revision). Data only: no code from the download is run; README.md is read by no script.

| file (not tracked by git) | MD5 | note |
|---|---|---|
| data/urdummlu/urdummlu.json | a25bbf61360af54747d2b1b8dbd96041 | SHA-256 b2187d0cb5bda0c2e0eb685c85c7264348850bc0bf43e2aaa4cd932aa85a49a9 |
| data/urdummlu/stats.json | cb6b24df4c212684043791256d4d4ff0 | the dataset's own counts |
| data/urdummlu/README.md | 370dee61a4ef6d46cbe8eacf5828beda | dataset card |

Row counts of the file (the only thing known before this freeze): 26,431 rows, ids 0 to 26,430
(unique integers), fields id, question, options, correct_key, domain, subdomain, level,
length_tier, source; 5 domains, 26 subdomains; options with the keys A-D on 26,272 rows and
A-E on 159 rows; correct_key E on 42 rows.

Fair script and its sealed output (as D14 9.1):

| file | MD5 |
|---|---|
| eval/error_analysis_tests/cot_csqa_baseline_fair.py | 70491653824525aeaa58a3282d5d177f |
| outputs/sdfr/cot_csqa_baseline_fair_qwen3_14b.jsonl | 40e09c2208cce4217f86fb54c1cc034c (190 / 300) |

Frozen code that D15 imports or reads (eval/error_analysis_tests/efbpt/): d14_common.py
116c2d5fef8ec4ba5b8df844a1410fd9; d14_score.py fb1d2df15619067f85f7bc0b8edb591d; d14_readout.py
1734cee1b977395ef6cb175875321380; d14_balance.py 4098c33f54f0c76f38a7a970c500a408; and through
d14_common (pinned there): d10_teacher.py 60bca5ba3d60e28e93fac1c643bf6802; d11o_teacher.py
338df12c1c8b2f3c362b1f2076244c3b; d1_eval_arms.py 10917f65b7852b406aec709a95d0a45b;
d11_ablate_ro.py b3d42f4c7b6805d4f91f45a2e28dca5b; d11_api_direct.py
4af85505d859b481aaf1ab0caf26905f; d11r_common.py 4205923e6aca8294d9c1e10f860523a7.

Documents cited: docs/EFBPT_D14_MORE_TASKS_FREEZE.md 31bc15881ad8102a2b35a676abf1c268;
docs/EFBPT_D14_AMENDMENT_1.md 6153e4ead003bc4b183df9ec28be470b; docs/EFBPT_D10_TEACHER_FREEZE.md
b3a822d8199034c7fe58d76cf5cc428f.

Models (no download): Qwen3-14B = downloaded_models/Qwen/Qwen3-14B; teacher =
downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4 (the d10_teacher
constant); commercial = qwen3.8-max through the gateway of the key file.

### 11.2 Subdomains and groups (N1)

The rule is applied to the lower-case `subdomain` field. Every name falls under exactly one rule;
no name is unclear, so none is logged as such. Observation only, no change: "civics", "current
and international affairs" and "general knowledge" may hold Pakistan-specific questions, but
their names contain neither word, so they are OTHER.

| subdomain | domain | rows in the file | group |
|---|---|---:|---|
| urdu literature | Humanities | 5859 | URDU |
| urdu language | Humanities | 2522 | URDU |
| urdu grammar | Humanities | 756 | URDU |
| pakistan studies | Social Sciences | 1895 | REGIONAL |
| islamic studies | Humanities | 1497 | REGIONAL |
| general knowledge | Other | 1365 | OTHER |
| chemistry | STEM | 936 | OTHER |
| biology | STEM | 931 | OTHER |
| general science | STEM | 926 | OTHER |
| economics | Social Sciences | 907 | OTHER |
| mathematics | STEM | 876 | OTHER |
| education | Social Sciences | 836 | OTHER |
| computer science | STEM | 823 | OTHER |
| sociology | Social Sciences | 778 | OTHER |
| health and physical education | Social Sciences | 742 | OTHER |
| civics | Social Sciences | 675 | OTHER |
| geography | Social Sciences | 630 | OTHER |
| physics | STEM | 621 | OTHER |
| professional development | Profession | 514 | OTHER |
| professional studies | Profession | 461 | OTHER |
| psychology | Social Sciences | 460 | OTHER |
| current and international affairs | Social Sciences | 437 | OTHER |
| commerce | Social Sciences | 423 | OTHER |
| ethics | Humanities | 228 | OTHER |
| psychometrics | Social Sciences | 185 | OTHER |
| fine arts | Humanities | 148 | OTHER |

URDU 3 subdomains, REGIONAL 2, OTHER 21. Each subdomain name belongs to one domain.

### 11.3 Rows, ids, gold, sample

- Row id `um_` + the id in 5 digits, e.g. `um_00042`.
- Valid (N1): `question` is a string that is non-empty after strip; `options` is a dict whose key
  set is exactly {A, B, C, D} and every value is a string that is non-empty after strip;
  `correct_key` is one of the strings A, B, C, D. Duplicates: among the valid rows, rows with the
  same `(question, options A, options B, options C, options D)`, compared as exact strings,
  are one group; the row with the lowest id is kept. Duplicates are looked for over the whole
  file, not inside a subdomain.
- Sample (N1): for each subdomain, in sorted order of the name, the list L of its kept rows
  sorted by id; a new `random.Random(1515)` for each subdomain; `rng.sample(L, 100)` if L has more
  than 100 rows, else all of L. The sample is the union. Run order = ascending id. TEST-5 = the
  first 5 ids of that order.
- The id list is written by d15_sample.py to eval/error_analysis_tests/efbpt/d15_sample_ids.txt
  (one id per line, ascending, newline after each), committed; its MD5 is pinned in
  d15_common.py and checked by every runner, the preflight and the readout.
- Gold = `correct_key`, read only by the readout (and by the preflight's leak check). No arm file
  holds a gold field. The prompt builders read only `question` and `options`.

### 11.4 Prompts

- UR (N2). The source segments of the `instr` and `q_block` assignments of the prompt loop of the
  pinned fair script are taken from its source; the four replacements are made on that source
  text, each asserted to occur exactly once in it: `پانچ` -> `چار`; `(A، B، C، D، E)` ->
  `(A، B، C، D)`; `A، B، C، D یا E` -> `A، B، C یا D`; the line `E: {E}` is removed by deleting the
  source text `\nE: {E}` (the escaped line break before it). The two changed statements and the
  unchanged `raw` statement are run with `item = {"question": question}` and `A, B, C, D` = the
  four option texts. The fair script's statements that read CSQA choices are not used. Fields are
  used as they are in the file, as the fair script does. UR user message = `raw`.
- The UR trailer is the part of `q_block` after the item lines: `Step-by-step reasoning (اردو
  میں):\n\nAnswer:`, taken from a probe row. A block added by an arm (S_ro, S_self, S_tr) sits
  between the item lines and the trailer: instruction, item lines, added block, trailer, joined
  by blank lines (D14 9.6).
- Item blocks (D14 9.4): five lines, labels `Question`, `A`, `B`, `C`, `D`; each line is label +
  ": " + text; inside an item block every run of white space in a field is one space and the
  field is stripped. Urdu item block = the labels with the Urdu fields (the translation USER
  message and the item block of UE). R1 / TL / SL item block = the valid translation (11.5).
- EN instruction and trailer as N2. UE = EN instruction + "\n\n" + Urdu item block + "\n\n" +
  "Step-by-step reasoning:\n\nAnswer:"; EN = the same with an English item block (R1 or TL).
- Added blocks: S_ro = the D14 translation label line (d14_common.tr_label_line, from the
  d11_ablate_ro constant) + "\n" + the R1 item block; S_self = the same label line + "\n" + the SL
  item block; S_tr = d1_eval_arms.facts_block(facts); a row with no fact left has no block, so its
  S_tr prompt equals its S_ur prompt (D14 9.6).
- Messages: S = one user message, chat template with the generation prompt and
  enable_thinking=True (the fair script); T = one user message through the model's chat template
  with the generation prompt; M = one user message, no system message.

### 11.5 Translations

- R1: D14 9.5 with the five D15 labels: qwen3.8-max, enable_thinking false, temperature 0,
  max_tokens 512, 4 concurrent requests, the D11-R retry policy (429, 500, 502, 503, 504 and
  connection errors, backoff 2 / 4 / 8 s, 3 attempts); SYSTEM = the D14 F3 message
  (d14_common.R1_SYSTEM, imported, unchanged); USER = the Urdu item block. An answered request
  with an invalid output is requested once more with the same messages; a second invalid output
  makes the row reading_failed; a row refused by the provider's input inspection (HTTP 400) on
  two separate requests is reading_failed with the failure "refused" (N3 "refused twice =
  refused"). A row with only network or 429 / 5xx failures stays open and is requested again on
  a resume.
- TL: the teacher with the D10 engine: d10_teacher.VLLM_ARGS unchanged (max_model_len 4096, awq,
  gpu_memory_utilization 0.90, max_num_seqs 1, enforce_eager), d10_teacher.SAMPLING (temperature
  0, top_p 1, seed 0), max_tokens 512. Messages: system = the D14 F3 message, user = the Urdu item
  block, through the model's chat template with the generation prompt. One generation per row.
- SL: Qwen3-14B with the student engine (11.7), `SamplingParams(temperature=0.0, max_tokens=512,
  stop=["<|im_end|>"])`; messages: system = the D14 F3 message, user = the Urdu item block,
  chat template with the generation prompt and enable_thinking=False. The translation text is the
  output after the last `</think>` if the output holds one, else the whole output. One
  generation per row.
- Validity (all three, D14 9.5): the output is stripped and split into lines; empty lines are
  dropped; every line is stripped. Valid if there are exactly five lines and the i-th line starts
  with the i-th label followed by ":" and has text after it. The item block is these lines, each
  rewritten as label + ": " + text with the text stripped, joined by "\n".
- An invalid TL or SL output, and a row whose TL or SL prompt does not fit (11.8), is
  reading_failed in the arms that use it (T_self, S_self) and counts as wrong there (N6).
- Flags (N3, reported, not a check), counted per translation on the final text of each row (the
  accepted output; for a reading_failed row the last output): (a) contains "correct"
  (case-insensitive); (b) contains "answer is" (case-insensitive); (c) a mark on one option:
  exactly one of the lines that start with "A:", "B:", "C:" or "D:" contains one of ✓ ✔ ✅ ☑ √ ←
  `*` `<-` `(correct)` `(answer)` (case-insensitive). The readout prints the three counts per
  translation and the count of rows with any of them.

### 11.6 Teacher facts (S_tr)

D14 9.7 unchanged: the D11 teacher fact prompt pair (d11o_teacher.oracle_prompts on the
d10_teacher T_SYSTEM and T_USER_TEMPLATE, through d14_common) with the R1 item block in its
question slot; max_tokens 512, temperature 0, top_p 1, seed 0; the D14 engine (11.7);
d10_teacher.parse_teacher_json, d10_teacher.verdict_filter, then the D14 filter
(d14_common.d14_filter, its patterns unchanged). Removed lines are stored and counted. A
reading_failed row (R1) has no facts and is not generated in S_tr.

### 11.7 Engines

- Student (S arms and SL): `LLM(model, dtype="bfloat16", max_model_len=8192,
  gpu_memory_utilization=0.85)` (the CSQA fair script's engine line, read from its source); S arms
  `SamplingParams(temperature=0.0, max_tokens=1024, stop=["<|im_end|>"])` (the CSQA fair value);
  answer text after the last `</think>` (the fair script's `strip().split("</think>")[-1]`). vLLM
  0.13.0 of env urbench_eval. SL and the S arms may run in one job with one loaded engine.
- Teacher arms and teacher facts: the D14 engine = d10_teacher.VLLM_ARGS with max_num_seqs 16 and
  max_num_batched_tokens 4096 (D14 Amendment 1 A1.2; d14_common.teacher_frozen), temperature 0,
  top_p 1, seed 0; arms max_tokens 1024, facts max_tokens 512.
- TL: the D10 engine (11.5), loaded in its own job; the teacher arms and facts are not run with
  it.
- Chunks of 64 prompts per call, in run order; every finished chunk is appended and synced
  (D14 9.6, 9.8).
- API arms: D14 9.9 settings (d11_api_direct constants through d14_common.api_settings):
  qwen3.8-max, enable_thinking false, temperature 0, 2 concurrent requests per arm process, the
  D11-R retry policy, max_tokens 1024. A row refused on two separate requests is recorded as
  refused, not requested again (D13 Amendment 3), and is outside set A.

### 11.8 Context length

As D14 9.10: prompt tokens + max_tokens <= max_model_len (8192 student, 4096 teacher). A row
whose prompt does not fit is not generated in that arm or translation; its record says too_long;
in an arm it counts as unparsed = wrong; in TL / SL it makes the row reading_failed for the arm
that uses it. The preflight (where the readings exist) and the readout report the counts.

### 11.9 API budget and order

- Block budget: at most 11,500 calls and 5,000,000 tokens (prompt + completion, summed from the
  usage fields). Every request that is sent counts as one call (retries and TEST-5 included). A
  shared ledger outputs/efbpt/d15/api_ledger.json is updated under a file lock by every process
  (the d14_common ledger code with the D15 path and caps); no request is sent once the calls
  reach the cap or the tokens reach 4,950,000 (so requests in flight cannot pass the cap). API
  work stops on the first quota or balance error.
- The amount used by the key is read before and after the API runs (the d14_balance query) and
  reported.
- Order: R1 first (alone); then M_ur, M_ue and M_r1, one process per arm, side by side.
- An arm cut by the budget is not sealed and not scored; its finished rows stay on disk and the
  readout lists it as missing. If an M arm is missing, set A is formed with the M arms that are
  sealed, and the readout says so; a test that needs a missing arm is reported missing and Holm
  still uses m = 3 / 2 / 1.
- Key handling (D14 9.9): the key file ~/.urbench_api_env is read only inside the Python process
  that makes the calls; never sourced into a shell, printed, logged or committed; every stored
  string is scrubbed; START and SEAL files record the host name only.

### 11.10 Scoring

- extract_answer: the function definition of the pinned fair script, taken from its source, with
  the letter class `[A-E]` replaced by `[A-D]` (asserted to occur exactly twice, once in each
  pattern) and run otherwise unchanged; the same function for every arm (the marker is "Answer"
  in UR, UE and EN). The preflight and the readout self-test run the same code path with the
  class left at `[A-E]` on the sealed CSQA fair output and must get 190 / 300 (fatal).
- Answer text: after the last `</think>` for Qwen3-14B (the fair script's
  `strip().split("</think>")[-1]`); the whole output for T and M.
- unparsed = empty prediction, or the row is reading_failed, refused or too_long in that arm.
  correct = prediction equals gold. truncated = finish reason "length".

### 11.11 Set A, lines, tests, predictions, reading

- Set A (N6): rows of the sample whose R1 status is ok and whose M_ur, M_ue and M_r1 records are
  answered (status ok). Dropped rows are counted overall, per group and per domain, split by
  cause (R1 reading_failed; refused in M_ur / M_ue / M_r1).
- Lines (D14 9.11, inside the row set of the comparison): LA = all rows; LP = rows parsed in both
  arms; LT = rows truncated in neither arm; LPT = both. Deciding line: LA; LP if the unparsed
  rates differ by more than 5.0 pp; LT if the truncated rates differ by more than 5.0 pp; LPT if
  both. Every line is printed for every test.
- Each H test is decided on set A. Printed beside it: the deciding line on all sampled rows; for
  S_self and T_self, the deciding line on the rows of set A with a valid local translation (SL
  for S_self, TL for T_self).
- McNemar: exact two-sided binomial on the discordant pairs; p = 1 when there is none. Holm:
  step-down inside each family (m = 3 for H1, 2 for H2, 1 for H3) on the raw p of the deciding
  lines on set A (the d14_readout functions). Delta = first arm minus second, in pp, on the
  deciding line. Wording: "helps" only if delta > 0 and Holm p < 0.05; "hurts" only if delta < 0
  and Holm p < 0.05; otherwise "no clear effect".
- Descriptive comparisons (N7) are printed with every line and raw p only. Group and domain
  splits: the rows of set A inside the group or domain, all four lines' counts, b, c and raw p on
  the deciding line of that split. Per-subdomain table: accuracy of each arm and the H1 / H2 / H3
  deltas on set A, no p.
- Predictions (N8) are read on LA of set A and marked hit or miss; "the delta in URDU is not
  above the delta in OTHER" is read on LA of set A inside each group.
- Reading (N9): H1 as written. H2, read "the same way" with two tests: both "helps" = the gain
  needs no stronger translator; none = not shown; otherwise "mixed", stated test by test.

### 11.12 Outputs, seals, TEST-5, readout

- Outputs under outputs/efbpt/d15/: tr/ (R1, TL, SL), facts/, student/, teacher/, api/. TEST-5
  outputs under outputs/efbpt/d15/_test/ with the same layout; never results, never scored. The
  full runs generate the TEST-5 rows again.
- Per arm or translation: one row file, one START file, one SEAL file (row count, MD5, script and
  freeze MD5s, job id). A seal is written only when every row is present; a sealed file is never
  rewritten. Each row stores the full prompt, the raw output, finish reason, seconds and tokens.
  Row files carry no pred, correct or gold field. No job computes or prints accuracy. A crashed
  run resumes; finished rows are not generated again.
- A full run needs the d15 scripts committed and unchanged.
- The readout is one script, d15_readout.py, committed and pushed before it opens any D15 arm
  file, with a self-test (the scorer on the sealed CSQA fair output with `[A-E]`, 190 / 300;
  hand-made Holm, McNemar, line, set-A and wording cases). It scores only when every arm is sealed
  or is an M arm listed as missing because the API budget ended. It is run once and writes
  docs/handoff/D15_READOUT.md.
- Leak check: the builders receive a copy of the row with `question` and `options` only, so
  `correct_key` cannot reach a prompt; the preflight asserts that every row copy given to a
  builder has exactly these two fields and that no prompt of any kind contains the string
  `correct_key`.

### 11.13 Jobs

SLURM partition q_intel_share_L20, conda env urbench_eval (conda.sh sourced directly), account
GPU limit 2, `--exclude=L20004,L20007`, time limit 24 h. API runs are started detached on the
login node (setsid nohup) and are resumable. Order: CPU preflight; TEST-5; R1, then TL and SL;
API arms; teacher facts; student arms; teacher arms. Job order affects no number.
