# EFBPT D14 — READING EFFECT AND TEACHER FACTS ON THREE MORE TASKS (FREEZE)

Status: FROZEN 2026-11-15, before any d14 script, job or output exists.

Purpose: measure, on three more URBench tasks, (1) the reading effect: the same model on the
Urdu item against an English translation of it, and (2) the teacher-facts method with the
Qwen3-14B student. D14 selects nothing and tunes nothing. No earlier verdict changes whatever
D14 shows. No arm is run again after it is scored.

Sections F1 to F8 are the frozen design. Section 9 holds the operational details that make
F1 to F8 executable; it changes no arm, prompt, model, decoding or decision rule.

No existing file under docs/, no existing d1* ... d13* script or fair-baseline script and no
sealed file under outputs/ is edited, moved or deleted. New code is in new files with the
prefix d14_.

## F1. Sets

All rows of the Urdu source files, with the original English row at the same position
(docs/handoff/D14_PREP.md §1):

| task | rows | Urdu source file | English source file |
|---|---:|---|---|
| CSQA | 1500 | data/csqa_raw/csqa_train_1500_ur.jsonl | data/csqa_raw/csqa_train_1500_en.jsonl |
| PIQA | 750 | data/piqa_raw/piqa_train_750_ur_fixed.jsonl | data/piqa_raw/piqa_train_750.jsonl |
| GSM8K | 700 | data/gsm8k_raw/gsm8k_main_train_700_ur.jsonl | data/gsm8k_raw/gsm8k_main_train_700.jsonl |

BoolQ is left out (long passages). The known oddities of docs/KNOWN_DATA_ISSUES.md stay as
they are. D14 uses no demonstrations and no pool.

## F2. Prompts

UR = the fair baseline prompt of the task, byte-identical: cot_csqa_baseline_fair.py,
cot_piqa_baseline_fair.py, cot_gsm8k_baseline_fair.py (same messages for every model).

EN = this English mirror, same layout. Each prompt is instruction + "\n\n" + item block +
"\n\n" + trailer. The strings below are written as string literals; the only line breaks
inside them are the "\n" that are written.

CSQA
- instruction: `"You are given a question and five options (A, B, C, D, E).\nFirst think briefly about which answer is correct.\nAt the end use only this form: Answer: X\nwhere X is only A, B, C, D or E."`
- item block: `"Question: {q}\nA: {A}\nB: {B}\nC: {C}\nD: {D}\nE: {E}"`
- trailer: `"Step-by-step reasoning:\n\nAnswer:"`

PIQA
- instruction: `"A goal and two possible solutions are given.\nFirst think step by step and in detail about which solution is more sensible and agrees with common sense.\nThen at the end write only one digit: 0 or 1.\n0 means Solution 1 is correct, 1 means Solution 2 is correct."`
- item block: `"Goal: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}"`
- trailer: `"Reasoning steps:\nFinal answer:"`

GSM8K
- instruction: `"You are given a math word problem.\nFirst think step by step and solve the problem.\nAfter that, write only one numeric answer after \"Final answer:\".\nDo not include any unit, words or extra text in the final answer."`
- item block: `"Problem: {question}"`
- trailer: `"Reasoning steps:\nFinal answer:"`

Item block = the item lines with the English labels of the EN prompt (Question / A..E,
Goal / Solution 1 / Solution 2, Problem).

## F3. Translation R1

One API call per row, qwen3.8-max, thinking off, temperature 0, max_tokens 512, 4 concurrent,
D11-R retry policy.

SYSTEM: `"You are an expert Urdu-to-English translator. Translate the text exactly, preserving its meaning and its line structure: keep every label (such as 'A:', 'Goal:', 'Solution 1:') and every number unchanged, one output line per input line. Named entities are usually transliterations of English names; write each in its standard English spelling. Do not answer, do not solve and do not explain. Output only the English translation."`

USER: the item block with the Urdu text in it. The gold answer is never sent.

A translation is valid if it has the same labels in the same order. An invalid one is
requested once more; if still invalid, or refused by the provider twice, the row is marked
reading_failed and counts as unparsed = wrong in every arm that needs its reading.

## F4. Arms, for each task

Student S = Qwen3-14B with the engine and sampling of the task's fair script (vLLM, bf16,
thinking on, its max_tokens, answer read from the text after `</think>`):

| arm | definition |
|---|---|
| S_ur | UR prompt. |
| S_ro | UR prompt, with one block put between the last item line and the reasoning trailer: the Urdu label for "English translation:" then the R1 item block. The label is built from the d11_ablate_ro constant by removing its two leading words; it is not retyped. |
| S_og | as S_ro with the original English item block (ceiling of perfect reading). |
| S_tr | (CSQA and PIQA only) UR prompt with a facts block at the same place, in the frozen d1_eval_arms format. Facts: the D11 teacher fact prompt unchanged (JSON, at most 6 facts), its question slot filled with the R1 item block; D10 verdict filter T2 unchanged; in addition a fact line is removed if it contains, case-insensitive, "option", "choice", "solution 1", "solution 2" or an option label such as "A:" or "(A)". Removed lines are stored and counted. Rows with no fact left get an empty block. |

Teacher T = Llama-3.1-70B-AWQ, D10 engine with max_num_seqs 16, max_tokens 1024:
T_ur = UR prompt; T_r1 = EN prompt with the R1 item block; T_og = EN prompt with the original.

Commercial M = qwen3.8-max, D11 A1.6 settings with max_tokens 1024:
M_ur, M_r1, M_og with the same three prompts.

All arms of one model and task run in one job with identical engine arguments.

Arms per task: CSQA 10, PIQA 10, GSM8K 9 (no S_tr). 29 arm files in all.

## F5. Scoring

Each task's extract_answer and gold rule from its fair script, unchanged, on the answer text
(after `</think>` for Qwen3-14B; the whole output otherwise). For EN-prompt arms of PIQA and
GSM8K the same function with "Final answer" (case-insensitive) in place of the Urdu marker.
Empty prediction = unparsed = wrong. Truncated = reached max_tokens.

## F6. Tests

Exact two-sided McNemar, b = first arm right and second wrong.

Family G1, reading effect, 9 tests, Holm at 0.05: S_ro vs S_ur, T_r1 vs T_ur, M_r1 vs M_ur,
on each of the three tasks.

Family G2, method, 2 tests, Holm at 0.05: S_tr vs S_ur on CSQA and on PIQA.

Descriptive, no claim: *_og vs *_ur, *_og vs *_r1, S_tr vs S_ro, S_tr vs T_r1.

Deciding line as in D13 Amendment 2: all rows; rows parsed in both arms if the unparsed
rates differ by more than 5.0 pp; rows truncated in neither if the truncated rates differ by
more than 5.0 pp; the intersection if both. The all-rows line is always printed beside it.

Wording, fixed now: "helps" only if delta > 0 and Holm p < 0.05; "hurts" only if delta < 0
and Holm p < 0.05; otherwise "no clear effect".

Power (rough): a raw p < 0.05 needs about 2-3 pp on CSQA and GSM8K and about 3-4 pp on PIQA.

## F7. Predictions, written before any run (all rows)

- S_ro - S_ur in [0, +8] on CSQA, [-2, +8] on PIQA, [-2, +4] on GSM8K.
- T_r1 - T_ur and M_r1 - M_ur in [0, +10] on every task.
- S_tr - S_ur in [-3, +6] on CSQA and on PIQA.
- Every *_og is at least its *_r1 minus 2.

## F8. Checks

Preflight (CPU): row counts 1500 / 750 / 700; Urdu and English rows aligned by position (ids
equal where they exist, except the one known CSQA position); the rebuilt UR prompts equal the
prompts stored in the sealed fair outputs wherever that field exists; the task scorers
reproduce 190/300, 108/150 and 621/700 on the sealed fair outputs (fatal). No gold answer in
any prompt. TEST-5 = rows 0-4 of each task, every arm, never scored and never a result. One
seal per arm; nothing is scored before every arm of a task is sealed. A crashed arm resumes;
finished rows are not generated again. Seconds and tokens per row are logged.

## 9. Operational details, fixed now

### 9.1 Pinned inputs (MD5)

Data files (not tracked by git):

| file | MD5 |
|---|---|
| data/csqa_raw/csqa_train_1500_ur.jsonl | 396601c0f6c1b26aaa4c29c3fc6cf018 |
| data/csqa_raw/csqa_train_1500_en.jsonl | 4a47619e60c4be2478757b8c8e6581bc |
| data/piqa_raw/piqa_train_750_ur_fixed.jsonl | 01586d82da54b781aefb1d5427389515 |
| data/piqa_raw/piqa_train_750.jsonl | 9176d3b16cddebde5401ffe54dec49c2 |
| data/gsm8k_raw/gsm8k_main_train_700_ur.jsonl | ffb57c3df417799ad1d03f79a5aafc12 |
| data/gsm8k_raw/gsm8k_main_train_700.jsonl | 651a81016197dbaba7bc4b8a64d55c82 |

Fair scripts (eval/error_analysis_tests/) and their sealed outputs (outputs/sdfr/):

| script | MD5 | sealed output | MD5 | correct / rows |
|---|---|---|---|---|
| cot_csqa_baseline_fair.py | 70491653824525aeaa58a3282d5d177f | cot_csqa_baseline_fair_qwen3_14b.jsonl | 40e09c2208cce4217f86fb54c1cc034c | 190 / 300 |
| cot_piqa_baseline_fair.py | cd6f074c4ea96820226b48768767ad37 | cot_piqa_baseline_fair_qwen3_14b.jsonl | d969451aa8563c4524e89aef959553d5 | 108 / 150 |
| cot_gsm8k_baseline_fair.py | 0ac419f37b59b5695434736f9abe9a4a | cot_gsm8k_baseline_fair_qwen3_14b.jsonl | d94ea3127817cf7357ca085f7a06e273 | 621 / 700 |

The sealed fair outputs are the evaluation splits (docs/handoff/D14_PREP.md §1): CSQA source
positions 1200 to 1499, PIQA 600 to 749, GSM8K 0 to 699. They store no prompt field, so the
prompt-equality check of F8 has nothing to compare; the preflight reports this.

Frozen code that D14 imports or reads (eval/error_analysis_tests/efbpt/): d10_teacher.py
60bca5ba3d60e28e93fac1c643bf6802; d11o_teacher.py 338df12c1c8b2f3c362b1f2076244c3b;
d1_eval_arms.py 10917f65b7852b406aec709a95d0a45b; d11_ablate_ro.py
b3d42f4c7b6805d4f91f45a2e28dca5b; d11_api_direct.py 4af85505d859b481aaf1ab0caf26905f;
d11r_common.py 4205923e6aca8294d9c1e10f860523a7.

Documents cited: docs/KNOWN_DATA_ISSUES.md 4168fcd9a0de984445adc560cb6fff0b;
docs/handoff/D14_PREP.md bc2480c98ed4c98bab18be22e5f9db8c; docs/EFBPT_D13_AMENDMENT_2.md
12809c115ae8b6fba6bf6afe7fb6c51c.

Models: Qwen3-14B = downloaded_models/Qwen/Qwen3-14B; teacher =
downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4 (the d10_teacher
constant). No download.

### 9.2 Rows, ids, gold

- A row is the row at one position of the Urdu source file together with the row at the same
  position of the English source file. Row id = task and 0-based position, e.g. `csqa_0000`.
- Gold is taken from the Urdu source row with the rule of the fair script: CSQA
  `answerKey.strip().upper()`; PIQA `str(label)`; GSM8K `extract_gold(answer)`. Gold is used
  only by the readout. It is not stored in any arm file.
- Seen on 2026-11-15 while writing this freeze, left as it is: the PIQA Urdu and English source
  files give a different `label` at 5 of 750 positions (0-based 524, 631, 637, 640, 704). The
  Urdu label is the gold, as in the fair script. The readout reports the count; nothing is
  changed.

### 9.3 UR prompt

The user message is built by the statements of the pinned fair script itself (the assignments
of the option letters, `instr`, `q_block` and `raw` inside its prompt loop are taken from the
script source and run on the Urdu source row), so the text is not retyped. The fields are used
as they are in the file, as the fair script does. The message list of every model is this one
user message; there is no system message. Each model's own chat template is used with the
generation prompt added (Qwen3-14B: enable_thinking=True, as in the fair script; the API
receives the message list).

The UR user message has three parts joined by blank lines: instruction, item lines, trailer.
The trailers are the last part of `q_block` in each fair script.

### 9.4 Item blocks

- An item block has one line per label: CSQA 6 lines (Question, A, B, C, D, E), PIQA 3 lines
  (Goal, Solution 1, Solution 2), GSM8K 1 line (Problem). Each line is label + ": " + text.
- Inside an item block every run of white space in a field (line breaks included) is one
  space and the field is stripped, so that a field is one line. This applies to item blocks
  only, never to the UR prompt. (One English PIQA row, 0-based 49, has a line break in a
  field; 27 Urdu PIQA rows have a leading or trailing space in a field.)
- Urdu item block = the labels with the Urdu fields (the USER message of F3).
- Original item block = the labels with the fields of the English source row.
- R1 item block = the valid translation of the row (9.5).

### 9.5 Translation R1

- Request: model qwen3.8-max, enable_thinking false, temperature 0, max_tokens 512; system and
  user as F3; 4 concurrent requests. Retry policy of D11-R (d11r_common constants): 429, 500,
  502, 503, 504 and connection errors, backoff 2 / 4 / 8 s, 3 attempts.
- Validity: the output is stripped and split into lines; empty lines are dropped; every line
  is stripped. It is valid if the number of lines equals the number of labels and the i-th
  line starts with the i-th label followed by ":" and has text after it. The R1 item block is
  these lines, each rewritten as label + ": " + text with the text stripped, joined by "\n".
- An answered request with an invalid output is requested once more with the same messages.
  If the second output is also invalid the row is reading_failed. A request refused by the
  provider's input inspection (HTTP 400) on two separate requests is reading_failed (the D13
  Amendment 3 rule). A row that has only network or 429 / 5xx failures is not decided: it
  stays open and is requested again on a resume.
- One sealed file per task: outputs/efbpt/d14/r1/d14_r1_<task>.jsonl, one record per row in
  row order with the row id, the messages, every raw output, the R1 item block or
  reading_failed, seconds and tokens. The record has no gold field.
- A reading_failed row is not generated in the arms that need the reading (S_ro, S_tr, T_r1,
  M_r1). Its arm record says reading_failed and it counts as unparsed = wrong there.

### 9.6 Student arms

- Engine and sampling of the fair scripts: vLLM, `LLM(model, dtype="bfloat16",
  max_model_len=8192, gpu_memory_utilization=0.85)`, `SamplingParams(temperature=0.0,
  max_tokens=N, stop=["<|im_end|>"])` with N = 1024 for CSQA, 2048 for PIQA and 2048 for GSM8K
  (the values on the quoted lines of D14_PREP §2). vLLM 0.13.0 of env urbench_eval.
- S_ro and S_og user message: instruction, item lines, added block, trailer, joined by blank
  lines. Added block = label line + "\n" + item block. The label line is the d11_ablate_ro
  constant RO_LABEL without its two leading words and without the space after its colon.
- S_tr user message: instruction, item lines, facts block, trailer, joined by blank lines.
  Facts block = d1_eval_arms.facts_block(facts) (header line, then one "- fact" line per
  fact). A row with no fact left has no block text at all: its S_tr prompt is then equal to
  its S_ur prompt (the D10 / D11 layout rule for a row without facts); the row is still
  generated in S_tr and counted.
- The prompts of one job are generated in row order in chunks of 64 prompts per call; every
  finished chunk is appended and synced. Arms of one task run one after another in one job
  with one loaded engine. The answer text is the text after the last `</think>` (the fair
  script's `split("</think>")[-1]`).

### 9.7 Teacher facts (for S_tr; CSQA and PIQA)

- Prompt pair of D11 (d11o_teacher.oracle_prompts on the d10_teacher T_SYSTEM and
  T_USER_TEMPLATE, imported): the question slot is filled with the R1 item block. Nothing else
  from the row enters the prompt. max_tokens 512 (the D10 / D11 value), temperature 0, top_p 1,
  seed 0.
- Engine: the D10 engine arguments (d10_teacher.VLLM_ARGS, imported) with max_num_seqs 16, the
  same engine as the teacher arms.
- Parsing: d10_teacher.parse_teacher_json, unchanged. Filter: d10_teacher.verdict_filter,
  unchanged (it keeps at most 6). Then the D14 filter on the kept lines: a line is removed if
  it matches, case-insensitive, `option|choice|solution 1|solution 2`, or, case-sensitive, an
  option label `(?<![A-Za-z0-9])[A-E]:` or `\([A-E]\)`. Lines removed by each filter are stored
  and counted.
- A row whose reply does not parse, or with no line left, has no facts.
- One sealed file per task: outputs/efbpt/d14/facts/d14_facts_<task>.jsonl.

### 9.8 Teacher arms

- Engine: d10_teacher.VLLM_ARGS (max_model_len 4096, awq, gpu_memory_utilization 0.90,
  enforce_eager) with max_num_seqs 16; temperature 0, top_p 1, seed 0 (d10_teacher.SAMPLING);
  max_tokens 1024. One user message through the model's chat template.
- Chunks of 64 prompts per call, in row order, appended and synced. The three arms of one task
  run in one job.

### 9.9 API arms

- Settings of D11 Amendment 1 A1.6 (d11_api_direct constants, imported): qwen3.8-max,
  enable_thinking false, temperature 0, 2 concurrent requests per arm process, the D11-R retry
  policy; max_tokens 1024. One user message, no system message.
- A row refused by the provider on two separate requests is recorded as refused, not requested
  again, and counts as unparsed = wrong (D13 Amendment 3).
- Order: M_ur, M_r1, M_og. The three tasks may run side by side, one process per task; within
  a task the order is M_ur, M_r1, M_og. If the token use seen in TEST-5 shows that the budget
  cannot cover all nine arms, M_og (descriptive only) is run last on every task, after every
  M_ur and M_r1. This choice is made from TEST-5 token counts only, never from an answer.
- Budget for the whole block: at most 13,000 calls and 7,000,000 tokens. Every request that is
  sent counts as one call (retries and TEST-5 included); tokens are prompt + completion from
  the usage fields. A shared ledger file under outputs/efbpt/d14/ is updated under a file lock
  by every process; a process sends no request once a cap is reached. API work stops on the
  first quota or balance error. An arm cut by the budget is not sealed and not scored; its
  finished rows stay on disk and the readout lists it as missing.
- Key handling: the key file ~/.urbench_api_env is read only inside the Python process that
  makes the calls. It is never sourced into a shell, never printed, logged or committed; every
  stored string is scrubbed; START and SEAL files record the host name only.

### 9.10 Context length

A prompt must fit: prompt tokens + max_tokens <= the engine's max_model_len (8192 for the
student, 4096 for the teacher). A row whose prompt does not fit in an arm is not generated
there; its record says too_long and it counts as unparsed = wrong in that arm. The preflight
and the readout report the count per arm.

### 9.11 Scoring details

- Scorers: the functions `extract_answer` (and `extract_gold`) are taken from the source of
  the pinned fair scripts. The EN variant for PIQA and GSM8K is the same source with the
  marker pattern `حتمی\s*جواب` replaced by `(?i:Final\s*answer)`; CSQA uses one function for
  every arm (its marker is "Answer" in both prompts). UR-prompt arms (S_ur, S_ro, S_og, S_tr,
  T_ur, M_ur) use the unchanged function; T_r1, T_og, M_r1, M_og use the EN variant.
- unparsed = the prediction is the empty string, or the row is reading_failed, refused or
  too_long. correct = prediction equals gold.
- truncated = finish reason "length".
- Lines: LA = all rows; LP = rows parsed in both arms; LT = rows truncated in neither arm;
  LPT = both. Deciding line: LA; LP if the unparsed rates differ by more than 5.0 pp; LT if
  the truncated rates differ by more than 5.0 pp; LPT if both (D13 Amendment 2). Every line is
  printed for every comparison.
- Delta = accuracy of the first-named arm minus the second, in pp, on the deciding line.
- Holm as in the D13 freeze §3: step-down on the raw p-values of the deciding lines of one
  family; the adjusted p of the i-th smallest of m is the largest of min(1, (m - j + 1) * p_(j))
  over j <= i. A test whose arm is missing is reported as missing and takes no wording; Holm
  still uses m = 9 (G1) and m = 2 (G2).
- McNemar: exact two-sided binomial on the discordant pairs; p = 1 when there is none.
- Predictions (F7) are read on LA and marked hit or miss.

### 9.12 Outputs, seals, TEST-5

- Outputs under outputs/efbpt/d14/: r1/, facts/, student/<task>/, teacher/<task>/, api/<task>/.
  TEST-5 outputs under outputs/efbpt/d14/_test/ with the same layout; they are never results
  and never scored. The full runs generate rows 0-4 again.
- Per arm: one row file, one START file, one SEAL file (row count, MD5, script and freeze
  MD5s, job id). A seal is written only when every row is present; a sealed file is never
  rewritten. Each row stores the full prompt, the raw output, finish reason, seconds and
  tokens. Seconds of a row generated in a chunk are the chunk's seconds divided by its rows.
  Row files carry no pred, correct or gold field. No job computes or prints accuracy.
- A full run needs the d14 scripts committed and unchanged.
- The readout is one script, committed and pushed before it opens any D14 arm file, with a
  self-test on the sealed fair outputs and on hand-made Holm cases. A task is scored only when
  every arm of the task is sealed or listed as missing because the API budget ended.
- Leak check: no prompt of any arm contains the gold answer field of its row as a separate
  slot; the builders never read `answerKey`, `label` or `answer`. The preflight asserts that
  the GSM8K `answer` text of a row is not a substring of any of its prompts.

### 9.13 Jobs

SLURM partition q_intel_share_L20, conda env urbench_eval (conda.sh sourced directly), account
GPU limit 2, `--exclude=L20004,L20007`, time limit 24 h. API runs are started detached on the
login node (setsid nohup) and are resumable. Order: CPU preflight; TEST-5; R1 translations;
API arms; teacher facts; student jobs; teacher arm jobs. Job order affects no number.
