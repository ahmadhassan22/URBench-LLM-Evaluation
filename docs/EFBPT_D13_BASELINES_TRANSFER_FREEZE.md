# EFBPT D13 — BASELINES AND TRANSFER AFTER THE D11 HELD-OUT TEST (FREEZE)

Status: FROZEN 2026-11-12, before any D13 script, job or output exists.
(The name D12 was reserved in the D11 freeze §7 for a fallback that never ran, so this is D13.)

Purpose: baselines and a transfer test asked for after the D11 held-out test.
D13 selects nothing and tunes nothing. The D11 verdict CONFIRMED (docs/handoff/
D11_EVAL458_READOUT_LINES.md, MD5 9f95a95ecd0a92fe6362b438c3f900a4) stays as it is whatever D13
shows. No arm is run again after it is scored.

Conventions, unchanged: exact two-sided McNemar (d5_balanced_extract.mcnemar); unparsed = wrong;
b = the first-named arm right and the second wrong, c = the reverse; E0 = the frozen extractor
(efbpt_eval_dev200.extract_answer); E3 = the validated scorer (d11_scorer_audit.e3, script MD5
246cf6c9687b3fa67324722cb5cd16b2). No existing file in docs/, no existing d1*/d8*/d10*/d11*
script and no sealed file under outputs/ is edited, moved or deleted. New code is in new files
with the prefix d13_ and imports the frozen D10/D11 code.

## 1. Sets and sealed inputs

DEV200: 200 rows, 87 yes / 113 no, floor 56.50. Descriptive only.
eval458: 458 rows, 221 yes / 237 no, floor 51.75. eval458 decides.
Both are loaded with d11_common.load_set_rows (pinned item files, counts and overlap 0 asserted).

"Reading" always means the sealed D11-R reading file of that set (MD5 checked against the
reader's R_SEAL.json and against the values below):

| reading | DEV200 MD5 | eval458 MD5 |
|---|---|---|
| R1 = api_qwen3.8-max | 1f5b2db3c1403f63b3525a1562edd176 | 3211b45ff4bf6b75f07d7ed60d4991eb |
| R3 = L1_llama70b_awq | 0d06ec1a926fe3c186c125ad723139cf | 9833dc91c9a2ee42c667f4f70747dbec |

Sealed R1 teacher facts (the facts of arm TR): DEV200 outputs/efbpt/d11/teacher/dev200/R1/
t_facts.jsonl MD5 160406e12205635e69f5c95140f9936f; eval458 outputs/efbpt/d11/teacher/eval458/R1/
t_facts.jsonl MD5 4e2c4f30fd9603711ded98df90fdc477.

Sealed reference arms, read only, never run again:

| arm | set | file (outputs/efbpt/…) | MD5 | E0 acc % (correct) |
|---|---|---|---|---|
| A_r2_eval | eval458 | d11/student/eval458/d11_armA_r2_eval.jsonl | f139ba9247cdd9fef191f00d404d8212 | 54.80 (251) |
| TR | eval458 | d11/student/eval458/d11_armTR.jsonl | e38d64114089a88b2deb202de0240ca9 | 68.78 (315) |
| TR_L1_eval | eval458 | d11/student/eval458/d11_armTR_L1_eval.jsonl | 4a1b2118bbcaa9348713a600b9281733 | 64.63 (296) |
| A_r2 | DEV200 | d10/student/d10_armA_r2.jsonl | 7ccec942c8f0eee08156d258baaf806d | 60.00 (120) |
| TR | DEV200 | d11/student/dev200/d11_armTR.jsonl | bab987cc492012f4e32dbcd348be9668 | 68.50 (137) |
| TR_L1 | DEV200 | d11/student/dev200/d11_armTR_L1.jsonl | b8db599f8c4b427ddcb2d772fd8f09aa | 67.00 (134) |
| T_direct_F | DEV200 | d10/teacher/t_direct_f.jsonl | 40ebfe643e244c5a1a3fff729f23462c | 55.50 (111) |
| API_direct_ur | DEV200 | d11/api_direct/dev200/api_direct_ur.jsonl | 31c82fa180a2677ce9df85db2ac3e0d7 | 69.00 (138) |
| RO | DEV200 | d11/ablate_ro/dev200/d11_armRO.jsonl | d3a8850836389d89eac5369601de2f98 | 57.50 (115) |

These numbers were recomputed from the sealed files on 2026-11-12 with E0. They are what is
known before D13: every DEV200 number above, and the D11 eval458 readout. No D13 arm exists.

## 2. Arms (B1)

Group 1, large model alone (no student):

| arm | definition | sets |
|---|---|---|
| T_direct_F_eval | Llama-3.1-70B-AWQ, the D10 Amendment 1 A1.1 prompt unchanged (Urdu question + final-answer line) | eval458 |
| T_tr_R1 | same teacher and engine. Input = the R1 English reading only. English instruction that ends with one line: "Final answer: yes" or "Final answer: no" | dev200, eval458 |
| T_tr_L1 | as T_tr_R1 with the R3 (L1_llama70b_awq) reading. Fully local | dev200, eval458 |
| API_direct_ur_eval | D11 Amendment 1 A1.6 unchanged (qwen3.8-max, Urdu question alone) | eval458 |
| API_tr_R1 | qwen3.8-max, input = the R1 reading, system "Answer only yes or no.", other settings as A1.6 | dev200, eval458 |

Group 2, traditional RAG (student = Qwen3-14B, D10 §2 answer pass, layout B, the retrieved text
sits in the facts block):

| arm | query | sets |
|---|---|---|
| RAG_ur | the Urdu question | dev200, eval458 |
| RAG_en | the R1 reading | dev200, eval458 |

Retriever: rag/retrieve.py, wikipedia_full.index (23,963,971 vectors), MiniLM, no prefix.
Top 5 chunks in rank order, each as "Title: text". No filter, no rerank. k = 5 is fixed.
Also reported, per arm: the share of gold evidence titles found in the top 5 (D8 §3 rule).

Group 3, ablation:

| arm | definition | set |
|---|---|---|
| RO_eval | D11 Amendment 3 A3.6 arm RO on eval458 (translation line, no facts) | eval458 |

Group 4, transfer to other students. Set: eval458 only.
For each S in {Llama-3.1-8B-Instruct, Gemma-2-9B-it, Qwen3-8B}:
S_A = layout A, no facts. S_TR = layout B with the same sealed R1 teacher facts as TR.
Same Urdu instruction, greedy, max_new 1024, batch size 1, nf4 double-quant, the model's own chat
template, thinking off where it exists. Gemma has no system role: the system text is put at the
start of the user turn. The weights must already be on the cluster. If one is missing: STOP and
report. No download.

Arm names in Group 4: Llama8B_A, Llama8B_TR, Gemma9B_A, Gemma9B_TR, Qwen8B_A, Qwen8B_TR.
Total new arms: 5 + 2 + 1 + 6 = 14; with DEV200 copies of T_tr_R1, T_tr_L1, API_tr_R1, RAG_ur and
RAG_en there are 19 sealed arm files.

## 3. Tests (B2)

eval458 decides. DEV200 numbers are descriptive.

Family F1 (7 tests, Holm at 0.05): TR vs T_direct_F_eval, T_tr_R1, T_tr_L1, API_direct_ur_eval,
API_tr_R1, RAG_ur, RAG_en. TR is the sealed D11 eval458 arm.

Family F2 (3 tests, Holm at 0.05): S_TR vs S_A for each S.

Descriptive, no claim: TR_L1_eval vs T_direct_F_eval and vs T_tr_L1 (all-local pair);
TR vs RO_eval; RO_eval vs A_r2_eval; RAG_ur and RAG_en vs A_r2_eval.

Every comparison prints the four lines L1-L4 of D11 Amendment 5 A5.3, plus
L5 = E0 on the 408 eval458 rows left after dropping the 50 rows that the old answer-free probes
used (the first 50 rows of data/sdfr_splits/strategyqa_eval.jsonl in file order, MD5
8f26dae65cfe22678b46359ff9ad43d2; D8 freeze §1; assert 408 rows remain).
L1 (E0, all rows) decides. Unparsed rule = D11 Amendment 3 A3.1: if the unparsed rates of the two
arms differ by more than 5.0 pp, the test is read on L3 (rows parsed in both arms) and the L1
numbers are reported beside it.

Checked on 2026-11-12: that file has the same 458 qids in the same order as the eval458 loader;
after dropping its first 50 rows, 408 rows remain (197 yes / 211 no).

Holm: step-down on the raw p-values of the deciding lines of one family; the adjusted p of the
i-th smallest of m is the largest of min(1, (m - j + 1) * p_(j)) over j <= i. "Holm p" below is
this adjusted p. Delta = accuracy of the first-named arm minus the second, in pp, on the deciding
line.

Power: with 120 to 170 discordant pairs, raw p < 0.05 needs |b - c| >= 24 to 28 rows (5.2 to
6.1 pp). The first Holm step in F1 (0.05/7) needs 32 to 36 rows (7.0 to 7.9 pp); in F2 (0.05/3)
28 to 34 rows (6.1 to 7.4 pp). So a gap under about 5 pp cannot be separated on eval458.
(Recomputed with the exact test at 120 and at 170 discordant pairs; an odd number of pairs
moves a bound by one row.)

DEV200, descriptive: TR (sealed) vs T_direct_F (sealed D10), T_tr_R1, T_tr_L1, API_direct_ur
(sealed D11), API_tr_R1, RAG_ur, RAG_en; RAG_ur and RAG_en vs A_r2; TR_L1 vs T_tr_L1. Lines L1-L4,
raw p only, no wording.

## 4. Wording rules, fixed now (B3)

"TR beats X" only if delta > 0 and Holm p < 0.05. "X beats TR" only if delta < 0 and
Holm p < 0.05. Everything else is "not distinguishable". No other wording is allowed.
Transfer "holds" if at least 2 of 3 students have delta > 0 with Holm p < 0.05;
"partial" if 1; "not shown" if 0.

## 5. Predictions, written before any run (B4)

- T_direct_F_eval in [48, 60] (DEV200 55.50).
- API_direct_ur_eval in [62, 72] (DEV200 class rates 40/87 and 98/113 moved to 221/237 = 67.06).
- T_tr_R1 on eval458 in [60, 76]; T_tr_L1 within 4 pp of T_tr_R1.
- RAG_ur - A_r2_eval and RAG_en - A_r2_eval in [-6, +4].
- RO_eval - A_r2_eval in [-3, +5].
- S_TR - S_A > 0 for each S.
- Expected reading: TR and API_direct_ur_eval not distinguishable.

Predictions are read on L1 (E0, all rows) of eval458 and marked hit or miss in the readout.

## 6. Checks, all fatal unless said (B5)

- C1. Before any Qwen3-14B student job: the D11 Amendment 2 A2.3 regression (DEV200, arm TR,
  rows 0-4 byte-identical to the sealed file, MD5 bab987cc492012f4e32dbcd348be9668), run through
  the D13 student path. Every Qwen3-14B student job depends afterok on it.
- C2. Each new student: 5 DEV200 rows generated twice must be byte-identical.
- C3. Leak guards of d11_common (with Amendment 6) on every prompt of every arm (§7.5).
- C4. TEST-5 uses DEV200 rows only. No eval458 row is generated in any test.
- C5. One seal per arm. Scoring is in two stages. Stage 1: Groups 1-3 on both sets (family F1
  and the descriptive comparisons) are scored when every Group 1-3 arm is sealed. Stage 2:
  Group 4 (family F2) is scored when all six Group 4 arms are sealed. The readout script is
  committed before stage 1 and is not changed between the stages. A crashed arm resumes;
  finished rows are never generated again.
- C6. Seconds and generated tokens per row are logged for every arm (reported, not a check).
- C7. CPU preflight before any job: row counts 200 and 458, 221 / 237, reading MD5s, leak guards.

## 7. Operational details, fixed now

These make §2-§6 executable. They change no arm, model, decoding or decision rule.

### 7.1 Teacher arms (Llama-3.1-70B-Instruct-AWQ-INT4)

Engine exactly D10 §2 (d10_teacher constants, imported): vLLM 0.13.0, max_model_len 4096,
tensor_parallel_size 1, quantization awq, gpu_memory_utilization 0.90, max_num_seqs 1,
enforce_eager; temperature 0.0, top_p 1.0, seed 0; max_tokens 1024; the model's chat template.

- T_direct_F_eval: system = SYSTEM_MESSAGE; user = build_user(instruction, row, None, "A") +
  blank line + TDF_LINE, all imported from the frozen code (D10 Amendment 1 A1.1, MD5
  ac828661e26d77ca9bf00695ce4b8b93). In TEST-5 the five DEV200 prompts must be byte-identical to
  the prompts stored in the sealed D10 file t_direct_f.jsonl (fatal); whether the five
  generations are byte-identical is reported.
- T_tr_R1, T_tr_L1: system = SYSTEM_MESSAGE ("You are a helpful assistant. Answer the user's
  question."); user = the three parts below joined by blank lines. It is the English mirror of the
  T_direct_F prompt (instruction, question, final-answer line):

      Answer this question.

      {reading}

      At the end, write on a separate line: Final answer: yes or Final answer: no

  Nothing else from the row enters the prompt.
- Scoring of T_tr_R1 and T_tr_L1 (English output). E0 and E3 take the LAST English yes/no word
  in the whole text, so a 'no' used as an ordinary word after the answer line would be read as
  the answer. For these two arms the label is E0fa: the first word-bounded yes/no after the LAST
  'Final answer' (case-insensitive); if the marker is absent, E0. E0fa replaces E0 for these two
  arms on lines L1, L3 and L5. Per arm the readout also reports the plain E0 and E3 accuracy,
  the number of rows without the marker, and the number of rows where E0fa and E0 differ.
  This is the English mirror of the Rule 2 marker cut.

### 7.2 API arms (qwen3.8-max)

Settings of D11 Amendment 1 A1.6: enable_thinking=false, temperature 0, max_tokens 256,
2 concurrent, D11-R retry policy, resumable (ok records kept), run on the login node.
- API_direct_ur_eval: system "صرف ہاں یا نہیں لکھیں" (the d11_api_direct constant), user =
  question_ur. In TEST-5 the five DEV200 answers are compared with the sealed DEV200 file
  (reported, not fatal).
- API_tr_R1: system "Answer only yes or no.", user = the R1 reading.
Key handling: the key file ~/.urbench_api_env is read only inside the Python process that makes
the calls. It is never sourced into a shell, never printed, logged or committed; every stored
string is scrubbed; START and SEAL files record the host name only. The key ends about
2026-11-20, so the API arms run first.

### 7.3 RAG arms

- Retrieval is a separate step with its own sealed file per arm and set. Retriever class of
  rag/retrieve.py (MD5 c1ecdf1df7ebf8738425cb836bf3d218), unchanged: FAISS flat inner-product
  index rag/index/wikipedia_full.index (assert ntotal = 23,963,971), embedder
  paraphrase-multilingual-MiniLM-L12-v2, normalized, no prefix. The query string is exactly
  question_ur (RAG_ur) or exactly the R1 reading (RAG_en). top_k = 5.
- Each retrieved chunk becomes one facts-block entry "Title: text", in rank order. Runs of
  white space inside a chunk (line breaks included) are replaced by one space so that an entry is
  one line. Chunks are not cut, filtered or reordered.
- Student prompt: D10 §2 student (Qwen3-14B, nf4 double-quant, greedy, thinking off, max_new
  1024, batch size 1, instruction MD5 f3b58d766fe3ec2573ff4f24761cf0c9), layout B: instruction,
  facts block with the 5 entries, Urdu question. The block must equal the sealed retrieval
  record of that qid (fatal).
- Title share (reported in the readout only): gold titles = the titles of the row's evidence
  ids (d3_oracle_retrieval.required_titles). The D8 freeze §3 rule (docs/
  EFBPT_D8_REACHABLE_CEILING_FREEZE.md, MD5 d95438fac2cb097d38cccdf5db6a3d35) maps each gold
  title to a corpus title or to ABSENT. A gold title is found if its corpus title equals the
  title of at least one of the 5 chunks. Reported per arm and set: found / all gold titles,
  found / gold titles present in the corpus, and the share of rows with at least one found.
  The title map is written by the retrieval job and sealed; it holds no model output.

### 7.4 RO_eval and the other students

- RO_eval: the D10 §2 student, layout A, plus one line directly after the Urdu question:
  "سوال کا انگریزی ترجمہ: {R1 reading}", built with the d11_ablate_ro constants (A3.6). In TEST-5
  the five DEV200 prompts must be byte-identical to the prompts in the sealed DEV200 RO file
  (fatal); whether the five generations are byte-identical is reported.
- Other students, weights on the cluster (checked 2026-11-12, all shards present):
  Llama-3.1-8B-Instruct = downloaded_models/LLM-Research/Meta-Llama-3___1-8B-Instruct;
  Gemma-2-9B-it = downloaded_models/AI-ModelScope/gemma-2-9b-it;
  Qwen3-8B = downloaded_models/Qwen/Qwen3-8B.
- Their user message is the Qwen3-14B user message of the same row: S_A = the layout-A text of
  A_r2_eval, S_TR = the layout-B text of TR with the sealed R1 teacher facts. On eval458 the
  user text must equal the user turn of the sealed A_r2_eval / TR prompt of that row (fatal, CPU).
- System text = SYSTEM_MESSAGE. Llama-3.1-8B and Qwen3-8B: system role. Gemma-2-9B: user turn =
  SYSTEM_MESSAGE + blank line + user message. Each tokenizer's own chat template with the
  generation prompt added; Qwen3-8B with enable_thinking=False.
- Loading and decoding as the D10 §2 student: 4-bit nf4, double quantization, bf16 compute;
  greedy (do_sample=False); max_new_tokens 1024; batch size 1; generation stops at the
  tokenizer's end-of-sequence token or the template's end-of-turn token. A row that reaches
  1024 new tokens is marked truncated.
- C2 in practice: in TEST-5, rows 0-4 of DEV200 are generated twice in both layouts (S_A and
  S_TR, DEV200 R1 teacher facts) and each pair must be byte-identical. Every full job of that
  student first repeats the five S_TR rows and compares them with the TEST-5 file (reported in
  the seal, not fatal).

### 7.5 Leak guards (C3)

Needles = d11_common.needles_for of the set: question_en of every row, gold facts, evidence ids;
and every reading string where stated.
- Arms whose input is the Urdu question (T_direct_F_eval, API_direct_ur_eval, S_A arms, the
  RAG_ur query): no question_en, no gold fact, no evidence id and no reading string anywhere in
  the prompt. Fatal.
- Arms whose input is a reading (T_tr_R1, T_tr_L1, API_tr_R1, RO_eval, the RAG_en query): D11
  Amendment 1 A1.1 (a)-(c). (a) The prompt with the reading slot blanked has no needle. (b) The
  reading has no other row's question_en, no gold fact and no evidence id; a reading equal to its
  own row's question_en is counted and reported. (c) The reading equals the sealed file. Fatal.
- S_TR arms: as D11 arm TR, with Amendment 6: a non-specific evidence-id needle (fewer than 3
  letters) inside a teacher fact is not fatal when the facts equal the sealed teacher output and
  the teacher prompt passed its guard; it is logged and listed. Gold-fact substrings inside
  teacher facts are reported (V6).
- RAG student prompts: outside the retrieved block every needle and every reading string is
  fatal. Inside the retrieved block, question_en of any row and any reading string are fatal.
  An evidence-id or gold-fact string inside retrieved Wikipedia text is logged and listed in the
  readout, not fatal, when the block equals the sealed retrieval record and the query passed its
  own guard: the retriever sees only the query and the public corpus, so such a hit is a string
  collision (the Amendment 6 reading, applied to retrieved text). Gold titles in retrieved text
  are the measured title share, not a leak.

### 7.6 Seals, scoring, outputs

- Outputs: outputs/efbpt/d13/<group>/<set>/ with <group> = large_alone, rag, ablation,
  transfer. TEST-5 outputs go to outputs/efbpt/d13/_test/ and regression outputs to
  outputs/efbpt/d13/_regress/; they are never results.
- Per arm and set: one row file, one START file, one SEAL file (row count, MD5, script and
  contract MD5s, job id). A seal is written only when every row is present; a sealed file is
  never rewritten. Each row stores the full prompt, the raw output, seconds and generated
  tokens. Row files carry no pred, correct or gold field (D11 A3.3). No job computes or prints
  accuracy on either set.
- Scoring is in two stages. Stage 1: Groups 1-3 on both sets (family F1 and the descriptive
  comparisons) are scored when every Group 1-3 arm is sealed. Stage 2: Group 4 (family F2) is
  scored when all six Group 4 arms are sealed. The readout script is committed before stage 1
  and is not changed between the stages.
- Readout: one script, committed before it is run on eval458 and tested first on the sealed
  DEV200 arms (it must reproduce the E0 numbers of the table in §1). It writes
  docs/handoff/D13_READOUT.md: per-arm table (E0 and E3, unparsed, predicted-yes rate, accuracy
  on gold-yes and gold-no, balanced accuracy), every comparison with L1-L5, raw and Holm p, the
  §4 wording for each test, predictions hit or missed, retrieval title share, time and tokens
  per row.

### 7.7 Jobs

SLURM partition q_intel_share_L20, conda env urbench_eval (conda.sh sourced directly), account
GPU limit 2, student answer pass at batch size 1; the retrieval job asks for at least 60 GB of
memory. Order: CPU preflight; TEST-5 (DEV200 rows 0-4, every arm, all checks); then the full
runs: the two API arms first, then the teacher arms, then RAG and RO_eval, then the other
students. Job order affects no number.
