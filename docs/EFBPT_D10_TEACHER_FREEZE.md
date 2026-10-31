# EFBPT D10 — TEACHER FACT INJECTION (FREEZE)

Status: FROZEN. No number from any D10 run may be read before this file and the
D10 script are committed with their MD5s recorded. Changes before FULL go in a
numbered amendment file, committed before the FULL job. No change after FULL.

## 1. Question and hypothesis
Correct English facts raise Qwen3-14B's Urdu StrategyQA accuracy (D1 B−A +20.5pp;
D8 R1 +14.7pp). Every automatic fact source so far failed: retrieval from named
pages (D8 R2) and the 14B's own knowledge (D8 R3, which misreads Urdu entities).
Hypothesis H1: a 70B teacher reads Urdu entities correctly where the 14B does not;
its short English facts, given to the 14B, raise accuracy by at least +8pp.
Null: facts from the teacher do not beat the no-facts floor by +8pp at p<0.05.

## 2. Fixed inputs
- Items: DEV200 (data/strategyqa_official/dev200_seed4242.jsonl, seed 4242, n=200). Floor 56.50%.
- eval458 (data/sdfr_splits/strategyqa_eval.jsonl) is NOT touched by D10. Overlap with DEV200: 0.
- Student: Qwen3-14B, HF transformers, NF4 double-quant, bf16 compute, greedy,
  thinking off, max_new_tokens 1024, BATCH SIZE 1. Frozen Urdu instruction
  prompts/efbpt/plan_a_instruction_ur.txt, MD5 f3b58d766fe3ec2573ff4f24761cf0c9. Prompt construction
  by d1_eval_arms.build_user, unchanged. Scorer d1_score_dual (primary
  Perso-Arabic, secondary Devanagari). McNemar from d5_balanced_extract.mcnemar
  (exact two-sided binomial on discordant pairs). Unparsed = wrong.
- Teacher: Meta-Llama-3.1-70B-Instruct-AWQ-INT4 at /mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4, vLLM
  0.13.0, 1 GPU (NVIDIA L20, 46068 MiB), greedy (temperature 0), chat template of
  the model, max_model_len 4096 (as the old baseline), tensor_parallel_size=1, quantization="awq", gpu_memory_utilization=0.90, max_num_seqs=1, enforce_eager=True, trust_remote_code=True, disable_log_stats=True.
- No question_en, gold facts, or evidence ids enter any prompt in either job.

## 3. Job T — teacher (vLLM)
T1. T-facts. For each DEV200 row the teacher gets:
  SYSTEM (verbatim):
  "You are a careful research assistant with broad world knowledge. You read a
  question written in Urdu. You do not answer it. You identify the real-world
  entities and concepts the question is about, and you write short, true,
  self-contained English facts about them that someone would need in order to
  decide the question. You only write facts you are confident are true. You
  never state or hint at the answer to the question, and you never use the
  words yes, no, therefore, thus, hence, or answer."
  USER (verbatim, {question_ur} = the Urdu question):
  "Question (Urdu): {question_ur}

  Return only a JSON object with two keys:
  "entities": a list of the entities and concepts in the question, written as
  their English Wikipedia article titles;
  "facts": a list of at most 6 short English facts (one sentence each) about
  those entities that are needed to decide the question.
  No other text."
  max_tokens 512. Output per row: qid, raw text, parsed entities, parsed facts,
  parse_ok. Rows that fail JSON parsing keep parse_ok=false and an empty fact
  list; they are never dropped.
T2. Verdict filter (pre-declared, applied once, counted): a fact line is removed
  if it matches, case-insensitive,
  \b(therefore|thus|hence|the answer|this question|the question|in conclusion)\b
  or starts with "Yes" or "No" followed by a comma, period, or colon. Removed
  lines are stored in a separate field and counted (V5). Facts are truncated to
  the first 6 after filtering.
T3. T-direct. The teacher gets the student's exact no-facts prompt (frozen
  instruction + Urdu question, built by build_user with arm "A", and
  the same SYSTEM_MESSAGE from efbpt_eval_dev200 as the system turn), through its chat
  template, max_tokens 1024, greedy. Scored with d1_score_dual. This is the
  "teacher alone" baseline. If T-direct unparsed > 10%, R2 is also reported on
  parsed rows.
T4. Derangement map for TF_wrong: a permutation of the 200 qids with no fixed
  point, seed 1010, written to file and sealed.
T5. Leak guard on every teacher prompt: question_en, gold facts, evidence ids —
  FATAL (abort, no output).
T6. Outputs under outputs/efbpt/d10/teacher/: t_facts.jsonl, t_direct.jsonl,
  derangement.json, T_START.json, T_SEAL.json (MD5, bytes, lines of every file,
  script MD5, this freeze's MD5). Every row stores its full prompt text. Sealed
  before Job S starts.

## 4. Job S — student (HF, batch size 1)
Preflight: verify every MD5 in T_SEAL.json; abort on mismatch. Verify the
instruction MD5. Determinism probe V2 (below). Record all in S_START.json.
Arms, n=200 each, same job, same order, batch size 1:
| Arm | Question | Facts block | Purpose |
|---|---|---|---|
| A_r2 | Urdu | none | floor |
| TF | Urdu | teacher facts of the same qid (after T2) | the method |
| TF_wrong | Urdu | teacher facts of derangement(qid) | relevance control (as D1 arm C) |
| TF_noq | none | teacher facts of the same qid | leak probe (as D1 arm G) |
Layouts (build_user, unchanged): A_r2 = arm "A". TF and TF_wrong = arm "B"
layout with row["facts"] REPLACED by the post-T2 teacher facts of the intended
qid, exactly as D8 pass 2 did: build_user(instruction, dict(row, facts=list(facts)),
None, "B"). TF_noq = arm "G" layout with row["facts"] replaced the same way.
Gold facts never remain in row["facts"] of any D10 prompt.
TF and TF_wrong rows with an empty fact list get the arm "A" prompt (identical to
A_r2); TF_noq rows with an empty fact list get the arm "G" layout with an empty
block. All are kept, scored, and counted separately, as in D6.
Leak guard on every student prompt: question_en and evidence ids FATAL; verbatim
gold-fact substrings REPORTED (V6), not fatal (as D8 Amendment 2).
Outputs under outputs/efbpt/d10/student/: one jsonl per arm, d10_summary.json,
S_START.json, S_SEAL.json.

## 5. Validity (checked before any reading)
V1 Unparsed spread over A_r2, TF, TF_wrong ≤ 5.0pp (AMENDMENT 5D). TF_noq is
   excluded from V1, as arm G was in D1, and its unparsed rate is reported.
V2 Determinism probe: 20 A_r2 prompts generated twice at batch size 1 in the
   preflight; 20/20 identical strings = PASS. Otherwise DETERMINISM FLAG is
   recorded and the run continues (readings are paired inside one job). In TEST
   mode the probe uses 5 prompts.
V3 Leak guards fired: none fatal in either job.
V4 Teacher JSON parse rate ≥ 95% (≥190/200). Below that: FORMAT FLAG; readings
   still made; the fix iteration (§7) may add 2 format examples drawn from the
   training pool only.
V5 Verdict-filter count reported. If > 100 lines removed: TEACHER-ANSWERING FLAG.
V6 Gold-fact substring count in TF prompts reported.
V7 T_SEAL MD5s verified by Job S: PASS required.
V8 Prompt-content check: for every TF, TF_wrong and TF_noq prompt, the fact
   lines in the prompt equal the sealed post-T2 facts of the intended qid
   (own qid, or derangement(qid) for TF_wrong). Any mismatch is FATAL.
V9 Latin-script fraction of all injected teacher facts, reported (D8 V3 style;
   GEN was 0.9972).

## 6. Readings (all on the same 200 qids, paired, exact McNemar)
R1 PRIMARY: TF vs A_r2. GO if Δ ≥ +8.0pp and p < 0.05. CONDITIONAL if
   Δ ≥ +5.0pp. HARM if Δ ≤ −3.0pp. Else NO-GO.
R2 TF vs T-direct (McNemar, reported with both accuracies). Three pre-named
   outcomes: (a) TF ≥ T-direct: combination beats both parts; (b) A_r2+8 ≤ TF
   < T-direct: injection works, teacher is the ceiling → D11 distillation;
   (c) TF below gate: see §7.
R3 Controls: |TF_wrong − A_r2| ≤ 4.0pp expected; above → FORMAT-EFFECT FLAG.
   TF_noq overall accuracy < A_r2 expected; ≥ A_r2 → LEAK FLAG. TF_noq
   parsed-only accuracy is reported beside the gold-facts reference (D1 arm G:
   42.0% overall, 63.2% parsed-only).
R4 Mechanism: teacher entity-reading score, computed the D6 G4 way, title-level:
   numerator = gold evidence titles (required_titles of each row) that appear,
   after norm(), in the teacher's parsed "entities" list; denominator = total
   required titles. Computed (a) on the 71 qids of the D6 run, taken from the
   sealed D6 outputs, where the reference is Qwen3-14B D6 G4 = 42/200 = 21.0%;
   and (b) on all 200 DEV200 rows, descriptive. Teacher ≤ 30% on (a) ->
   MECHANISM FLAG (H1's reason is not supported even if R1 passes).
Descriptive, never gates: predicted-yes rate per arm (A_r 47%, GEN 26% in D8);
facts per question; parse failures; V5 count; TF wins/losses split by
parse_ok; a seed-8 sample of 30 TF facts written to a file for a hand check of
truth and relevance.

## 7. Decision tree (pre-declared)
- R1 GO and R2(a): write the eval458 freeze; one run on eval458, everything frozen.
- R1 GO and R2(b): write the D11 freeze (teacher facts on the training pool of
  1632 questions → QLoRA the student to write English facts then
  answer in Urdu; tested on DEV200 vs A_r and vs D8 GEN). eval458 once, at the end.
- R1 CONDITIONAL: exactly ONE fix iteration, chosen from this list and declared
  in an amendment before it runs: (i) two format examples from the training
  pool; (ii) teacher self-consistency filter (3 samples at T=0.7, keep facts
  present in ≥2); (iii) decomposition prompt (sub-questions before facts).
  Then one FULL re-run. No second iteration.
- R1 NO-GO or HARM: fact injection is closed as a family (retrieval, self,
  teacher all tested). Remaining path: training-based distillation of teacher
  Urdu answers; thesis contributions otherwise stand on the diagnosis.
- Any FORMAT-EFFECT FLAG: TF − TF_wrong is reported as the content effect, and
  the method claim is reduced to that difference.
- Any LEAK FLAG: TF is reported as teacher-answer transfer, not fact injection,
  whatever R1 says.

## 8. Rules
- TEST-5 (--limit 5) for Job T and Job S before FULL; TEST outputs are not
  readings and are deleted or archived under a _test/ folder.
- Job T FULL time limit 04:00:00; Job S FULL time limit 08:00:00.
- A readout document is written from sealed files only, numbers copied, not
  typed from memory.
- Carried forward, still open: D8 V5 hand check of 30 Xp facts.
