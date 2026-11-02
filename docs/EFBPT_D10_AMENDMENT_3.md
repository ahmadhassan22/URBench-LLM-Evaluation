# EFBPT D10 — AMENDMENT 3: the one fix iteration (freeze §7, option iii)

Basis (docs/handoff/D10_LOSS_WIN_ROWS.md): of the 28 TF losses, 13 are
teacher misreadings of a question term (facts about the wrong or an invented
entity), 7 are right entities with the deciding bridging fact missing, 7 are
student errors with sufficient facts, 1 is a false teacher fact. The 38 wins
almost all carry the specific bridging fact. The teacher returned exactly 6
facts on 199/199 parsed rows (padding).

A3.1 Fix = freeze §7 (iii) decomposition. New T1 prompt (T1-v2), verbatim.
SYSTEM:
"You are a careful research assistant with broad world knowledge. You read a
question written in Urdu. You do not answer it. You first write what the
question asks in English, then the smaller questions whose answers would
decide it, then short, true, self-contained English facts that answer those
smaller questions. You only write facts you are confident are true, and only
about terms you have identified with confidence; if you cannot identify a
term, you list it as unknown and write nothing about it. You never state or
hint at the answer to the question, you never compare the entities with each
other, and you never use the words yes, no, therefore, thus, hence, or answer."
USER:
"Question (Urdu): {question_ur}

Return only a JSON object with these keys, in this order:
"question_en": the question in English, one sentence, your best reading;
"unknown_terms": a list of terms in the question you could not identify with
confidence (empty list if none);
"sub_questions": a list of 2 to 4 short English questions whose answers
decide the question;
"entities": a list of the entities and concepts you identified, written as
their English Wikipedia article titles;
"facts": a list of at most 6 short English facts (one sentence each), each
answering one of the sub-questions, only about entities you identified with
confidence.
No other text."
max_tokens 768. Parse rule as before; all five keys must be present with the
right types, else parse_ok=false and empty lists. T2 verdict filter unchanged,
applied to facts only. ONLY the post-T2 facts are injected; question_en,
unknown_terms and sub_questions are stored for audit and never enter any
student prompt (leak guard unchanged: the gold question_en remains a fatal
needle in every prompt).

A3.2 Jobs. Teacher v2 job: T1-v2 only, plus derangement (seed 1010; must
equal the sealed v1 derangement.json MD5 8f00e2f9e54f27138964a0577a8eff52).
T-direct and T-direct-F are NOT re-run; the student v2 job reads the sealed
v1 files outputs/efbpt/d10/teacher/t_direct.jsonl (MD5
bac96cd9a9773b5c5e719e982ecfe499) and t_direct_f.jsonl (MD5
40ebfe643e244c5a1a3fff729f23462c) and aborts on mismatch. Student v2 arms,
layouts, batch size 1, V1–V9, R1–R4, descriptives: unchanged from the freeze
and Amendments 1–2. Outputs under outputs/efbpt/d10v2/{teacher,student}/;
TEST under outputs/efbpt/d10v2/_test/. FULL limits: teacher 02:00:00,
student 12:00:00. TEST-5 before FULL for both.

A3.3 New descriptives (never gates): unknown_terms per row; rows with empty
facts; facts-per-row distribution; share of gold titles whose normalised
string occurs in question_en (restatement-level entity reading).

A3.4 V1 clarification (pre-declared before any v2 number): if V1 fails only
because TF_wrong's unparsed rate moves the spread above 5.0pp while
|unparsed(TF) − unparsed(A_r2)| ≤ 5.0pp, R1 is read as valid and all
TF_wrong readings are reported as VOID.

A3.5 Decision after the v2 FULL run (pre-declared):
- R1 GO (≥ +8.0pp, p < 0.05): write the eval458 freeze; one run.
- R1 CONDITIONAL (≥ +5.0pp): eval458 is run once as the final pre-registered
  test, gate ≥ +5.0pp and p < 0.05 at n = 458, with whichever teacher variant
  (v1 or v2) has the higher DEV200 Δ.
- R1 < +5.0pp: the teacher-facts route is closed; freeze §7 last branch.
No further DEV200 iteration on this route in either case.
