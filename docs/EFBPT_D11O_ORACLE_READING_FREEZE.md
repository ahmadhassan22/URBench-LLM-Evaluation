# EFBPT D11-O — ORACLE-READING CEILING DIAGNOSTIC (FREEZE)
Status: FROZEN before any run. Diagnostic, not a method: its numbers decide
the design of D11, they are never claimed as a method result.

## 1. Question
Teacher facts from the Urdu question give +5.0 (v1) / +4.5 (v2) pp; gold
facts gave +20.5 pp (D1, batch size 8). How much of the gap is the teacher's
READING of Urdu, and how much is its KNOWLEDGE? Oracle reading = the teacher
gets the gold English question instead of the Urdu one; the student still
reads only the Urdu question.

## 2. Arms (student Qwen3-14B exactly as D10: nf4, greedy, thinking off,
batch size 1, frozen instruction f3b58d76…, build_user layouts; n = 200)
| arm | student input | facts source |
|---|---|---|
| A_r2 (floor) | Urdu question | none — the SEALED D10 file
  outputs/efbpt/d10/student/d10_armA_r2.jsonl, MD5 7ccec942c8f0eee08156d258baaf806d,
  reused under V2' (below); not re-generated |
| GF (ceiling) | Urdu question | gold urbench_facts, layout B (= D1 arm B byte for byte) |
| TO1 | Urdu question | teacher facts, v1 prompt, teacher input = gold question_en |
| TO2 | Urdu question | teacher facts, v2 prompt, teacher input = gold question_en |
Reference arms from sealed files, not re-run: TF_v1 (d10_armTF.jsonl,
c54b7553baa5ffa77204bed1dab37250), TF_v2 (d10v2 d10_armTF.jsonl,
1a3cdba1f26ccca4b010a00676197f0a).

## 3. Teacher (Llama-3.1-70B AWQ, vLLM, §2 of the D10 freeze unchanged)
TO1 prompt = D10 T1 SYSTEM/USER with exactly two edits: "a question written
in Urdu" -> "a question written in English"; "Question (Urdu): {question_ur}"
-> "Question: {question_en}". TO2 prompt = D10v2 T1-v2 SYSTEM/USER with the
same two edits (the "write what the question asks in English" step stays).
Parse rules, T2 verdict filter, max_tokens (512 / 768), A4/A5 repairs: as in
D10 / D10v2. Leak guard on teacher prompts: gold facts and evidence ids FATAL;
question_en is the intended input in this diagnostic and is NOT a needle for
the teacher prompts. Outputs: outputs/efbpt/d11o/teacher_v1/, teacher_v2/
(t_facts.jsonl, T_START, T_SEAL).

## 4. Student
Leak guard on student prompts: question_en and evidence ids FATAL in every
arm; gold urbench_facts FATAL in TO1/TO2 (the teacher never saw them),
PERMITTED only in GF (it is the ceiling arm; recorded as such). V8 prompt
content check on GF, TO1, TO2. V2' determinism probe: 20 A_r2 prompts
generated once at batch size 1 must be byte-identical to the 20 sealed A_r2
generations (not merely identical to each other); PASS required to reuse the
sealed A_r2 as the floor; otherwise A_r2 is re-generated in this job.
Outputs: outputs/efbpt/d11o/student/ (one jsonl per arm, d11o_summary.json,
S_START, S_SEAL).

## 5. Readings (paired exact McNemar on the same 200 qids; descriptive —
this is a diagnostic, there is no GO gate)
O1 GF − A_r2: the knowledge ceiling at batch size 1.
O2 TO1 − A_r2 and TO2 − A_r2: the perfect-reading teacher gain.
O3 TO1 − TF_v1 and TO2 − TF_v2 (same prompt, Urdu vs English input): the
   reading effect itself.
O4 Reading share = (TO − TF) / (GF − TF) per version.
O5 R4 entity reading for TO1/TO2 (gold titles in parsed entities; expect far
   above the Urdu-input 34–36%); facts per row; parse and repair counts;
   V6 verbatim gold-fact substrings in TO prompts (reported).
O6 Splits of TO vs A_r2 by gold label (the "no"-bias check).

## 6. Pre-declared design decision for D11 (the method)
- max(O2) ≥ +9.0 pp: reading is the main wall -> D11 = reading-robust
  injection (multi-reading agreement gate; title-grounded entity reading).
- max(O2) ≤ +6.0 pp: the teacher's knowledge is the main wall -> D11 targets
  fact relevance (decomposition-guided retrieval for the teacher, or a second
  teacher for agreement); the reading route is not pursued.
- between: both levers, agreement gate first.
In every case eval458 stays reserved for the final D11 method, run once.

## 7. Rules
TEST-5 for each of the three jobs before FULL (chained afterok). Teacher
FULL limit 02:00:00 each; student FULL 08:00:00 (3 arms × ~1.5 h + probe).
Readout from sealed files only. Freeze MD5 recorded in every START/SEAL.
