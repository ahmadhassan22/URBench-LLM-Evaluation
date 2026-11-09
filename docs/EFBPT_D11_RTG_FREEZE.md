# EFBPT D11 — READ-THEN-GROUND (RTG): EXPLICIT READING + TEACHER FACTS (FREEZE)
Status: FROZEN before any D11 answer run.
Method: Urdu question → frozen reader (D11-R selected) → English question →
v1 teacher facts (Llama-3.1-70B-AWQ, D11-O v1 prompt, D10 §2 engine,
Amendments 4/5 repairs) → frozen student (Qwen3-14B nf4, D10 §2, batch 1)
answers in Urdu with the facts under حقائق (layout B).
References, sealed, not re-run: A_r2 60.0 (D10 file MD5 7ccec942…),
GF 76.0 and TO1 74.0 (D11-O readout 9e06403).
## 1. Item set and student
DEV200 (seed 4242, floor 56.50). Student exactly D10 §2: instruction MD5
f3b58d766fe3ec2573ff4f24761cf0c9, greedy, thinking off, max_new 1024, batch
size 1, d1_score_dual scorers, extractor Rule 2, exact two-sided McNemar.
## 2. Readers (inputs = sealed D11-R DEV200 reading files)
R1 api_qwen3.8-max (SELECTED) readings_dev200.jsonl MD5 1f5b2db3c1403f63b3525a1562edd176
R2 api_deepseek-v4-pro (RUNNER-UP) MD5 cd77944e45a98292777382b9d9f04aef
R3 L1_llama70b_awq (best local; API-free variant) MD5 0d06ec1a926fe3c186c125ad723139cf
## 3. Teacher
d11o_teacher.py v1 path; English input = the reading for that qid, nothing
else from the row. Same sampling, max_tokens, parse repairs. Fatal leak
guards: gold question_en substring, evidence ids or titles in any prompt.
V6 gold-fact restatements reported.
## 4. Arms (student, DEV200)
TR = R1 facts, layout B.  TR2 = R2 facts, layout B.  TR_L1 = R3 facts,
layout B.  TR_wrong = TR facts deranged, seed 1010, must equal the D10
derangement (MD5 8f00e2f9…), layout B.  TR_noq = TR facts, layout G.
A_r2 reused from D10; V2' probe: 5 rows regenerated, byte-identical.
Descriptive reference, not gated: API_direct_ur = qwen3.8-max,
enable_thinking=false, temperature 0, the Urdu question alone, instruction
"صرف ہاں یا نہیں لکھیں"; 200 calls; accuracy = reader model answering alone.
## 5. Checks (fatal unless stated)
V1 unparsed spread ≤5 pp across arms. V2' byte-identical. V3 leak guards §3.
V4 readings used equal the sealed files (MD5), one line each; rows whose
reading contains yes/no/ہاں/نہیں reported. V5 TR_wrong − A_r2 ≤ −3 expected
(A2.1 sign rule). V6 reported. V8 layout check with the <|im_end|> cut (A1.6).
## 6. Pre-registered predictions
TR +9 to +13 vs A_r2 (between D10 +5.0 and TO1 +14.0; reader M1 91.5 vs
gold). TR2 within ±2 of TR. TR_L1 +7 to +11; if TR_L1 ≥ TR − 2 the
mechanism is the explicit reading stage, not the API model. TR_noq ≤ TR.
TR_wrong ≤ A_r2 − 3.
## 7. Decision (DEV200, TR vs A_r2, exact McNemar)
GO ≥ +8.0 and p < 0.05 → eval458 once (§8). CONDITIONAL +5.0 to +7.9 →
eval458 once with the DEV200-best of {TR, TR2, TR_L1}. < +5.0 → RTG closed;
fallback D12 = agreement-gated reading. eval458 arm = DEV200-best of
{TR, TR2, TR_L1}; tie → TR. Decision recorded in docs/D11_DECISION.md.
## 8. eval458 protocol (declared now; runs only if docs/D11_GO_EVAL458.md
exists). Items: is_eval rows (458; 221 yes / 237 no; floor 51.75; overlap 0).
Arms: A_r2_eval (layout A), the chosen TR arm, TR_wrong_eval. Readings = that
reader's sealed D11-R eval458 file, opened only at this step. One run. Gate
+5.0 and p < 0.05. All loaders take --set dev200|eval458; the 200-row,
D6-71 and pinned-file asserts become set-conditional.
## 9. Runs
TEST-5 (5 rows, all arms, teacher + student, all checks) → FULL, chained
--dependency=afterok. Teacher job 03:00:00; student job 16:00:00. Outputs
outputs/efbpt/d11/{teacher,student}/<set>/; R_START/R_SEAL with this MD5.
Readout from sealed files only: docs/handoff/D11_FULL_READOUT.md.
