# EFBPT D10 — AMENDMENT 1 (before FULL; applies to freeze b3a822d8199034c7fe58d76cf5cc428f)

Trigger: teacher TEST-5, job 109050. T-direct returned 2/5 Urdu prose answers
with no ہاں/نہیں token (unparsed under the frozen extractor). The frozen Urdu
instruction asks for an answer but not for a yes/no word; Qwen3-14B produces
one by habit, Llama-3.1-70B often does not. A teacher-alone baseline that
loses rows to format is not a fair baseline.

A1.1 New teacher arm T-direct-F. Identical to T-direct (build_user arm "A",
SYSTEM_MESSAGE, greedy, max_tokens 1024), with ONE line appended to the user
message after the question, separated by a blank line:
  "آخر میں ایک الگ سطر پر لکھیں: حتمی جواب: ہاں یا حتمی جواب: نہیں"
  (At the end, on a separate line, write: Final answer: yes, or Final answer: no.)
The marker حتمی جواب must be byte-identical to the Rule-2 marker constant in
efbpt_eval_dev200.extract_answer (copy it from the code, do not retype it).
The line's MD5 is recorded in T_START.json. Output t_direct_f.jsonl, stored
with the full prompt, sealed in T_SEAL.json with the other files. T-direct
(exact student prompt) is kept unchanged.

A1.2 R2 (replaces freeze §6 R2). T-best = whichever of T-direct and
T-direct-F has the higher primary accuracy on the 200 rows. A tie goes to
T-direct (the exact student prompt). TF is compared
with T-best; outcomes (a), (b), (c) exactly as in the freeze with T-best in
place of T-direct. Both teacher-direct arms are reported in full (accuracy,
unparsed, pred-yes, McNemar vs TF). The parsed-only clause of freeze T3
applies to any teacher-direct arm whose unparsed rate exceeds 10%.

A1.3 The teacher TEST-5 is re-run after this amendment and the earlier TEST
outputs under outputs/efbpt/d10/_test/teacher/ are overwritten. The student
script's V7 required-file set gains t_direct_f.jsonl.

A1.4 Nothing else changes: T1 prompt, T2 filter, T4, T5, all student arms,
V1–V9, R1, R3, R4, §7, §8. Recorded for the readout, not acted on:
(i) on 2 of 5 TEST rows the teacher fabricated entities (SnapCap → 7-Eleven;
Dorothea Wendling → an invented American racing driver). This is the F1
mechanism that R4 measures; the T1 prompt is deliberately left unchanged so
that it is measured, not patched. (ii) The frozen Rule-2 substring match
scores "معلومات نہیں ہے" as "no"; known since D1, identical for all arms.

A1.5 Budget: Job T FULL gains 200 generations (~27 min at 8 s/row); the
04:00:00 limit is unchanged.

A1.6 V8 checker correction (rule unchanged). Student TEST job 109056
failed in the CPU preflight: in the arm-G layout the facts block ends
the user turn, so the chat template's <|im_end|> followed the last fact
line with no newline and the checker read it as part of the fact. The
prompt was correct. The checker now cuts the user text at <|im_end|>
before reading the '- ' lines. No generation had run; no file was written.
