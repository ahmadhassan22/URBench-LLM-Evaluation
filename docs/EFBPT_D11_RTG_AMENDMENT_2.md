# EFBPT D11 RTG — AMENDMENT 2 (before eval458 is opened)
A2.1 Hand check, 22 TR-vs-A_r2 loss rows (readout §8 numbering), primary
 cause: reader error 4 (rows 8, 10, 11, 12); Urdu source item fault 3 (4, 13,
 19 — the Urdu question itself changed or dropped the entity; the reading is
 faithful); teacher wrong or misleading fact 3 (1, 9, 22); generic facts
 without the decisive hop 4 (3, 5, 6, 14); student reasoning with relevant
 facts 4 (7, 15, 16, 21); scoring / polarity 4 (2 negated Urdu question, TR
 content agrees with gold; 17, 18 TR opens with ہاں = gold, the last-mention
 rule returns no; 20 both arms open alike, last mention differs). Notes only;
 the frozen scorer is unchanged.
A2.2 eval458 arms. PRIMARY (gate, unchanged): TR (R1) vs A_r2_eval, >= +5.0 pp
 and exact two-sided McNemar p < 0.05. CONTROL (unchanged): TR_wrong_eval.
 SECONDARY, added now: TR_L1_eval = R3 (L1_llama70b_awq) sealed eval458
 readings -> same teacher -> layout B; the API-free variant. Fixed-sequence
 test: TR_L1_eval vs A_r2_eval is tested at 0.05 only if the primary passes;
 otherwise it is descriptive. No other arm.
A2.3 Operational. One SLURM job per student arm (batch size 1), one teacher
 job per reader (R1, R3); per-arm seals; a later CPU merge verifies all seals
 and writes the summary. Before any eval458 job: REGRESSION on DEV200 through
 the per-arm path, arm TR, rows 0-4; generations must be byte-identical to
 the sealed d11_armTR.jsonl (MD5 bab987cc492012f4e32dbcd348be9668); fatal
 otherwise; every eval458 job depends afterok on it.
A2.4 No TEST on eval458 rows. CPU preflight only: 458 rows, 221 yes / 237 no,
 overlap 0 with DEV200 and RDEV300, reading-file MD5 = R_SEAL entry, leak
 guards (Amendment 1 A1.1) on all prompts. No job computes or prints
 accuracy; the merge runs only after A_r2_eval, TR and TR_wrong_eval are
 sealed (TR_L1_eval is merged when sealed).
A2.5 Crash rule. A crashed arm resumes from its last completed row; completed
 rows are never regenerated; nothing is scored before the merge.
A2.6 Predictions (eval458). A_r2_eval accuracy 55-63. TR - A_r2_eval +5 to
 +10 (DEV200 class rates transferred to 221/237: +7.7). TR_wrong_eval -
 A_r2_eval <= -3. TR_L1_eval - A_r2_eval +4 to +9. Gain on gold-no rows >
 gain on gold-yes rows.
A2.7 Reading of outcomes. Primary passes -> RTG confirmed on held-out data.
 Positive but gate missed -> reported as not confirmed; pooled DEV200 +
 eval458 only as description. No second eval458 run in any case.
