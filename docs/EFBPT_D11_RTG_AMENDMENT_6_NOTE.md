# Amendment 6 — note (before any eval458 arm is scored)
N1 Preflight (0988511): 916 TR / TR_wrong_eval prompts, 0 fatal hits. The one
 non-specific hit is '8-1' inside "0.8-1.0 g/cm3" (row 98; the evidence id
 belongs to row 10; the reading does not contain it). Five teacher facts
 equal a gold fact string (V6, reported).
N2 The A5.2 readout script stays byte-identical (MD5 8fb00396...). The
 listing required by A6.3 / A6.4 (non-specific hits, V6 hits, per-arm script
 MD5s, both regression records, teacher parse counts, empty-facts rows,
 amendment MD5s) is produced by a separate script,
 d11_eval458_provenance.py, into docs/handoff/D11_EVAL458_PROVENANCE.md. It
 computes no statistic on answers.
N3 Crash rule for A_r2_eval (115529), which started under script a733caed:
 if it stops before sealing, it is resumed with that same script version
 (taken from git at 43b61ca into a separate file), from its last completed
 row. This arm has no facts, so the guard change does not concern it.
N4 Expectations under the validated scorer E3, from DEV200 class rates
 transferred to 221 yes / 237 no (descriptive, not a gate): A_r2_eval 55.8,
 TR 71.3 (+15.5), TR_L1_eval 69.7 (+13.9), TR_wrong_eval 48.7. Predictions:
 TR - A_r2_eval under E3 in [+9, +20]; TR_L1_eval - A_r2_eval under E3 in
 [+7, +19]; under E3 TR is not below A_r2_eval on gold-yes rows and is at
 least +10 pp above it on gold-no rows; TR's predicted-yes rate is closer to
 the gold rate (48.3 %) than A_r2_eval's.
N5 Scorer audit, naming: the PRIMARY labels were produced by adjudicator A
 with a written rubric (Amendment 4 note); the two API models are
 adjudicators B and C. Agreement A-B 96.2 %, A-C 95.2 %, B-C 97.1 %; Fleiss'
 kappa 0.933; E3 is the top extractor under the PRIMARY, ALT and majority
 labels. An independent labelling of a 30-item subset by the author (native
 Urdu speaker) is added when available (N6).
N6 Author sheet: 30 items = the 6 items where adjudicators disagree (28, 40,
 41, 78, 95, 101) + 24 drawn with random.Random(2026) from the other 99;
 shuffled with the same seed; no labels shown. The author's labels are
 compared with PRIMARY and with the majority (agreement and kappa on the 24
 random items; the 6 listed one by one). They select nothing.
