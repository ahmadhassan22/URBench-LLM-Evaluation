# EFBPT D11 RTG — AMENDMENT 3 (eval458 running; nothing scored)
A3.1 V1 on eval458 protects the comparison being claimed, pair by pair.
 Primary: |unparsed(TR) - unparsed(A_r2_eval)| <= 5.0 pp -> the gate is read
 on all 458 rows. If > 5.0 pp -> the same gate (>= +5.0 pp and exact McNemar
 p < 0.05) is read on the rows parsed in both arms, and the all-rows numbers
 are reported beside it. Control: the unparsed rate of TR_wrong_eval is
 reported; if it differs from A_r2_eval by > 5.0 pp the control comparison is
 also reported on rows parsed in both arms; it never changes the primary.
 Secondary: the primary rule applied to TR_L1_eval vs A_r2_eval. The
 parsed-in-both analysis is reported in every case as a sensitivity line.
 Reason: on DEV200 the wrong-facts arm had the highest unparsed rate (5.5 %
 vs TR 1.5 %); a control arm must not be able to void the held-out test. This
 supersedes any V1 label printed by the merge script; if the script must be
 changed to apply A3.1, only the merge/summary code is touched and only after
 every eval458 student arm is sealed.
A3.2 Secondary test = exact two-sided McNemar p < 0.05 with a positive delta,
 read only if the primary passes; no pp threshold.
A3.3 Per-arm eval458 files carry no pred / correct / gold fields; the merge
 writes the scored copies, the summary and S_SEAL.
A3.4 Disclosures. (i) Before submission a CPU dry run with a stub in place of
 the model scored random placeholder strings against eval458 gold in a
 scratch directory; no model output existed. (ii) The preflight leak guard
 printed reading descriptors on eval458 (readings equal to gold question_en:
 R1 57, R3 39; readings containing a required title: R1 294, R3 286). Neither
 carries information about any arm's answers.
A3.5 The account GPU limit is 2 (AssocGrpGRES); job order follows from it and
 affects no number (batch size 1; regression 5/5 on another node).
A3.6 DEV200 baseline ablation, descriptive, not gated, no eval458 use.
 RO = frozen student, layout A, plus one line directly after the Urdu
 question: "سوال کا انگریزی ترجمہ: {R1 reading}". RO_gold = the same with the
 gold question_en (oracle translate-test ceiling). No facts in either. V2'
 probe (5 A_r2 rows byte-identical) fatal. Exact McNemar against the sealed
 arms: RO vs A_r2, TR vs RO, RO_gold vs A_r2, TO1 vs RO_gold. Predictions:
 RO - A_r2 in [0, +6]; RO_gold - A_r2 in [+1, +8]; TR - RO >= +2;
 TO1 - RO_gold >= +5. Unparsed, truncation and Devanagari-answer counts
 reported.
