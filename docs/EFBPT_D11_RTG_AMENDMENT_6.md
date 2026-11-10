# EFBPT D11 RTG — AMENDMENT 6 (evidence-id guard: non-specific needles)
A6.1 Event. Teacher R1 on eval458 completed and sealed (458 rows). Jobs
 115530 (TR) and 115531 (TR_wrong_eval) stopped in preflight after 17 s and
 11 s, before the model loaded: the string '8-1', which is an evidence id of
 the eval458 item set, occurs in the teacher facts of row 98 (and, through
 the derangement, in TR_wrong_eval prompt 50). Nothing was generated.
A6.2 Reading. Evidence ids have the form <title>-<paragraph index>. '8-1'
 has no letters and is a substring of ordinary text (for example "2018-19").
 The teacher's only input is the sealed reading (V4), and its own prompt
 passed the guard, so a gold evidence id cannot reach its output. The hit is
 a string collision, not a leak.
A6.3 Rule. An evidence-id needle with fewer than 3 letters is NON-SPECIFIC.
 A non-specific hit is not fatal when (i) it lies inside a teacher fact,
 (ii) V8 holds for that prompt (its facts equal the sealed teacher output of
 the source qid) and (iii) the teacher prompt of that qid passed its guard.
 Every such hit is logged and listed in the readout: prompt row, needle, the
 rows that own the evidence id, the fact sentence, and whether the needle
 also occurs in the reading. All other needles stay fatal: question_en, gold
 facts as before, and evidence ids with 3 or more letters. The rule applies
 to TR, TR_wrong_eval and TR_L1_eval.
A6.4 Code. Only the guard changes; prompt building, generation and scoring
 are untouched. The A2.3 regression (DEV200, TR rows 0-4, byte-identical) is
 re-run with the changed script before any arm starts. A_r2_eval (115529)
 keeps the earlier script MD5; the merge lists both MD5s and both regression
 records.
A6.5 The sentence containing the needle is printed for the record; it
 carries no label information. No eval458 accuracy exists at this point.
