# EFBPT D11 RTG — AMENDMENT 4 (scorer audit; eval458 still unscored)
A4.1 Finding (DEV200 supplement 2f33c49). The frozen extractor and a
 first-mention variant label 27-39 rows per arm differently; TR vs A_r2 is
 +8.5 pp (p 0.040) under the frozen rule and +17.0 pp (p 0.0002) under the
 variant. The frozen rule takes the last ہاں/نہیں substring, so a negation
 inside the explanation, or a word such as یہاں / وہاں / جہاں, can flip the
 label. The frozen extractor stays the pre-registered scorer of every gate.
A4.2 Audit (DEV200 only). Items: for arms A_r2 and TR, every row where the
 frozen label and the first-mention label differ, plus 20 rows per arm drawn
 with random.Random(4242) from the rows where they agree. Each item shows the
 Urdu question and the generation (first 300 and last 250 characters; the
 whole text if shorter). Items are shuffled with seed 4242; arm, gold, qid
 and all extractor labels are hidden and kept in a key file. The adjudicator
 labels the verdict the text states: yes / no / none (no verdict, or both).
A4.3 Candidate extractors, fixed now. E0 frozen. E1 first-mention (S1). E2 =
 E1 with whole-word matching (ہاں and نہیں as separate words; یہاں, وہاں,
 جہاں, کہاں and other longer words do not count). E3 = opening verdict: a
 whole-word ہاں / نہیں within the first 12 words of the answer segment; else
 the last whole-word mention; else unparsed. For each: agreement with the
 adjudicated labels in both strata and the stratified estimate per arm.
 VALIDATED scorer = highest stratified agreement (mean of the two arms);
 ties -> E2, then E3, then E1.
A4.4 Use. DEV200: all sealed arms re-scored with the validated scorer,
 descriptive. eval458: next to the frozen-scorer gate, two declared lines are
 reported for every comparison: the validated scorer, and (A3.1) rows parsed
 in both arms. They do not change the gate. Reading of outcomes, fixed now:
 frozen gate passes -> confirmed. Frozen gate missed but the validated scorer
 gives >= +5.0 pp and p < 0.05 -> reported as "confirmed under the validated
 scorer; pre-registered gate missed", with this audit as the reason. Neither
 -> not confirmed.
