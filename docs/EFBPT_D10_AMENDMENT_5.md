# EFBPT D10 — AMENDMENT 5 (before the v2 FULL run): second parse-repair shape; chained submission

Trigger: 25-row teacher TEST (job 110625): 24/25 parsed, 4 by the A4.1 brace
repair. The one failure (qid 076f905a…) has the final "}" but lacks the "]"
closing the facts list; a balanced block is found and json.loads fails, so
A4.1 does not fire.

A5.1 If the balanced block fails json.loads, try once more: insert "]" before
the final "}" and parse again; tag repaired = "bracket". Full order of
attempts: plain; if no balanced block, append "}" then "]}" (A4.1); if the
block does not parse, insert "]" before the final "}" (A5.1). Nothing else in
parsing changes. Repaired counts are reported as in A4.1.

A5.2 Submission: TEST-5 teacher -> TEST-5 student -> FULL teacher -> FULL
student in one --dependency=afterok chain; the FULL jobs start only if both
TEST-5 jobs exit 0. Basis: the parser has been exercised on 25 rows (A4.2),
the student script is unchanged (MD5 fc5802f1667769eb23af0ff8771dea01), and
the new repair is unit-tested on the failing raw text before submission.

A5.3 Recorded for the readout, not acted on: in the 25 TEST rows the teacher
never used unknown_terms (0/25), including 7 rows whose English restatement
names a different entity than the question (e.g. SnapCap -> Snapdeal, Acela
Express -> MTR Airport Express, abyssal plains -> Abyssinian meadows). The
abstention mechanism did not engage; this is reported as a finding (teacher
overconfidence).
