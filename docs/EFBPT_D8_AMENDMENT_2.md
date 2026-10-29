# D8 — Amendment 2 (FROZEN 2026-10-29, pre-result)

Status: amends docs/EFBPT_D8_REACHABLE_CEILING_FREEZE.md and
docs/EFBPT_D8_AMENDMENT_1.md after FULL job 107156 was submitted and before
any paired D8 reading exists. Job 107156 computed no paired reading (see
A2.1); its outputs are archived and unused. The freeze and Amendment 1 stay
in force except where this file says otherwise. No change after a paired
reading exists.

## A2.1. Reason

FULL job 107156 (2026-10-28, node L20002, 01:02:23 elapsed) stopped at
PASS 2 arm Xp, prompt 186, qid f07a6acf27e25dedbb8d, with

    FATAL: LEAK in Xp/pass2 prompt 186: contains gold fact 'Andrew Johnson was the 17th president of the United States.'

The pass-1 Xp extraction from the corpus page for that qid produced the
sentence "Andrew Johnson was the 17th president of the United States.",
byte-identical to one of the two gold facts of the question. All pass-1
prompts were verified clean by the leak guard before generation (log lines
71-74: A_r/pass2, Xp/pass1, GEN/pass1, X1u/pass1, each against 200
question_en, 532 gold facts, 862 evidence ids), so the string is the model
restating a page fact in its own pass-1 output, not a leak of gold data into
a prompt. The same scan over the archived pass-1 outputs shows GEN pass 2
would have carried 3 such strings (qids 06215d4b6f1a4b975d9c,
06ebd3ca1fe98d5796f7, f07a6acf27e25dedbb8d) and X1u pass 2 none.

Pass 1 (Xp, GEN, X1u extractions) and PASS 2 arm A_r completed in that job;
PASS 2 Xp, GEN and X1u did not start; no summary and no seal were written.
The job's outputs are archived under outputs/efbpt/d8_job107156_failed/ and
are not used for any reading. The read-only readout is
docs/handoff/D8_FULL_READOUT.md.

## A2.2. Rule: verbatim gold-fact substrings in pass-2 prompts of Xp, GEN, X1u

In the pass-2 prompts of arms Xp, GEN and X1u, a verbatim gold-fact substring
is not fatal. It is counted and listed per arm (qid + matched fact) and
reported in d8_summary.json as V6 ("V6_gold_fact_verbatim_pass2"), printed
after V5. V6 is a report, not a gate: no verdict depends on it.

Unchanged and still fatal in every prompt: a question_en match and an
evidence-id match. Unchanged and still fatal: a gold-fact match in every
pass-1 prompt (Xp, GEN, X1u) and in the A_r pass-2 prompt, which carries no
facts.

Reason. The pass-1 prompts never contain gold facts (guarded). A fact the
model extracts from a gold page can coincide with a gold fact because the
gold facts were written from the same pages; that coincidence is the
mechanism D8 measures, not a contamination path. Dying on it discards the
whole job and reports nothing.

## A2.3. Additional report, no gate: V2b

If outputs/efbpt/d8_job107156_failed/d8_armA_r.jsonl exists, the summary
reports V2b ("V2b_Ar_vs_archived_Ar") next to V2: the accuracy difference and
the row agreement between that archived A_r (job 107156; same stack, same
batch size 8, same frozen prompt) and the rerun's A_r. V2b is a report, not a
gate. If the archive file is absent, V2b is skipped with a note in the
summary.

For reference, from the readout: job 107156 A_r scored 124/200 (62.00%)
against D1 arm A's 115/200 (57.50%), with 155/200 rows agreeing and 27/200
byte-identical generations. V2 (A_r vs D1 arm A) and its threshold are
unchanged.

## A2.4. Everything else unchanged

Arms, page set (Amendment 1 A1), pre-run counts (Amendment 1 A2: 256 EXACT,
156 EXACT-and-present, 124 qids, {0: 76, 1: 96, 2: 24, 3: 4}), GEN system
message (Amendment 1 A3), wall time (Amendment 1 A4), V1-V5, R1-R3, the
secondary readings and every threshold are unchanged. Declared before any
paired reading exists.

## Execution

Script updated in the same change as this file, minimally: leak_guard gains
a parameter fatal_gold_facts (default True) and returns the list of
(prompt index, matched fact); run_pass2 calls it with
fatal_gold_facts=(arm == "A_r") and keeps the matches per arm keyed by qid;
the summary gains V6 (printed after V5) and V2b (next to V2); AMENDMENT_PATH
becomes a list of both amendment files, each required to exist, each MD5
recorded in d8_START.json and d8_SEAL.json next to the freeze MD5. Nothing
else changes. --preflight-only must PASS with the Amendment 1 A2 counts
before TEST-5 (--limit 5 --dry-run) and FULL, as in freeze §9.
