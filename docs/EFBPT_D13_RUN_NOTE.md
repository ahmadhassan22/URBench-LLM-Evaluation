# EFBPT D13 — run note (interruptions, one failed job, resubmission)

Companion to docs/EFBPT_D13_BASELINES_TRANSFER_FREEZE.md (MD5 2ee24716b0e0ca2db0cc5f8fb986ac08).
Written 2026-11-14. It records what happened during the full runs. It changes no rule, no
script and no sealed file. **No arm was scored**: every seal has `scored: false`, no D13 readout
script exists yet, and no job computed or printed accuracy.

Scripts: the d13 scripts are those of commit 13dd959, unchanged (every run printed
"d13 scripts are committed and unchanged").

## 1. API arm API_direct_ur_eval (eval458): interrupted twice, resumed, sealed

The API runner keeps every ok record in a work file and never requests a finished row again
(freeze §7.2).

| run | start | rows already ok | what happened |
|---|---|---:|---|
| 1 | 2026-11-13 12:18:02 | 0 | stopped by a restart of the session at 12:34; 374 ok rows on disk |
| 2 | 2026-11-13 12:35:48 | 374 | 42 more ok rows; then the proxy dropped: 42 rows ended with network errors (12:48:18 to 12:57:49), exit 3 at 12:57:53; 416 ok rows on disk |
| 3 | 2026-11-14 14:28:57 | 416 | the remaining 42 rows, all ok; sealed 14:29:21 |

Seal: outputs/efbpt/d13/large_alone/eval458/SEAL_API_direct_ur_eval.json, 458 rows, row file MD5
9531a4a66ff937958883f99d5db3434b, finish reason "stop" on 458 rows. The 42 network-error records
stay in the work file (MD5 984a1274ba7fc91dd923daa2d5d0d32b); they carry no answer. The three
runs are listed in START_API_direct_ur_eval.json.

## 2. API arm API_tr_R1

Started after §1 was sealed, on 2026-11-14.

- dev200: one run, 200 rows, sealed 14:31:16, row file MD5 df0ea78cdf605a1e6788b8127b8c8490.
- eval458: **not sealed.** 456 of 458 rows are ok. For two rows the provider refuses the
  request before any answer: HTTP 400, `data_inspection_failed`, "Input text data may contain
  inappropriate content." The qids are 90538ec7aa1100a34829 and 78f2f99d04b9acd8bada. The refusal
  came on the first run (14:34) and again on a resume of these two rows only (14:36), so it is
  not a network fault. HTTP 400 is not in the retry list. The 456 ok records are kept in
  outputs/efbpt/d13/large_alone/eval458/d13_API_tr_R1.work.jsonl. Under freeze §7.6 a seal needs
  every row, so the arm stays open. What to do with these two rows is not decided here.

## 3. Retrieval job 116804 failed before writing output

Job 116804 (d13_retrieve FULL dev200,eval458) ran on node L20007 on 2026-11-13 from 13:35:08 to
13:35:22 and ended FAILED (exit 1): "git is not usable (FileNotFoundError); a FULL run needs
the committed d13 scripts". L20007 has no git, so the committed-scripts check could not run.
The job stopped before any model was loaded and wrote nothing under outputs/efbpt/d13/rag/.

Nodes checked on 2026-11-14 with `command -v git`: L20004 and L20007 have no git.

## 4. C1 regression job 116805 stands

Job 116805 (d13_student qwen14 REGRESS) ran on L20007 on 2026-11-13 and passed: 5/5 generations
byte-identical to the sealed D11 file (outputs/efbpt/d13/_regress/qwen14/REGRESS.json). The
REGRESS mode does not need git. On 2026-11-14 all 23 script MD5s recorded in REGRESS.json
(19 under `scripts`, 4 under `gate_scripts`) were compared with the files on disk: all
unchanged. The regression was not run again.

## 5. Jobs 116806-116812 cancelled unrun, resubmitted with a node exclusion

Jobs 116806-116809 depended on `afterok:116804` and could never start; jobs 116810-116812
depended on 116806-116809. All seven were cancelled on 2026-11-14 without having started
(no start time, no log, no output).

Resubmitted on 2026-11-14, every job with `--exclude=L20004,L20007`, same batch files and
arguments as before:

| job | what | dependency |
|---|---|---|
| 117476 | d13_retrieve FULL dev200,eval458 | none |
| 117477 | qwen14 FULL eval458 RAG_ur | afterok:117476 |
| 117478 | qwen14 FULL eval458 RAG_en | afterok:117476 |
| 117479 | qwen14 FULL dev200 RAG_ur,RAG_en | afterok:117476 |
| 117480 | qwen14 FULL eval458 RO_eval | none |
| 117481 | llama8 FULL eval458 | none |
| 117482 | gemma9 FULL eval458 | none |
| 117483 | qwen8 FULL eval458 | none |

RO_eval and the three Group 4 jobs no longer depend on the retrieval job, so a retrieval
failure cannot block them. Job order affects no number (freeze §7.7).
