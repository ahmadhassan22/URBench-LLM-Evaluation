# D11 eval458 — provenance listing (Amendment 6 A6.3 / A6.4; note N2)

Written by `eval/error_analysis_tests/efbpt/d11_eval458_provenance.py` (MD5 `a80f1bb555bc2d11961b71f3b9732506`). It computes no statistic on answers: no gold label is loaded and the student generation files are never parsed (bytes only, to compare md5 / bytes / lines with their seal). Sources: S_START / S_SEAL / S_RESUME of every arm, T_SEAL and t_facts.jsonl of the teacher, the A2.3 regression records, the contract files.

Arms listed: A_r2_eval, TR, TR_wrong_eval, TR_L1_eval.

| arm | state |
|---|---|
| A_r2_eval | sealed |
| TR | sealed |
| TR_wrong_eval | sealed |
| TR_L1_eval | sealed |

## 1. Contract MD5s

| document | file | MD5 on disk | last commit of the file | recorded at the start of |
|---|---|---|---|---|
| D11 freeze | `docs/EFBPT_D11_RTG_FREEZE.md` | `ce7d68801c302fa5a8d173bd3ee93eb7` | 24ee106 | A_r2_eval =; TR =; TR_wrong_eval =; TR_L1_eval = |
| Amendment 1 | `docs/EFBPT_D11_RTG_AMENDMENT_1.md` | `0b7202af45eac170ec2d8fe281269e9e` | 24ee106 | A_r2_eval =; TR =; TR_wrong_eval =; TR_L1_eval = |
| Amendment 2 | `docs/EFBPT_D11_RTG_AMENDMENT_2.md` | `4ea8b0e6406a8a7ea05e33b212215f96` | e1f4498 | A_r2_eval =; TR =; TR_wrong_eval =; TR_L1_eval = |
| Amendment 3 | `docs/EFBPT_D11_RTG_AMENDMENT_3.md` | `39cbac213583c9d08ab3be7c07a97f3c` | 1ac898c | — |
| Amendment 4 | `docs/EFBPT_D11_RTG_AMENDMENT_4.md` | `605c56e95c05fcb05c0c248cbe7ccac1` | e2cee99 | — |
| Amendment 4 note | `docs/EFBPT_D11_RTG_AMENDMENT_4_NOTE.md` | `f9d690a7396de16612f40a879004c0b4` | e3fa2ca | — |
| Amendment 5 | `docs/EFBPT_D11_RTG_AMENDMENT_5.md` | `45c118de9804390d12853f231e7632a0` | 5afe2cb | — |
| Amendment 6 | `docs/EFBPT_D11_RTG_AMENDMENT_6.md` | `bbc3d856ff9e9e7a745a90342d5b0519` | 61d3983 | TR =; TR_wrong_eval =; TR_L1_eval = |
| Amendment 6 note | `docs/EFBPT_D11_RTG_AMENDMENT_6_NOTE.md` | `1c29f903e3b403d8adfeebf1f3150f53` | 2f9ed28 | — |
| GO eval458 | `docs/D11_GO_EVAL458.md` | `a6b06c39c37652e1f474ed9d3d0ba70f` | e1f4498 | — |
| D11-R reader freeze | `docs/EFBPT_D11R_READER_FREEZE.md` | `ed260e0ea2373337e8a3fb850bebe963` | 273b141 | A_r2_eval =; TR =; TR_wrong_eval =; TR_L1_eval = |
| D11-R Amendment 1 | `docs/EFBPT_D11R_AMENDMENT_1.md` | `fb7bc3c922bb28d667241a7dd75b6938` | 73e5645 | A_r2_eval =; TR =; TR_wrong_eval =; TR_L1_eval = |
| D11-R Amendment 2 | `docs/EFBPT_D11R_AMENDMENT_2.md` | `6c466ab2d31bffd3bfe96f43c24678f3` | cd5a520 | A_r2_eval =; TR =; TR_wrong_eval =; TR_L1_eval = |
| D10 teacher freeze | `docs/EFBPT_D10_TEACHER_FREEZE.md` | `b3a822d8199034c7fe58d76cf5cc428f` | 0074ff4 | A_r2_eval =; TR =; TR_wrong_eval =; TR_L1_eval = |

"=": the MD5 in the arm's S_START equals the file on disk. An arm that started before Amendment 6 (script a733caed) does not record it (A6.4).

## 2. Scripts on disk

| script | MD5 on disk | last commit of the file |
|---|---|---|
| `eval/error_analysis_tests/efbpt/d11_student.py` | `ce0786f85ea005b77e2b3c0cdcd0c039` | 0988511 |
| `eval/error_analysis_tests/efbpt/d11_common.py` | `803e97b873cffabf6813041166d5cfdf` | 43b61ca |
| `eval/error_analysis_tests/efbpt/d11o_student.py` | `d66c93e563deb37a6cd0b5b59beb5aab` | 3048554 |
| `eval/error_analysis_tests/efbpt/d11_teacher.py` | `108ca5301910270cf2255a670fc6e9ce` | 43b61ca |
| `eval/error_analysis_tests/efbpt/d11_eval458_readout.py` | `8fb00396fc676a1066d924dfdb3ed874` | 302b506 |
| `eval/error_analysis_tests/efbpt/d11_scorer_audit.py` | `246cf6c9687b3fa67324722cb5cd16b2` | e3fa2ca |

Readout script (A5.2) MD5 = `8fb00396fc676a1066d924dfdb3ed874` named in the Amendment 6 note N2: **yes**.

## 3. A2.3 regression records (DEV200, arm TR, rows 0-4, byte compare with the sealed file)

| record | MD5 of the record | verdict | identical rows | SLURM job | node | end | git head | `d11_student.py` | `d11_common.py` | `d11o_student.py` | Amendment 6 in its contracts | sealed DEV200 TR file |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `outputs/efbpt/d11/_regress/student/dev200/REGRESS.json` | `f784ae506da1e5daff2e4856652b7346` | PASS | 5 / 5 | 115574 | L20002 | 2026-11-10 14:43:16 | `e8b8702` | `ce0786f85ea005b77e2b3c0cdcd0c039` | `803e97b873cffabf6813041166d5cfdf` | `d66c93e563deb37a6cd0b5b59beb5aab` | yes | `bab987cc492012f4e32dbcd348be9668` |
| `outputs/efbpt/d11/_regress/student/dev200_job115527_script_a733caed/REGRESS.json` | `280cf988128b56939777fc541a83cc47` | PASS | 5 / 5 | 115527 | L20003 | 2026-11-10 12:00:39 | `43b61ca` | `a733caede04d1bbb9c2ba9fd9506b133` | `803e97b873cffabf6813041166d5cfdf` | `d66c93e563deb37a6cd0b5b59beb5aab` | no | `bab987cc492012f4e32dbcd348be9668` |

## 4. Student arms: start, scripts, regression record, seal

| arm | state | SLURM job (start) | node | start | git head at start (dirty) | `d11_student.py` | = that file at the git head | `d11_common.py` | regression record it ran under | same scripts as that record |
|---|---|---|---|---|---|---|---|---|---|---|
| A_r2_eval | sealed | 115529 | L20002 | 2026-11-10 12:01:10 | 43b61ca (False) | `a733caede04d1bbb9c2ba9fd9506b133` | yes | `803e97b873cffabf6813041166d5cfdf` | job 115527, `outputs/efbpt/d11/_regress/student/dev200_job115527_script_a733caed/REGRESS.json` | yes (PASS) |
| TR | sealed | 115575 | L20002 | 2026-11-10 14:43:32 | e8b8702 (False) | `ce0786f85ea005b77e2b3c0cdcd0c039` | yes | `803e97b873cffabf6813041166d5cfdf` | job 115574, `outputs/efbpt/d11/_regress/student/dev200/REGRESS.json` | yes (PASS) |
| TR_wrong_eval | sealed | 115576 | L20002 | 2026-11-10 16:26:56 | 2f9ed28 (False) | `ce0786f85ea005b77e2b3c0cdcd0c039` | yes | `803e97b873cffabf6813041166d5cfdf` | job 115574, `outputs/efbpt/d11/_regress/student/dev200/REGRESS.json` | yes (PASS) |
| TR_L1_eval | sealed | 115577 | L20001 | 2026-11-10 19:07:30 | 2f9ed28 (False) | `ce0786f85ea005b77e2b3c0cdcd0c039` | yes | `803e97b873cffabf6813041166d5cfdf` | job 115574, `outputs/efbpt/d11/_regress/student/dev200/REGRESS.json` | yes (PASS) |

`d11_student.py` versions used (A6.4): `a733caede04d1bbb9c2ba9fd9506b133` — A_r2_eval; `ce0786f85ea005b77e2b3c0cdcd0c039` — TR, TR_wrong_eval, TR_L1_eval.

| arm | seal | SLURM job (seal) | end | rows | `script_md5` in the seal | sealed files = disk (md5, bytes, lines) | `scored` | resumes (time, job, rows completed before) |
|---|---|---|---|---|---|---|---|---|
| A_r2_eval | `outputs/efbpt/d11/student/eval458/S_SEAL_A_r2_eval.json` | 115529 | 2026-11-10 16:26:39 | 458 | `a733caede04d1bbb9c2ba9fd9506b133` | yes (2 files) | False | none |
| TR | `outputs/efbpt/d11/student/eval458/S_SEAL_TR.json` | 115575 | 2026-11-10 19:07:03 | 458 | `ce0786f85ea005b77e2b3c0cdcd0c039` | yes (2 files) | False | none |
| TR_wrong_eval | `outputs/efbpt/d11/student/eval458/S_SEAL_TR_wrong_eval.json` | 115576 | 2026-11-10 20:45:36 | 458 | `ce0786f85ea005b77e2b3c0cdcd0c039` | yes (3 files) | False | none |
| TR_L1_eval | `outputs/efbpt/d11/student/eval458/S_SEAL_TR_L1_eval.json` | 115577 | 2026-11-10 23:08:25 | 458 | `ce0786f85ea005b77e2b3c0cdcd0c039` | yes (2 files) | False | none |

## 5. Teacher parse counts

| reader | used by | SLURM job | end | teacher script MD5 | rows | parse_ok (T_SEAL / recount) | repaired rows (T_SEAL / recount) | facts removed by the verdict filter T2 (T_SEAL / recount) | rows with no facts | sealed files = disk | reading file MD5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 (api_qwen3.8-max) | TR, TR_wrong_eval | 115528 | 2026-11-10 13:21:04 | `108ca5301910270cf2255a670fc6e9ce` | 458 | 457 / 457 | 0 / 0 | 3 / 3 | 1 | yes (2 files) | `3211b45ff4bf6b75f07d7ed60d4991eb` |
| R3 (L1_llama70b_awq) | TR_L1_eval | 115532 | 2026-11-10 14:39:30 | `108ca5301910270cf2255a670fc6e9ce` | 458 | 456 / 456 | 0 / 0 | 0 / 0 | 2 | yes (2 files) | `9833dc91c9a2ee42c667f4f70747dbec` |

- R1: row 450 (qid e1ebc4b698eb5bd4fbbe) has no facts — parse_ok False, facts parsed 0, removed by T2 0.
- R3: row 271 (qid df5d1bdd7a56f0090f60) has no facts — parse_ok False, facts parsed 0, removed by T2 0.
- R3: row 450 (qid e1ebc4b698eb5bd4fbbe) has no facts — parse_ok False, facts parsed 0, removed by T2 0.

## 6. Empty-facts rows per arm

| arm | layouts (S_START) | empty-facts prompts (S_START) | prompt rows without facts |
|---|---|---|---|
| A_r2_eval | A 458 | 458 | all rows (no-facts arm, layout A) |
| TR | A 1, B 457 | 1 | row 450 (qid e1ebc4b698eb5bd4fbbe) |
| TR_wrong_eval | A 1, B 457 | 1 | row 104 (qid d15cac26462700b9d6ca) <- facts of row 450 (qid e1ebc4b698eb5bd4fbbe) |
| TR_L1_eval | A 2, B 456 | 2 | row 271 (qid df5d1bdd7a56f0090f60); row 450 (qid e1ebc4b698eb5bd4fbbe) |

## 7. A6.3 non-specific evidence-id hits (logged, not fatal)

| arm | prompt row (qid) | facts of row (qid) | needle (letters) | rows owning the evidence id | needle in the reading | fact sentence |
|---|---|---|---|---|---|---|
| TR | 98 (d1ca8206707ec8277030) | 98 (d1ca8206707ec8277030) | `8-1` (0) | 10 (bbf5422e2b837a2e1ff6) | no | `The density of a standard bowling pin is approximately 0.8-1.0 g/cm³.` |
| TR_wrong_eval | 50 (4173af9c4eeb33828e37) | 98 (d1ca8206707ec8277030) | `8-1` (0) | 10 (bbf5422e2b837a2e1ff6) | no | `The density of a standard bowling pin is approximately 0.8-1.0 g/cm³.` |
| TR_L1_eval | none | — | — | — | — | — |

A_r2_eval has no facts; the rule does not apply to it. Evidence ids with 3 or more letters, question_en and the readings stay fatal: an arm with such a hit does not start.

## 8. V6 hits: gold fact strings contained in a facts prompt (reported, not fatal)

| arm | hits (S_START) | prompt row (qid) | gold fact string |
|---|---|---|---|
| A_r2_eval | 0 (gold facts are fatal for this arm: no facts arm) | — | — |
| TR | 5 in 5 prompt(s) | 3 (feeabf84f23bf6d09d7d) | `Katharine Hepburn died on June 29, 2003.` |
| TR | 5 in 5 prompt(s) | 36 (910d7db694573a1683dd) | `Europa is a moon of Jupiter` |
| TR | 5 in 5 prompt(s) | 76 (cf1365a8abd4e35c2f0b) | `Harry Potter is a fictional character` |
| TR | 5 in 5 prompt(s) | 184 (a73446e82b72b4a99e5b) | `Masaharu Morimoto is a Japanese chef` |
| TR | 5 in 5 prompt(s) | 377 (5055ac2e84f806669a3f) | `Dave Mustaine was fired from Metallica in 1983.` |
| TR_wrong_eval | 5 in 5 prompt(s) | 120 (ffd8a720264778c0fd6e) | `Europa is a moon of Jupiter` |
| TR_wrong_eval | 5 in 5 prompt(s) | 128 (f9abd88f0bd83d3316f0) | `Katharine Hepburn died on June 29, 2003.` |
| TR_wrong_eval | 5 in 5 prompt(s) | 169 (10bcf9c1d9026f741f75) | `Masaharu Morimoto is a Japanese chef` |
| TR_wrong_eval | 5 in 5 prompt(s) | 266 (3a373a74e76c72176e39) | `Harry Potter is a fictional character` |
| TR_wrong_eval | 5 in 5 prompt(s) | 294 (b8976035f0804cbd356e) | `Dave Mustaine was fired from Metallica in 1983.` |
| TR_L1_eval | 5 in 5 prompt(s) | 76 (cf1365a8abd4e35c2f0b) | `Harry Potter is a fictional character` |
| TR_L1_eval | 5 in 5 prompt(s) | 176 (bf739e062c0321178c7a) | `The Great Gatsby was published in 1925.` |
| TR_L1_eval | 5 in 5 prompt(s) | 184 (a73446e82b72b4a99e5b) | `Masaharu Morimoto is a Japanese chef` |
| TR_L1_eval | 5 in 5 prompt(s) | 377 (5055ac2e84f806669a3f) | `Dave Mustaine was fired from Metallica in 1983.` |
| TR_L1_eval | 5 in 5 prompt(s) | 412 (463faa7aaa7d9e1e366b) | `Europa is a moon of Jupiter` |

## Checks

All checks of this listing hold: seals equal the files on disk, every started arm ran under a PASSED regression record with the same scripts, its `d11_student.py` is the file at its git head, the teacher recounts equal the T_SEAL counts, the recorded contract MD5s equal the files on disk.
