# D11 scorer audit — two further adjudicators (Amendment 5 A5.6)

Agreement only: these labels select nothing; the validated scorer is the one chosen by the PRIMARY labels (E3). Written by `eval/error_analysis_tests/efbpt/d11_scorer_audit_adjudicators.py` (MD5 `4b2d04432912dd32ce35f7bb698c0d73`). Items `docs/handoff/D11_SCORER_AUDIT_ITEMS.md` (MD5 `7448bae2e400790f2b93c00359b26960`), 105 items; PRIMARY labels MD5 `fb47ed80e34aac4c782bfc90d0f89fcb`. Each model got the rubric N2 R1–R4 in English as the system message (MD5 `fb5e753d92689e7d2fab7484a6920617`; the item numbers of the note are not in it) and one item per call (the Urdu question and the text shown to adjudicator A); it did not receive arm, gold, qid or any label. temperature 0, default thinking, max_tokens 4000, one retry with enable_thinking=false on an empty reply.

| adjudicator | yes | no | none | invalid | strict one-word replies | lenient | retried with enable_thinking=false | raw replies (sealed) |
|---|---|---|---|---|---|---|---|---|
| PRIMARY (adjudicator A) | 61 | 33 | 11 | 0 | — | — | — | `docs/handoff/D11_SCORER_AUDIT_LABELS_PRIMARY.json` |
| qwen3.8-max | 60 | 32 | 13 | 0 | 105 | 0 | 0 | `outputs/efbpt/d11/_audit/adjudicators/qwen3.8-max/replies.jsonl` (MD5 `4a0b0017c0c5b01ed2f1e1c71139be4c`) |
| deepseek-v4-pro | 58 | 34 | 13 | 0 | 105 | 0 | 0 | `outputs/efbpt/d11/_audit/adjudicators/deepseek-v4-pro/replies.jsonl` (MD5 `5c1c3ec6916d33373a86f5f07d2aecb1`) |
| majority of the three | 60 | 33 | 12 | 0 | — | — | — | — |

## Pairwise agreement and Cohen's kappa

| pair | items agreeing | agreement % | Cohen's kappa |
|---|---|---|---|
| PRIMARY vs qwen3.8-max | 101 / 105 | 96.19 | 0.932 |
| PRIMARY vs deepseek-v4-pro | 100 / 105 | 95.24 | 0.916 |
| qwen3.8-max vs deepseek-v4-pro | 102 / 105 | 97.14 | 0.950 |

Fleiss' kappa (three adjudicators, 105 items): **0.933** (mean pairwise agreement per item 96.19 %). All three agree on 99 items.

## Items where either model disagrees with PRIMARY

| item | PRIMARY | qwen3.8-max | deepseek-v4-pro |
|---|---|---|---|
| 28 | no | none | no |
| 40 | yes | yes | none |
| 41 | none | yes | yes |
| 78 | yes | yes | no |
| 95 | yes | none | none |
| 101 | yes | none | none |

6 items; on 3 of them both models give the same label, against PRIMARY (the majority label then differs from PRIMARY).

## A4.3 recomputed with the majority label (sensitivity; selects nothing)

Majority = the label given by at least two of PRIMARY, qwen3.8-max, deepseek-v4-pro; no two agree → none. For comparison, the PRIMARY column repeats the table of the audit.

| arm | extractor | differ rows right (PRIMARY) | differ rows right (majority) | agree rows right (PRIMARY) | agree rows right (majority) | stratified % (PRIMARY) | stratified % (majority) |
|---|---|---|---|---|---|---|---|
| A_r2 | E0 | 8 / 38 | 8 / 38 | 20 / 20 | 19 / 20 | 85.00 | 80.95 |
| A_r2 | E1 | 22 / 38 | 23 / 38 | 20 / 20 | 19 / 20 | 92.00 | 88.45 |
| A_r2 | E2 | 23 / 38 | 24 / 38 | 20 / 20 | 19 / 20 | 92.50 | 88.95 |
| A_r2 | E3 | 24 / 38 | 24 / 38 | 20 / 20 | 19 / 20 | 93.00 | 88.95 |
| TR | E0 | 1 / 27 | 1 / 27 | 20 / 20 | 20 / 20 | 87.00 | 87.00 |
| TR | E1 | 23 / 27 | 22 / 27 | 20 / 20 | 20 / 20 | 98.00 | 97.50 |
| TR | E2 | 23 / 27 | 22 / 27 | 20 / 20 | 20 / 20 | 98.00 | 97.50 |
| TR | E3 | 24 / 27 | 23 / 27 | 20 / 20 | 20 / 20 | 98.50 | 98.00 |

| extractor | items right of 65 differ (PRIMARY / majority) | items right of 40 agree (PRIMARY / majority) | mean stratified agreement % (PRIMARY / majority) |
|---|---|---|---|
| E0 | 9 / 9 | 40 / 39 | 86.00 / 83.97 |
| E1 | 45 / 45 | 40 / 39 | 95.00 / 92.97 |
| E2 | 46 / 46 | 40 / 39 | 95.25 / 93.23 |
| E3 | 48 / 47 | 40 / 39 | 95.75 / 93.47 |

Highest mean under the majority label (ties → E2, E3, E1): E3. Under PRIMARY: E3 (the validated scorer; unchanged by this table).
