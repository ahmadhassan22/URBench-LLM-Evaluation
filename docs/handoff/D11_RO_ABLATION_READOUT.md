# D11 A3.6 readout: DEV200 ablation RO / RO_gold (descriptive)

Recorded 2026-11-12. This file gives the RO ablation a committed source. It copies the sealed summary
`outputs/efbpt/d11/ablate_ro/dev200/d11_ablate_ro_summary.json` (MD5 `54d22ca19f094811bbaf919431c8d91c`),
which is under the gitignored `outputs/` tree. No model was run to write it and no sealed file was changed.

Every number below was also recomputed on 2026-11-12 from the sealed arm files: the frozen extractor
(`efbpt_eval_dev200.extract_answer`) was run again on each sealed generation and reproduces the sealed
`pred` on every row, and the exact two-sided McNemar test is `d5_balanced_extract.mcnemar`.
All recomputed values equal the sealed summary.

Spec: `docs/EFBPT_D11_RTG_AMENDMENT_3.md` §A3.6 (MD5 `39cbac213583c9d08ab3be7c07a97f3c`).
Descriptive only: not gated, no decision depends on it, no eval458 row was used.

## Arms

- **RO** = the frozen student (Qwen3-14B), layout A (no facts), plus one line directly after the Urdu
  question: `سوال کا انگریزی ترجمہ: {R1 reading}`. The R1 reading is the sealed `api_qwen3.8-max`
  reading of DEV200 (MD5 `1f5b2db3c1403f63b3525a1562edd176`).
- **RO_gold** = the same with the gold `question_en` in place of the reading (oracle translate-test ceiling).
- No facts in either arm.

## Run

| item | value |
|---|---|
| set | DEV200, n = 200 (87 gold yes / 113 gold no), majority floor 56.50% |
| SLURM job | 115548, host L20002 |
| start / end | 2026-11-10 20:49:05 / 2026-11-11 00:29:24 |
| script | `eval/error_analysis_tests/efbpt/d11_ablate_ro.py`, MD5 `b3d42f4c7b6805d4f91f45a2e28dca5b` |
| V2' probe (5 A_r2 rows byte-identical to the sealed file) | 5 / 5 identical, pass |
| readings equal to the gold `question_en` | 25 of 200 |
| readings that contain "yes" or "no" (V4, reported) | 1 of 200 |

Sealed files (all in `outputs/efbpt/d11/ablate_ro/dev200/`):

| file | MD5 | lines |
|---|---|---|
| `d11_armRO.jsonl` | `d3a8850836389d89eac5369601de2f98` | 200 |
| `d11_armRO_gold.jsonl` | `7295b2533b98e0c0b1b68025ce056806` | 200 |
| `d11_ablate_ro_summary.json` | `54d22ca19f094811bbaf919431c8d91c` | 237 |
| `S_SEAL_RO.json` | `249abbad92db31e08ee8ffb5ab78e22e` | 24 |
| `S_SEAL_RO_gold.json` | `cfba367a5ff4b95323f0c5bb250d172c` | 24 |
| `S_SEAL.json` | `55938bd5a81227904f8dfc808680fee3` | |
| `S_START.json` | `fbe4fe6e6874f4b301a8b2d4fbc47d5d` | 369 |

Sealed comparison arms: A_r2 `outputs/efbpt/d10/student/d10_armA_r2.jsonl` (MD5 `7ccec942c8f0eee08156d258baaf806d`);
TR `outputs/efbpt/d11/student/dev200/d11_armTR.jsonl` (MD5 `bab987cc492012f4e32dbcd348be9668`);
TO1 `outputs/efbpt/d11o/student/d11o_armTO1.jsonl` (MD5 `6548b408a68a8e94bb65b9ca6d023fdd`).

## Per-arm table (frozen extractor E0, unparsed = wrong)

| arm | source | acc % (correct) | unparsed | truncated | predicted yes % | secondary-extractor acc % |
|---|---|---|---|---|---|---|
| A_r2 | sealed | 60.00 (120) | 8 | 11 | 41.0 | 60.00 |
| RO | job 115548 | 57.50 (115) | 9 | 9 | 35.5 | 57.50 |
| RO_gold | job 115548 | 61.50 (123) | 7 | 11 | 37.5 | 61.50 |
| TR | sealed | 68.50 (137) | 3 | 9 | 33.5 | 68.50 |
| TO1 | sealed | 74.00 (148) | 2 | 14 | 32.0 | 74.00 |

Rows with a Devanagari answer string in the generation: 0 in every arm.

By gold class (recomputed from the two sealed arm files; not in the sealed summary):
RO is right on 39 of 87 gold-yes and 76 of 113 gold-no rows; RO_gold on 44 of 87 and 79 of 113.

## Comparisons

b = the first-named arm right and the second wrong; c = the reverse; p = exact two-sided McNemar.

| comparison | acc first | acc second | Δ pp | b | c | p |
|---|---|---|---|---|---|---|
| RO vs A_r2 | 57.50 | 60.00 | −2.5 | 29 | 34 | 0.6147 |
| TR vs RO | 68.50 | 57.50 | +11.0 | 41 | 19 | 0.006218 |
| RO_gold vs A_r2 | 61.50 | 60.00 | +1.5 | 40 | 37 | 0.8199 |
| TO1 vs RO_gold | 74.00 | 61.50 | +12.5 | 42 | 17 | 0.001547 |

## Predictions of A3.6

| prediction | observed | result |
|---|---|---|
| RO − A_r2 in [0, +6] | −2.5 | miss |
| RO_gold − A_r2 in [+1, +8] | +1.5 | hit |
| TR − RO ≥ +2 | +11.0 | hit |
| TO1 − RO_gold ≥ +5 | +12.5 | hit |

## Reading

On DEV200, adding the English translation of the question to the prompt, with no facts, does not move the
student: RO is 2.5 pp below A_r2 (p = 0.61) and even the gold translation is only 1.5 pp above it
(p = 0.82). The arms that also carry teacher facts are 11.0 pp (TR vs RO) and 12.5 pp (TO1 vs RO_gold)
above the matching translation-only arm. This is a DEV200 description only; RO has not been run on eval458.
