# D11-R — RDEV300 reader readout and §4 selection (2026-11-09)

Read-only on reader files. The API resume chain (PID 3006865, Amendment 2) was
running on dev200 during this readout and was not touched; no dev200 or eval458
file was opened (audit in §6). eval458 readings remain sealed unread.

## 0. Provenance

| contract / input | MD5 |
|---|---|
| docs/EFBPT_D11R_READER_FREEZE.md | ed260e0ea2373337e8a3fb850bebe963 |
| docs/EFBPT_D11R_AMENDMENT_1.md | fb7bc3c922bb28d667241a7dd75b6938 |
| docs/EFBPT_D11R_AMENDMENT_2.md | 6c466ab2d31bffd3bfe96f43c24678f3 |
| data/strategyqa_official/efbpt/rdev300_seed2026.jsonl (300 rows) | e307ad857b2f646f423bcd09fd3060d7 |
| eval/error_analysis_tests/efbpt/d11r_score.py (scorer, unchanged since the freeze commit) | 978caaffe41721e6aa7d5938a1dc0dd3 |
| eval/error_analysis_tests/efbpt/d11r_common.py | 4205923e6aca8294d9c1e10f860523a7 |
| outputs/efbpt/d11r/_scores/scores_rdev300.json (written by this run, --rows) | 8f1061b04d8507281ac60e6a3d813c3f |

RDEV300 reading files scored (each MD5 equals the entry in the reader's R_SEAL.json, checked before scoring):

| reader | readings_rdev300.jsonl MD5 | R_SEAL sets_complete at check time |
|---|---|---|
| L1_llama70b_awq | 390098c8c20ad06c11d5faaf4a268fb2 | rdev300, dev200, eval458 |
| L2_gemma2_27b_nf4 | af345e2f16cb178ec7011fe8332ec1f5 | rdev300, dev200, eval458 |
| L3_qwen3_14b_nf4 | f9028133b6cea5bb672d189d18424c34 | rdev300, dev200, eval458 |
| L4_nllb200_3.3b | 0c9ff6e17089de775530312ee09dd0f5 | rdev300, dev200, eval458 |
| api_qwen3.8-max | 917ed0fac0b0e79b1446bf33a0d50704 | rdev300, dev200 |
| api_deepseek-v4-pro | 4380d5165acef32005734de70179c1df | rdev300 |
| api_deepseek-v4-flash | 63f7baf6f039a7cf01e68ca46f7a72f0 | rdev300, dev200 |
| api_qwen3.7-flash | 2c732f31335c4726fe09f2d39aec41b6 | rdev300, dev200 |

All four API readers: enable_thinking=false (Amendment 1), temperature 0, max_tokens 256; every reader
300/300 ok records, 0 missing, 0 failed.

Command: `python -B eval/error_analysis_tests/efbpt/d11r_score.py --set rdev300 --rows` (env urbench_eval,
sacrebleu 2.6.0; chrF signature nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0).

## 1. Scores (RDEV300, n = 300; M1 over the 233 rows with >= 1 gold proper noun, 67 rows without are excluded from M1 only)

| reader | M1 macro % (primary) | M1 micro % | M2 chrF corpus (sentence mean) | M3 exact % (n) | M4 title presence, ratio to gold ceiling % (in reading / in gold, of 396 required titles) | empty or garbled (< 3 Latin letters) | Urdu-script leakage | mean latency s (median, p90) | mean output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L1_llama70b_awq | 87.95 | 88.89 | 66.87 (66.84) | 7.33 (22) | 84.60 (335 / 396) | 0 | 0 | 0.950 (0.949, 1.039) | 13.5 |
| L2_gemma2_27b_nf4 | 82.29 | 86.11 | 65.39 (65.10) | 9.67 (29) | 78.79 (312 / 396) | 0 | 0 | 1.739 (1.671, 2.338) | 14.2 |
| L3_qwen3_14b_nf4 | 74.33 | 78.09 | 60.81 (60.46) | 5.67 (17) | 65.66 (260 / 396) | 0 | 0 | 0.932 (0.879, 1.300) | 12.8 |
| L4_nllb200_3.3b | 84.48 | 87.19 | 67.38 (67.51) | 11.00 (33) | 81.06 (321 / 396) | 0 | 0 | 0.291 (0.276, 0.370) | 15.2 |
| api_deepseek-v4-flash | 89.30 | 91.67 | 71.68 (71.52) | 12.67 (38) | 89.65 (355 / 396) | 0 | 0 | 1.125 (1.058, 1.308) | 13.5 |
| api_deepseek-v4-pro | **91.93** | 93.36 | 72.38 (72.05) | 13.67 (41) | **93.43** (370 / 396) | 0 | 0 | 1.914 (1.883, 2.256) | 12.5 |
| api_qwen3.7-flash | 83.45 | 86.11 | 67.27 (67.23) | 10.33 (31) | 82.58 (327 / 396) | 0 | 0 | 1.087 (1.029, 1.346) | 12.6 |
| api_qwen3.8-max | 91.55 | **93.52** | **73.16** (73.25) | **14.67** (44) | 91.41 (362 / 396) | 0 | 0 | 1.488 (1.415, 1.736) | 12.7 |

M4 gold ceiling: 396 of the 932 required evidence titles occur verbatim (norm substring) in the gold English
question; the ratio column divides titles found in the reading by that 396. Local latencies are per item as
recorded by the local reader scripts (batched GPU inference), API latencies are per request on the login
node; they are not comparable across the two groups. Reasoning tokens: 0 for local readers; not reported by
the gateway for the API readers (reasoning_evidence "not_reported_no_reasoning_content", Amendment 1 A1.1
counts this as 0). Cost: configs/d11r_api_prices.json is null, cost n/a (Amendment 1 A1.4); the cost
tie-break was not reached.

## 2. §4 selection rule, applied as written

Rule (freeze §4): best M1 on RDEV300; if two readers are within 1.0 pp of M1, the higher M2 wins; ties →
local over API, cheaper over dearer.

1. Best M1: api_deepseek-v4-pro 91.93. Within 1.0 pp: api_deepseek-v4-pro (91.93), api_qwen3.8-max (91.55;
   difference 0.38 pp). Next reader api_deepseek-v4-flash 89.30 is 2.63 pp below, outside the band.
2. M2 among the two: api_qwen3.8-max 73.16 > api_deepseek-v4-pro 72.38 → **api_qwen3.8-max wins**.
3. No M2 tie, so the local/API and cost tie-breaks were not used.
4. Runner-up, same procedure on the remaining seven: best M1 api_deepseek-v4-pro 91.93; within 1.0 pp: only
   itself → **api_deepseek-v4-pro**.

**SELECTED: api_qwen3.8-max** (M1 91.55, M2 73.16). **RUNNER-UP: api_deepseek-v4-pro** (M1 91.93, M2 72.38).
Both are carried into the D11 freeze (runner-up = agreement partner). Best local reader by the same rule:
L1_llama70b_awq (M1 87.95; L4 at 84.48 is outside its 1.0 pp band).

## 3. Paired row-level M1 (for the thesis)

233 rows with >= 1 gold proper noun. Row recall = found / gold proper nouns on that row. A win is a strictly
higher row recall. p-value: exact two-sided binomial sign test on the discordant rows (P(X <= min(wins,
losses)) × 2 under p = 0.5, capped at 1). Mean ΔM1 = mean of the per-row differences, in pp.

| pair A vs B | A > B | A < B | ties | discordant | sign-test p (two-sided) | mean ΔM1 (A − B), pp |
|---|---:|---:|---:|---:|---:|---:|
| SELECTED api_qwen3.8-max vs RUNNER-UP api_deepseek-v4-pro | 12 | 11 | 210 | 23 | 1.00 | −0.38 |
| best API (selected) api_qwen3.8-max vs best local L1_llama70b_awq | 31 | 7 | 195 | 38 | 1.2 × 10⁻⁴ | +3.61 |
| best API by raw M1 api_deepseek-v4-pro vs best local L1_llama70b_awq | 27 | 4 | 202 | 31 | 3.4 × 10⁻⁵ | +3.98 |

Reading: on these 233 rows the selected and runner-up readers are indistinguishable on M1 (12 vs 11
discordant rows); the selection between them rests on the pre-declared M2 tie-break (0.78 chrF points).
Both API readers beat the best local reader on M1 at row level. These are observations on this one 300-row
sample; generalisation to DEV200 / eval458 is not established here.

## 4. Hand check: the 10 RDEV300 rows where the SELECTED reader (api_qwen3.8-max) has the lowest M1

All ten have row M1 = 0. Order: M1 ascending, then more gold proper nouns first, then qid. "missed" lists the
gold proper nouns (freeze §3 definition, punctuation stripped) not found in the selected reading.

| # | qid | gold question_en | missed | selected reading (api_qwen3.8-max) | runner-up reading (api_deepseek-v4-pro), its M1 |
|---|---|---|---|---|---|
| 1 | 11f0ae8346b66bc1d948 | Is Tange Sazen hypothetically an ideal choice for a secretary job? | Tange, Sazen | Is a coachbuilder hypothetically the ideal choice for the secretary's job? | Is Tang Sazan hypothetically an ideal choice for the secretary's job? (0.00) |
| 2 | 0a32d7cfde6cec332fd6 | Could ABBA play a mixed doubles tennis game against each other? | ABBA | Can Eba play mixed doubles tennis against each other? | Can Ebba play mixed doubles tennis against each other? (0.00) |
| 3 | 1e8d7da6c0e0fea74a09 | Could casualties from deadliest war rival France's population? | Frances | Could the casualties of the deadliest war equal the population of France? | Could the casualties of the deadliest war equal the population of France? (0.00) |
| 4 | 1ffcadc478d4f408b988 | Are lengths measured in metres in the UK? | UK | Are lengths measured in meters in Britain? | Are lengths measured in meters in Britain? (0.00) |
| 5 | 37505be3ab3ef4b7f2ab | Is Pan a symbol of virtue and virginity in women? | Pan | Is the hymen a sign of virtue and virginity in women? | Is the pen a symbol of virtue and virginity in women? (0.00) |
| 6 | 3df705e3e1ff592c8149 | Would Hodor hypothetically be a good math mathematician? | Hodor | Could Hoder hypothetically be a good mathematician? | Could Hodor hypothetically be a good mathematician? (1.00) |
| 7 | 517614ab4adf946dbe20 | Was Eve involved in an incestuous relationship? | Eve | Did Hawa have a relationship with a close relative? | Did Eve have a relationship with a close relative? (1.00) |
| 8 | 54b08f028141c591badd | Is dyslexia the most common intellectual disability in US? | US | Is dyslexia the most common intellectual disability in America? | Is dyslexia the most common mental disability in America? (0.00) |
| 9 | a4526fcfad49a21eed15 | Can nitric acid break the Louvre? | Louvre | Can nitric acid damage leather? | Can nitric acid damage a louver? (0.00) |
| 10 | fb0f995e63ec3ba8836d | Are all students guaranteed lunch at school in the US? | US | Are all students in America guaranteed lunch at school? | Are all students in America guaranteed lunch at school? (0.00) |

Observations for the hand check (metric left as frozen; these are notes, not corrections):
- Rows 3, 4, 8, 10 are M1 misses by construction, not reading errors: "France's" becomes the token
  "Frances" after punctuation stripping and "France" does not contain it; "UK" → "Britain" and "US" →
  "America" are meaning-preserving renderings that M1 counts as misses for every reader.
- Rows 1, 2, 5, 6, 7, 9 are genuine entity errors in the selected reading (transliteration drift: Tange
  Sazen, ABBA, Hodor, Eve rendered as the Urdu/Arabic name Hawa, Louvre read as "leather"; row 5 "Pan" is
  itself an odd gold). The runner-up recovers Hodor and Eve (rows 6, 7) and gets closer on rows 1, 2, 9.
- Rows 2 and 9 contain qids that were among the 29 rows deepseek-v4-flash lost to the 2026-11-07 outage
  (a4526fcfad49a21eed15, fb0f995e63ec3ba8836d); they were re-read on 2026-11-09 and scored normally here.

## 5. Status of the other sets (not scored here)

API chain (Amendment 2) at 14:16 on 2026-11-09: rdev300 sealed for all four API readers; dev200 sealed for
qwen3.8-max, deepseek-v4-flash, qwen3.7-flash; deepseek-v4-pro dev200 in progress (0 errors so far);
eval458 not started. Local readers L1–L4: all three sets sealed since 2026-11-07.

## 6. Scorer file audit

The scorer ran under a Python audit hook recording every open() call. Non-library files opened: the 8
readings_rdev300.jsonl files (twice each: md5 and read), rdev300_seed2026.jsonl, d11r_score.py,
d3_oracle_retrieval.py (norm() source check), EFBPT_D11R_READER_FREEZE.md, EFBPT_D11R_AMENDMENT_1.md,
configs/d11r_api_prices.json, and the output scores_rdev300.json. Paths containing "eval458" or the mapped
official file: 0. No readings_dev200.jsonl was opened either. The scorer also refuses --set eval458 by code
(SCORABLE_SETS = rdev300, dev200).
