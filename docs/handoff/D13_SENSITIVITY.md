# D13 sensitivity check of the scorer — written after the results

**This check was written after the D13 results were seen. It changes no verdict.** The D13 readout (`docs/handoff/D13_READOUT.md`) and its wording stand as they are; the frozen extractor E0 remains the scorer of record. The lines below only show how the all-rows line of four comparisons moves under a wider scorer.

Written by `eval/error_analysis_tests/efbpt/d13_sensitivity.py` (MD5 `7be6638a65e3b7fb315cf0b09e0e19ea`). No model was run and no sealed file was changed. The arm files are read through the committed D13 readout code (same seal checks, same E0, same paired line).

Scorer E0x = E0, and only where E0 is unparsed, in this order: (1) the answer's first three words contain the Urdu word "جی" -> yes; (2) else Devanagari: the last of "नहीं" (no) and "हाँ" / "हां" (yes); (3) else Latin: the last match of `\b(nahi|nahin|nahee?n)\b` (no) or `\b(haan|ji haan|jee haan)\b` (yes), case-insensitive. A word is a maximal run of letters and combining marks. A row E0 parses keeps its E0 label.

Set: eval458, 458 rows, all rows. Unparsed = wrong. b = first arm right and second wrong; exact two-sided McNemar.

## 1. Sealed files read

| arm | file | MD5 |
|---|---|---|
| Llama8B_TR | `outputs/efbpt/d13/transfer/eval458/d13_Llama8B_TR.jsonl` | `25c3d30630bc64547398aab105f7cc81` |
| Llama8B_A | `outputs/efbpt/d13/transfer/eval458/d13_Llama8B_A.jsonl` | `742ebc8f56d060e7a053459006c53b9e` |
| Gemma9B_TR | `outputs/efbpt/d13/transfer/eval458/d13_Gemma9B_TR.jsonl` | `4e223aed0fc73888c1596af133b1370e` |
| Gemma9B_A | `outputs/efbpt/d13/transfer/eval458/d13_Gemma9B_A.jsonl` | `bafddee3cdac3bc3b27459b0257f1e0a` |
| Qwen8B_TR | `outputs/efbpt/d13/transfer/eval458/d13_Qwen8B_TR.jsonl` | `62325ec5c189d9f9ce24cc99a543bae3` |
| Qwen8B_A | `outputs/efbpt/d13/transfer/eval458/d13_Qwen8B_A.jsonl` | `1ead3c47fbff6dbe2c02996871fb62fa` |
| TR | `outputs/efbpt/d11/student/eval458/d11_armTR.jsonl` | `e38d64114089a88b2deb202de0240ca9` |
| A_r2_eval | `outputs/efbpt/d11/student/eval458/d11_armA_r2_eval.jsonl` | `f139ba9247cdd9fef191f00d404d8212` |

## 2. Rows that E0x newly parses, per arm

| arm | E0 unparsed | newly parsed by E0x | by rule 1 ("جی") | by rule 2 (Devanagari) | by rule 3 (Latin) | newly parsed and right | newly parsed and wrong | E0x unparsed | E0 acc % (correct) | E0x acc % (correct) |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama8B_TR | 118 | 47 | 47 | 0 | 0 | 40 | 7 | 71 | 49.34 (226) | 58.08 (266) |
| Llama8B_A | 188 | 32 | 32 | 0 | 0 | 15 | 17 | 156 | 30.79 (141) | 34.06 (156) |
| Gemma9B_TR | 108 | 45 | 0 | 45 | 0 | 37 | 8 | 63 | 51.97 (238) | 60.04 (275) |
| Gemma9B_A | 81 | 17 | 0 | 17 | 0 | 11 | 6 | 64 | 47.60 (218) | 50.00 (229) |
| Qwen8B_TR | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 64.63 (296) | 64.63 (296) |
| Qwen8B_A | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 40 | 52.62 (241) | 52.62 (241) |
| TR | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 68.78 (315) | 68.78 (315) |
| A_r2_eval | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 54.80 (251) | 54.80 (251) |

## 3. All-rows line under E0 and under E0x

| comparison | scorer | n | acc first % (correct) | acc second % (correct) | delta pp | b | c | raw p |
|---|---|---|---|---|---|---|---|---|
| Llama8B_TR vs Llama8B_A | E0 | 458 | 49.34 (226) | 30.79 (141) | +18.56 | 136 | 51 | 4.067e-10 |
| Llama8B_TR vs Llama8B_A | E0x | 458 | 58.08 (266) | 34.06 (156) | +24.02 | 151 | 41 | 5.345e-16 |
| Gemma9B_TR vs Gemma9B_A | E0 | 458 | 51.97 (238) | 47.60 (218) | +4.37 | 100 | 80 | 0.1565 |
| Gemma9B_TR vs Gemma9B_A | E0x | 458 | 60.04 (275) | 50.00 (229) | +10.04 | 108 | 62 | 0.0005199 |
| Qwen8B_TR vs Qwen8B_A | E0 | 458 | 64.63 (296) | 52.62 (241) | +12.01 | 121 | 66 | 7.027e-05 |
| Qwen8B_TR vs Qwen8B_A | E0x | 458 | 64.63 (296) | 52.62 (241) | +12.01 | 121 | 66 | 7.027e-05 |
| TR vs A_r2_eval | E0 | 458 | 68.78 (315) | 54.80 (251) | +13.97 | 117 | 53 | 1.021e-06 |
| TR vs A_r2_eval | E0x | 458 | 68.78 (315) | 54.80 (251) | +13.97 | 117 | 53 | 1.021e-06 |

The E0 rows are the L1 (all rows) numbers of the committed readouts: the three F2 pairs in `docs/handoff/D13_READOUT.md`, TR vs A_r2_eval in `docs/handoff/D11_EVAL458_READOUT_LINES.md`. In the D13 readout the three F2 tests were decided on other lines (the unparsed and truncation rules fired); those deciding lines and the Holm step are not recomputed here.

No wording is attached to the E0x rows. They are a sensitivity check, not a result.
