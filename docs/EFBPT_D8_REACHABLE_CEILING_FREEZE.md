# D8 — Reachable-page ceiling and English self-knowledge (FROZEN 2026-10-27)

Status: FROZEN before any script or job exists. No change after numbers exist. Extends docs/EFBPT_PLAN_A_FREEZE.md. Same conventions: McNemar exact two-sided on discordant pairs; AMENDMENT 5D unparsed-spread rule; dual scorer.

## 0. Why
D4 showed that facts extracted from the correct Wikipedia pages raise Urdu accuracy. D4 did not measure two things:
(a) its extraction prompt contained the human English question (question_en), which a real Urdu pipeline does not have;
(b) the model got all gold pages, but 52.5% of gold titles have no trace in the question, so a linker working from the question can reach only the other 47.5%.
D8 measures (a) and (b). It also tests whether the model can write the needed facts in English from its own knowledge; the earlier GenRead probe wrote facts in Urdu, which D1 showed is unusable, so it is not evidence either way.

## 1. Corrections recorded here (computed read-only, 2026-10-26/27)
- D2 coverage is 427/613 = 69.66%, not 432/613 = 70.47%. Five gold titles matched all-caps articles (Cactus->CACTUS, David->DAVID, Frog->FROG, Moose->MOOSE, Venus->VENUS). Three D4-71 qids received wrong-article chunks: 1d923b8a2778448a9716, 8eecf2ff459ef5a9b98a, f7a32d14efd589015fe1. The clean fully-covered subset is 68 qids ("D4-68").
- On D4-68: A 41/68 = 60.29%, X1 52/68 = 76.47%, O2 45/68 = 66.18%. X1 vs A: 18 wins, 7 losses (25 discordant pairs), exact two-sided McNemar p = 0.0433.
- D4 and D6 used enable_thinking=False and EXTRACT_MAX_NEW=512; experiments.md:4264-4265 says thinking ON and 2048. D4 pass 1 is not MD5-checked (experiments.md:4271 says it is).
- The D4/D6 extraction prompt contained question_en (d4_extract_facts.py:214, d6_endtoend_selflink.py:349).
- D6 stage-0 parse: ok=66, json_error=5 (experiments.md:4560 says 71/0). No accuracy number changes.
- D6 stage-0 recall on traced gold titles (D4-71): exact 37/96, loose 49/96; on all gold titles: exact 42/200, loose 70/200.
- The probes probe_strategyqa_genread_gate_test50.py and probe_strategyqa_factretrieval_test50.py used the first 50 rows of eval458 (answer-free probing, no tuning; 0 overlap with DEV200). To be disclosed; eval458 stays the final test set.

## 2. Data (all opened read-only; eval458 is not touched)
- DEV200: data/strategyqa_official/dev200_seed4242.jsonl (assert 200 rows).
- D4-68 as defined in §1 (assert 68).
- Lexical trace: data/strategyqa_official/efbpt/stage0/lexical_trace_instances.jsonl, git a6b0e50, MD5 5b361d60b33a64ab5c8935b40a3550e1; assert 636 instances, EXACT 256, PARTIAL 46, ABSENT 334.
- Corpus: rag/index/wikipedia_full_meta.jsonl, unchanged.
- Reused: outputs/efbpt/d1/arms/d1_armA.jsonl, outputs/efbpt/d4/d4_armX1.jsonl.
- Traced-and-present under §3 (computed before the run): 190/302 instances present; 136 DEV200 qids with >=1 page.

## 3. Title matching rule (replaces norm()-only matching for D8 and all later work)
A gold title T is present if (i) a corpus title equals T exactly; else (ii) exactly one corpus title equals T after norm() and that corpus title is not all upper-case. Otherwise absent. Every decision (gold title, corpus title or ABSENT, rule used) is written to outputs/efbpt/d8/title_map.jsonl.

## 4. Model and answer pass — identical to D4
HF transformers, Qwen3-14B nf4 double-quant exactly as d4_extract_facts.py; batch size 8; greedy (do_sample=False); enable_thinking=False; answer pass max_new_tokens 1024; Urdu instruction prompts/efbpt/plan_a_instruction_ur.txt, MD5 f3b58d766fe3ec2573ff4f24761cf0c9 asserted; facts block layout as D1 arm B / D4; same dual scorer (primary Perso-Arabic, secondary Devanagari).

## 5. Arms
| arm | qids | pass 1 (facts) | pass 2 |
|---|---|---|---|
| A_r | 200 | none | arm A re-run in the same job (control for batch effects) |
| Xp | 200 | extraction from gold pages with trace EXACT or PARTIAL and present (§3); <=3 chunks per title in file order; D4 system message and EXTRACT_INSTRUCTION unchanged; "Question: " + question_ur; EXTRACT_MAX_NEW 512; thinking off. A qid with 0 pages gets no pass 1 and is answered exactly like A_r | facts + question_ur |
| GEN | 200 | the model writes facts from its own knowledge. Instruction (frozen): "Write the facts needed to answer the question below.\nRules:\n1. Output ONLY short English factual sentences, one per line.\n2. No numbering, no bullets, no explanation, no answer to the question itself. Facts only.\n3. At most 6 facts." then "\n\nQuestion: " + question_ur. D4 system message. thinking off; 512 tokens; at most 6 facts kept | facts + question_ur |
| X1u | 68 | all gold pages (§3), extraction exactly as Xp | facts + question_ur |

Forbidden in any prompt: question_en, gold facts, evidence ids, gold answer, any title beyond the arm's own pages. The script asserts that no rendered prompt contains the question_en string of any row.

## 6. Validity — checked and logged in this order, before any accuracy is read
V1 unparsed spread across A_r, Xp, GEN <= 5pp; else the comparison is VOID.
V2 A_r vs D1 arm A: accuracy within 2pp and row-level agreement >= 95%; else a nondeterminism flag is raised, readings use A_r only, and the flag is reported.
V3 GEN facts: >= 90% of letters Latin (digits and punctuation excluded); else GEN is VOID.
V4 Xp: qids with >=1 page and the pages-per-qid histogram are reported; must equal the pre-run count in §2.
V5 30 random Xp facts hand-checked against their chunks for invention (as D4). Reported, not a gate.

## 7. Readings (pre-declared; all paired with A_r on the same qids)
R1 question_en confound (D4-68): X1u >= 49/68 -> extraction works without the English question; D4 stands, with X1u as the honest number. X1u < 49/68 -> D4 is overstated; X1u replaces X1 everywhere, and D8b (model-translated English question in the extraction prompt) is declared and run before any other step.
R2 reachable ceiling (200): Xp - A_r >= +8pp and p < 0.05 -> GO: build the linker (stage-0 titles -> exact page) and run end-to-end on DEV200 as D9. +5pp <= delta < +8pp, or p >= 0.05 -> CONDITIONAL: proceed only after the replacement corpus raises the traced-and-present count, and re-run Xp on it before D9. delta < +5pp -> NO-GO for the reachable-page route.
R3 English self-knowledge (200): GEN - A_r >= +8pp and p < 0.05 -> viable component; D9 combines page facts with GEN facts for unreached questions. -3pp <= delta < +8pp or p >= 0.05 -> the model cannot supply the missing knowledge in English either; closes the GenRead objection. delta < -3pp -> self-generated facts harm; never used.
Secondary, reported without claim: Xp vs A_r on the subgroup with >=1 page (matched control, same qids); Xp restricted to D4-68 vs X1u (effect of fewer pages alone).
Power: with about 50 discordant pairs of 200, +8pp gives p about 0.034 and +7pp about 0.066, so +8pp is the smallest declared gate. The D4-68 arms are diagnostics and support no method claim.

## 8. Decision after D8
GO on R2 -> D9: linker + end-to-end on DEV200, gate >= +8pp and p < 0.05; then one frozen run on eval458, gate >= +5pp and p < 0.05, reported on eval458 and on the 408 never-probed rows.
NO-GO on R2 and GO on R3 -> D9 is the GEN pipeline.
NO-GO on both -> remaining pathway: a larger English fact source (teacher model) feeding the Qwen3-14B Urdu answer pass; declared in a separate freeze before any run.

## 9. Execution
Script eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py, importing from d4_extract_facts.py and d3_oracle_retrieval.py without editing them. All outputs under outputs/efbpt/d8/ only: d8_START.json (MD5 of this file, the script, both pass-1 instructions, the Urdu instruction, the trace file; package versions; job id), title_map.jsonl, d8_extractions_{Xp,GEN,X1u}.jsonl, d8_arm{A_r,Xp,GEN,X1u}.jsonl, d8_summary.json (readings in the §6 order), d8_SEAL.json. TEST-5 (--limit 5 with --dry-run, per repo rule) before the full run. Files are fsync'd and line counts asserted. Nothing under outputs/efbpt/d1, d4 or d6 is opened for writing.
