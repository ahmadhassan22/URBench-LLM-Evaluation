# D8 FULL readout — job 107156

Read-only readout written 2026-10-29. Sources: `logs/d8_reachable_FULL_107156.log`, `outputs/efbpt/d8/`, `outputs/efbpt/d1/arms/d1_armA.jsonl`, `data/strategyqa_official/dev200_seed4242.jsonl`, `sacct -j 107156`.

**Headline: the run did NOT complete.** It died with `FATAL: LEAK` at the start of PASS 2 arm Xp (log line 311). Only pass 1 (Xp, GEN, X1u extractions) and pass 2 arm A_r finished. There is no `d8_SEAL.json`, no `d8_summary.json`, and no `d8_armXp.jsonl`, `d8_armGEN.jsonl` or `d8_armX1u.jsonl`. Every item below that depends on the summary or on the missing arm files is marked NOT AVAILABLE; everything that can be recomputed from the files on disk has been recomputed.

---

## 1. Run status

### Last 3 lines of the log (lines 309–311)

```
PASS 2 — arm Xp  (n=200, max_new=1024, frozen Urdu prompt)
======================================================================
FATAL: LEAK in Xp/pass2 prompt 186: contains gold fact 'Andrew Johnson was the 17th president of the United States.'
```

### d8_SEAL.json

Does NOT exist. Only `d8_SEAL_TEST.json` (from the TEST dry run, 2026-10-28 00:06) is present.

### Line counts, every non-_TEST file under outputs/efbpt/d8/

| lines | file |
|---|---|
| 200 | d8_armA_r.jsonl |
| 200 | d8_extractions_GEN.jsonl |
| 68 | d8_extractions_X1u.jsonl |
| 124 | d8_extractions_Xp.jsonl |
| 86 | d8_START.json |
| 636 | title_map.jsonl |

Absent (never written): d8_armXp.jsonl, d8_armGEN.jsonl, d8_armX1u.jsonl, d8_summary.json, d8_SEAL.json.

### MD5s in d8_START.json

| artifact | MD5 in d8_START.json | expected prefix | match |
|---|---|---|---|
| freeze `docs/EFBPT_D8_REACHABLE_CEILING_FREEZE.md` | d95438fac2cb097d38cccdf5db6a3d35 | d95438fa… | yes |
| amendment `docs/EFBPT_D8_AMENDMENT_1.md` | dfde668db216e66ad6a0d13c3d6a5858 | dfde668d… | yes |
| script `eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py` | 456c3e3a87947a1db1bccc72e9659260 | 456c3e3a… | yes |

The log preflight (lines 27–29) prints the same three MD5s. Also in d8_START.json: urdu_instruction MD5 f3b58d766fe3ec2573ff4f24761cf0c9; trace MD5 5b361d60b33a64ab5c8935b40a3550e1 (636 instances: EXACT 256, PARTIAL 46, ABSENT 334); model `Qwen/Qwen3-14B`, nf4 double-quant bf16 compute, batch 8, greedy, thinking off; extract_max_new 512, gen_max_new 512, answer_max_new 1024, chunks_per_title 3, gen_max_facts 6; title_map: 613 distinct gold titles, 427 present, 124 qids with ≥1 page, pages histogram {0: 76, 1: 96, 2: 24, 3: 4}.

### SLURM job id, start, end, runtime

| field | value |
|---|---|
| SLURM job id | 107156 (`d8_reachable_FULL`, node L20002) |
| sacct State / ExitCode | FAILED / 1:0 |
| sacct Start | 2026-10-28T00:16:08 |
| script `=== start` (log line 23) | 2026-10-28 00:16:11 CST |
| `[START]` / d8_START.json start_time | 2026-10-28 00:21:38 (after a 324 s corpus scan in preflight) |
| sacct End | 2026-10-28T01:18:31 |
| log mtime / d8_armA_r.jsonl mtime | 2026-10-28 01:18:30 |
| sacct Elapsed | 01:02:23 |

### Time per generation per pass (from the log; batch size 8, wall seconds per row)

| pass | n | total time | s/row | log line |
|---|---|---|---|---|
| model load (8 shards) | — | 3 min 11 s | — | 77 |
| PASS 1 Xp | 124 | 328.2 s | 2.6 | 126 |
| PASS 1 GEN | 200 | 263.4 s | 1.3 | 184 |
| PASS 1 X1u | 68 | 270.9 s | 4.0 | 246 |
| PASS 2 A_r | 200 | 2276.6 s | 11.4 | 303 |
| PASS 2 Xp | 200 | 0 (died before first batch) | — | 311 |
| PASS 2 GEN, X1u | — | not reached | — | — |

---

## 2. Validity

### V1–V5 lines, verbatim from the log

Only V3 is in the log. V1, V2, V4, V5 are computed in the summary stage, which was never reached.

- V1: NOT IN LOG (not reached)
- V2: NOT IN LOG (not reached)
- V3 (line 257):
  ```
  [V3] GEN kept facts: Latin letters 58278 / 58443 = 0.9972 (raw output 0.9972)
  ```
- V4: NOT IN LOG (not reached)
- V5: NOT IN LOG (not reached)

### V2 block from d8_summary.json

NOT AVAILABLE — d8_summary.json does not exist.

### V2 row agreement, recomputed (d8_armA_r.jsonl ⋈ d1/arms/d1_armA.jsonl on qid)

| quantity | value |
|---|---|
| qids joined | 200 / 200 |
| A_r correct | 124 / 200 = 62.00% |
| D1 arm A correct | 115 / 200 = 57.50% |
| accuracy diff (A_r − D1 A) | +9 rows = +4.50 pp |
| rows with identical pred | 155 / 200 = 77.5% |
| rows disagreeing | 45 |
| rows with byte-identical generation text | 27 / 200 |

Of the 45 disagreements, A_r is right and D1 A wrong in 26 (3 of these where D1 A was unparsed), D1 A right and A_r wrong in 17 (2 of these where A_r was unparsed), and both wrong in 2 (a2dfceff60d5b18fc70d, dabf94934721c26b2422). Net +9 for A_r.

The 45 disagreeing qids (A_r pred, D1 A pred, gold):

| qid | A_r pred | D1_A pred | gold |
|---|---|---|---|
| 064d8faaa8f164b55270 | yes | no | yes |
| 06f7878425a995c2a633 | yes | no | no |
| 0d22526961c82ad6ef4a | yes | no | yes |
| 1534ef2130cb48d6abb4 | yes | no | yes |
| 20abc7ba538a2fd1dd76 | no | yes | no |
| 24418db1d18a7fdf79f3 | yes | no | yes |
| 265dd54c248f8b048851 | no | None | no |
| 310968736449680ee61e | yes | no | no |
| 3295844627a9bd1b9135 | yes | None | yes |
| 3503e8d41364eb0b33bc | None | no | no |
| 35a5b27d84294ebf1fe0 | yes | no | yes |
| 3bba4ea27500361cb141 | no | yes | no |
| 3de88c80f0914eb08caf | yes | no | no |
| 3f5af97fdff20028347b | yes | no | no |
| 3f8a1bd6bf3a967cdeb6 | no | yes | no |
| 42c947475244ed02629e | yes | no | yes |
| 4589898e4852e5389728 | yes | no | yes |
| 5897ec22db850f7b416e | yes | no | no |
| 5ab66b27a0caee886970 | no | yes | no |
| 5f24cbd41798ed5c7608 | no | yes | no |
| 63fb8f40ca6c2226dd01 | yes | no | yes |
| 684d4c03be354635b80f | yes | no | no |
| 70b2122323424f6e80c2 | no | yes | no |
| 71ca1a8a75c5c582ab3f | no | None | no |
| 7dc74fd9b62e514750c2 | yes | no | yes |
| 867a5425c26092b30fbf | yes | no | no |
| 91917c5e30dbf8552b34 | yes | no | no |
| 9247b5afb2eef4c9f95b | no | yes | yes |
| 9384dfc0aabc73026925 | yes | no | yes |
| 9b491a31c4212e70b18c | no | yes | no |
| 9bb59488a227a5f5cd62 | yes | no | no |
| 9cfa675eeebfbbdb2487 | yes | no | no |
| 9e78c01675a198781cca | yes | no | yes |
| a2dfceff60d5b18fc70d | no | None | yes |
| a8bbf2e84d0649b1c52c | yes | no | no |
| acb7edac010554509519 | None | yes | yes |
| b77170e15c0b1bd72340 | no | yes | no |
| b8df6ac181c4c5b280f9 | yes | no | no |
| cac049d9ec0b86a7f911 | yes | no | no |
| d1dbb41001ab4ad3b924 | yes | no | no |
| d9aeae10998c093c0cc9 | no | yes | no |
| dabf94934721c26b2422 | yes | None | no |
| f087d393e2ab1f8783f5 | no | yes | no |
| f8e6087cd52c925fedad | yes | no | yes |
| fba624d8c01833419760 | yes | no | yes |

Note: 77.5% pred agreement and only 27/200 identical generations between A_r and D1 arm A, both run with the same frozen prompt, greedy, nf4. This is the number the V2 check in the (unreached) summary would have evaluated against its threshold; whether it passes depends on the threshold in the freeze, which this readout does not interpret.

---

## 3. Readings

R1, R2, R3 and both secondary readings are computed in the summary stage from the Xp, GEN and X1u pass-2 arm files. None of those exist.

- R1 (X1u correct count, wins/losses/p vs A_r): NOT AVAILABLE — no d8_armX1u.jsonl.
- R2 (Xp vs A_r delta, wins, losses, discordant, p): NOT AVAILABLE — no d8_armXp.jsonl.
- R3 (GEN vs A_r delta, wins, losses, p): NOT AVAILABLE — no d8_armGEN.jsonl.
- Secondary readings: NOT AVAILABLE — same reason (pred_secondary exists only for A_r, where it equals pred on all 200 rows).

Recomputation of wins/losses and McNemar p: NOT POSSIBLE — there is only one pass-2 arm on disk (A_r), so no paired comparison can be formed. Nothing to match against the summary.

---

## 4. Accuracy table

Verbatim accuracy table from the log/summary: NOT AVAILABLE (the table is printed in the summary stage). The only accuracy line in the log is for A_r (line 305):

```
  accuracy 62.00%  unparsed 2.50%  (124/200 correct)
```

Recomputed per-arm detail from the arm files on disk (only A_r exists):

| arm | n | correct | acc | SEC acc | pred yes | pred no | unparsed | pred-yes rate (all / of parsed) | truncated | gold yes | gold no |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_r | 200 | 124 | 62.00% | 62.00% | 94 | 101 | 5 (2.50%) | 47.0% / 48.2% | 16 (8.0%) | 87 | 113 |
| Xp | — | — | — | — | — | — | — | — | — | — | — |
| GEN | — | — | — | — | — | — | — | — | — | — | — |
| X1u | — | — | — | — | — | — | — | — | — | — | — |

A_r rows all have n_facts_given 0, n_pages 0, had_pass1 False (as designed). The gold split of DEV200 is 87 yes / 113 no (43.5% yes).

---

## 5. Xp detail (from d8_extractions_Xp.jsonl, 124 qids with pages)

| quantity | value |
|---|---|
| extractions with 0 facts | 0 (all 124 rows have exactly 6 facts; truncated 0) |
| extracted-fact lists containing a meta line (regex `not mentioned\|no information\|does not mention\|not provided\|not stated\|cannot be determined`) | 2 rows, 2 lines total — qids 00e13b8d03d35929c049 ("Snipt Kip is not mentioned in the passages."), d808a0c72dbe605309ab |
| Xp vs A_r wins/losses on qids WITH meta line | NOT AVAILABLE (no Xp arm). A_r alone: 1/2 correct |
| Xp vs A_r wins/losses on qids WITHOUT meta line | NOT AVAILABLE (no Xp arm). A_r alone: 77/122 correct |
| X1u meta-line count | 0 rows, 0 lines (68 rows; 67 with 6 facts, 1 with 1 fact; truncated 0) |

### The LEAK itself (explains the FATAL)

Prompt index 186 in both A_r/DEV200 order and Xp order is qid f07a6acf27e25dedbb8d, question_en "Does Andrew Johnson's presidential number exceed Elagabalus's Emperor number?", gold no, gold facts `['Andrew Johnson was the 17th president of the United States.', 'Elagabalus was the 25th Roman Emperor.']`.

The Xp pass-1 extraction for that qid (first fact) is byte-identical to the first gold fact:

```
Andrew Johnson was the 17th president of the United States.
Andrew Johnson served as president from 1865 to 1869.
Andrew Johnson was vice president under Abraham Lincoln.
Andrew Johnson was impeached by the House of Representatives in 1868.
Andrew Johnson was acquitted in the Senate by one vote.
Andrew Johnson was born into poverty and never attended school.
```

The pass-1 prompts were leak-guard clean (log lines 71–74). The string entered the pass-2 prompt via the model's own extraction from the corpus page, and the leak guard (`leak_guard`, substring match over all 532 gold facts) treats that as fatal. Scanning all extraction files for substring matches against the 532 gold facts:

| arm | extracted facts that equal/contain a gold fact |
|---|---|
| Xp | 1 — qid f07a6acf27e25dedbb8d (above) |
| GEN | 3 — 06215d4b6f1a4b975d9c "The Space Needle is located in Seattle, Washington."; 06ebd3ca1fe98d5796f7 "Douglas Adams was born in 1952."; f07a6acf27e25dedbb8d "Andrew Johnson was the 17th president of the United States." |
| X1u | 0 |

So even if Xp pass 2 had passed, GEN pass 2 would have hit the same fatal on 3 prompts. X1u pass 2 would have been clean.

---

## 6. GEN detail (from d8_extractions_GEN.jsonl, 200 rows)

Facts-per-row histogram: {6: 200}. Zero-fact rows 0, truncated 0. 2 rows contain a meta line (regex above).

10 random GEN fact lists, `random.seed(7); random.sample(rows, 10)` over the 200 rows in file order, with qid, question_en (from dev200_seed4242.jsonl) and gold:

**qid 5e2d9519c7c521411817 — gold yes — Would a TMNT coloring book have pizza in it?**
- TNT is a type of explosive material.
- TNT stands for Trinitrotoluene.
- TNT is commonly used in military and industrial applications.
- TNT is a yellow crystalline solid.
- TNT is highly explosive when detonated.
- TNT is not related to food or pizza.

**qid 2e508efbf0b72f1af2c2 — gold yes — Could Quartz be useful to humans if plants died off and there was no oxygen?**
- Quartz is a mineral composed of silicon and oxygen.
- It does not produce oxygen or support life processes.
- Plants are necessary for oxygen production through photosynthesis.
- Quartz does not have nutritional value for humans.
- Humans require oxygen to survive.
- Quartz is not a source of food or oxygen.

**qid 7b84d2bc643ddc2085f0 — gold no — WIll Noah's Ark hypothetically sail through flooded Lincoln Tunnel?**
- The Great Flood in the story of Noah is a religious and mythological event.
- The Suez Canal is a man-made waterway connecting the Mediterranean Sea to the Red Sea.
- The Suez Canal was completed in 1869.
- Noah's Ark is described in religious texts, not in historical or scientific records.
- The Suez Canal is not mentioned in the biblical account of Noah's Ark.
- The dimensions of Noah's Ark are not scientifically verified.

**qid cac8e6300a98cf6128af — gold no — Is number of different US President's in 1800s a lucky number in Hong Kong?**
- The number 1800 refers to the year 1800.
- The number 1800 is not a commonly recognized lucky number in Hong Kong.
- Lucky numbers in Hong Kong are often associated with cultural and linguistic meanings.
- The number 8 is considered a lucky number in Chinese culture.
- The number 4 is considered an unlucky number in Chinese culture.
- The U.S. had different presidents during the 1800s.

**qid 076f3f72b7e6d17073eb — gold no — Can Amtrak's Acela Express break the sound barrier?**
- The IM Track is a type of railway line in Pakistan.
- The ISLAMABAD-RAWALPINDI Express is a train service in Pakistan.
- The ISLAMABAD-RAWALPINDI Express runs between Islamabad and Rawalpindi.
- The IM Track is part of the railway network connecting major cities in Pakistan.
- The ISLAMABAD-RAWALPINDI Express is known for its speed and reliability.
- The IM Track is used for both passenger and freight services.

**qid 15885efe6f91724c16d9 — gold yes — Is sound barrier too much for Audi R8 V-10 Plus to break?**
- Audi A8 L is a luxury sedan produced by Audi.
- It features a high-end audio system with advanced sound technology.
- The vehicle includes noise-cancelling technology for improved cabin acoustics.
- Sound insulation is a standard feature in Audi A8 L models.
- Audi A8 L uses active noise control to reduce unwanted sounds.
- The car's design minimizes external noise intrusion.

**qid a0896de3fd13cd0f3e16 — gold no — Is Atlantic cod found in a vegemite sandwich?**
- Atlantic cod is a species of fish found in the Atlantic Ocean.
- Veggie sandwiches are typically made with plant-based ingredients.
- Atlantic cod is not a plant-based ingredient.
- Veggie sandwiches may include various types of vegetables and plant-based proteins.
- Some veggie sandwiches may use imitation meats or other non-fish products.
- Atlantic cod is commonly used in seafood dishes, not in vegetarian meals.

**qid 1d923b8a2778448a9716 — gold no — Should cactus soil always be damp?**
- Cacti are desert plants that require well-draining soil.
- Overwatering can lead to root rot in cacti.
- Cacti store water in their stems and leaves.
- The soil for cacti should be allowed to dry out between waterings.
- Cacti do not thrive in consistently moist soil.
- Proper drainage is essential for the health of cacti.

**qid 70b2122323424f6e80c2 — gold no — Would LeBron James hypothetically glance upwards at Yuri Gagarin?**
- Lionel Messi is a professional footballer from Argentina.
- Yuri Gagarin was the first human to travel into space.
- Lionel Messi has never been to space.
- Yuri Gagarin died in 1968.
- Lionel Messi was born in 1987.
- Yuri Gagarin was born in 1934.

**qid b3443bf3c312c9d144e4 — gold no — Is there a full Neptunian orbit between the first two burials of women in the Panthéon?**
- The Pantheon is an ancient Roman temple in Rome, Italy.
- It was originally built as a temple to all the gods of the Roman pantheon.
- The Pantheon's dome is considered an architectural marvel.
- The Pantheon has a circular central space with an oculus at the top.
- The Pantheon was converted into a church in the 7th century.
- The Pantheon's design includes a large central dome supported by pendentives.

Observation (not a reading): 6 of the 10 samples show the model misreading the Urdu entity (TMNT→TNT, Lincoln Tunnel→Suez Canal, Acela→Pakistani railway, R8→A8 L, vegemite→veggie, LeBron→Messi, Panthéon→Roman Pantheon). This is the GEN-from-Urdu-question failure mode; it is visible in pass 1 alone.

---

## 7. WARNING / Traceback / die / LEAK / VOID strings in the log

`grep -n -E 'WARNING|Traceback|die|LEAK|VOID'` over the 311-line log returns exactly one line:

```
311:FATAL: LEAK in Xp/pass2 prompt 186: contains gold fact 'Andrew Johnson was the 17th president of the United States.'
```

No WARNING, no Traceback, no "die", no VOID. (Line 110 carries a transformers notice, "The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']", which does not match any of the requested strings.)
