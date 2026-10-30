# D8 FULL readout: job 108196

Read-only readout of the completed D8 FULL run. Sources: `logs/d8_reachable_FULL_108196.log` (503 lines), `outputs/efbpt/d8/`, `outputs/efbpt/d1/arms/d1_armA.jsonl`, `outputs/efbpt/d8_job107156_failed/d8_armA_r.jsonl`, `outputs/efbpt/d4/d4_armX1.jsonl` (D4_X1_ref only), `data/strategyqa_official/dev200_seed4242.jsonl` (question_en only). I recomputed every number with Python read from stdin. No other file was written. No edits, commits, or sbatch.

Conventions: "wins" = the first-named arm is right and the second is wrong; "losses" = the reverse. p = exact two-sided binomial on the discordant pairs (`min(1, 2·Σ_{i≤min(b,c)} C(n,i)/2^n)`). I cross-checked it against `scipy.stats.binomtest` and it is identical.

---

## 1. Run status

**Last 3 lines of the log** (line 501 is blank):

```

D8 COMPLETE. Sealed 10 files.
=== end: 2026年 10月 30日 星期五 02:31:34 CST ===
```

**d8_SEAL.json**: mode `FULL`, slurm_job_id `108196`, end_time `2026-10-30 02:31:33`.

| item | MD5 in seal | recomputed now |
|---|---|---|
| freeze `docs/EFBPT_D8_REACHABLE_CEILING_FREEZE.md` | `d95438fac2cb097d38cccdf5db6a3d35` | match |
| Amendment 1 `docs/EFBPT_D8_AMENDMENT_1.md` | `dfde668db216e66ad6a0d13c3d6a5858` | match |
| Amendment 2 `docs/EFBPT_D8_AMENDMENT_2.md` | `19a46f5ee03cb5895bb6e7fe4b07f9ff` | match |
| script `eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py` | `6e5479d9b36d90d5e6ffe79ea7dd886f` | match |

Sealed files (all 10 MD5s, byte counts, and line counts recomputed and matching):

| file | MD5 | bytes | lines |
|---|---|---|---|
| `outputs/efbpt/d8/d8_START.json` | `8eae66b8fcf43afe1fac345efc833e28` | 2367 | 92 |
| `outputs/efbpt/d8/title_map.jsonl` | `287a1e9d924d58605fe449762900ba5e` | 156629 | 636 |
| `outputs/efbpt/d8/d8_extractions_Xp.jsonl` | `7109a109d40e23eb45be11a291254d36` | 141876 | 124 |
| `outputs/efbpt/d8/d8_extractions_GEN.jsonl` | `ece8b915c063e5dc3b900a4c1e34b875` | 192751 | 200 |
| `outputs/efbpt/d8/d8_extractions_X1u.jsonl` | `7427a8f7455c61b66406ed48eecddcb3` | 86158 | 68 |
| `outputs/efbpt/d8/d8_armA_r.jsonl` | `14fdfa23a512a537f30e022be1b9f8dd` | 294283 | 200 |
| `outputs/efbpt/d8/d8_armXp.jsonl` | `904b44eddb6f470d6fbee8d96c9bf5ae` | 281901 | 200 |
| `outputs/efbpt/d8/d8_armGEN.jsonl` | `de90692eef2a05cb5d27a78aea47ec96` | 252726 | 200 |
| `outputs/efbpt/d8/d8_armX1u.jsonl` | `6b9f1d6578422327dc65c5a22579b6a8` | 96476 | 68 |
| `outputs/efbpt/d8/d8_summary.json` | `b2d31ea72df8f55b6d16f1d37ac05e40` | 12424 | 473 |

**SLURM** (`sacct -j 108196`): State `COMPLETED`, ExitCode `0:0`, node `L20001`, Start `2026-10-30T00:13:51`, End `2026-10-30T02:31:34`, Elapsed `02:17:43`. The log's start banner reads 00:13:54. `d8_START.json` start_time is 00:15:40 (written after preflight).

**s/row per pass** (final progress line of each pass; exact = total s / rows):

| pass | arm | rows | total s | s/row (log) | s/row (exact) |
|---|---|---|---|---|---|
| 1 | Xp | 124 | 329.5 | 2.7 | 2.66 |
| 1 | GEN | 200 | 262.9 | 1.3 | 1.31 |
| 1 | X1u | 68 | 270.8 | 4.0 | 3.98 |
| 2 | A_r | 200 | 2293.6 | 11.5 | 11.47 |
| 2 | Xp | 200 | 2294.4 | 11.5 | 11.47 |
| 2 | GEN | 200 | 1802.7 | 9.0 | 9.01 |
| 2 | X1u | 68 | 788.9 | 11.6 | 11.60 |

Pass 1 reported `empty=0 truncated=0` for all three arms. Facts per row: Xp and GEN min=6 median=6 max=6; X1u min=1 median=6 max=6.

---

## 2. Validity

**Verbatim from the log (lines 467–473):**

```
V1 unparsed spread A_r/Xp/GEN: {'A_r': 2.5, 'Xp': 3.5, 'GEN': 2.5} -> 1.00 pp -> PASS
V2 A_r 62.00% vs D1 arm A 57.50% (diff 4.50 pp); row agreement 155/200 = 77.50% -> NONDETERMINISM FLAG (readings use A_r only)
V2b A_r 62.00% vs archived A_r (job 107156) 62.00% (diff 0.00 pp); row agreement 200/200 = 100.00% (reported, not a gate)
V3 GEN Latin fraction 0.9972 (raw 0.9972) -> PASS
V4 Xp qids with >=1 page 124, histogram {0: 76, 1: 96, 2: 24, 3: 4} -> PASS
V5 30 random Xp facts listed in the summary for hand checking (seed 8)
V6 verbatim gold-fact substrings in pass-2 prompts (reported, not a gate): Xp 1 in 1 qid(s), GEN 3 in 3 qid(s), X1u 0 in 0 qid(s)
```

**Recomputed V2 / V2b** (joined on qid; all three files have 200 rows and 200 unique qids, and gold labels agree on all 200):

| pair | A_r acc | other acc | rows agreeing (pred) | identical generations | A_r wins | A_r losses | discordant | p |
|---|---|---|---|---|---|---|---|---|
| A_r (108196) vs D1 arm A | 62.00% (124) | 57.50% (115) | 155/200 = 77.50% | 27/200 | 26 | 17 | 43 | 0.2221 |
| A_r (108196) vs archived A_r (107156) | 62.00% (124) | 62.00% (124) | 200/200 = 100.00% | 200/200 | 0 | 0 | 0 | 1.0 |

- Both recomputations match the V2 and V2b log lines exactly.
- The 108196 `d8_armA_r.jsonl` and the archived 107156 `d8_armA_r.jsonl` are byte-identical (both MD5 `14fdfa23a512a537f30e022be1b9f8dd`).
- D1 arm A MD5 is `481c6ff2d64103e5e0103d677394b74a`.

---

## 3. Readings

**Verbatim from the log (lines 478–487):**

```
  X1u      vs A_r        b=17  c=7    p=0.0639
R1 X1u 52/68 correct (threshold 49/68) -> D4 STANDS: extraction works without question_en; X1u is the honest number
  Xp       vs A_r        b=26  c=33   p=0.4350
R2 Xp - A_r = -3.5 pp, wins 26 losses 33, p = 0.43499280512215344 -> NO-GO for the reachable-page route
  GEN      vs A_r        b=36  c=44   p=0.4340
R3 GEN - A_r = -4.0 pp, wins 36 losses 44, p = 0.43404225534547874 -> HARM: self-generated facts hurt; never used
  Xp       vs A_r        b=23  c=24   p=1.0000
  Xp       vs X1u        b=5   c=12   p=0.1435
Secondary: Xp vs A_r on 124 qids with >=1 page: {"n": 124, "acc_Xp": 62.1, "acc_A_r": 62.9, "delta_pp": -0.8, "wins_Xp": 23, "losses_Xp": 24, "discordant": 47, "p_mcnemar_exact_two_sided": 1.0}
Secondary: Xp on D4-68 vs X1u: {"n": 68, "acc_Xp": 66.18, "acc_X1u": 76.47, "delta_pp": -10.29, "wins_Xp": 5, "losses_Xp": 12, "discordant": 17, "p_mcnemar_exact_two_sided": 0.143463134765625}
```

**Recomputed from the arm jsonl files:**

| comparison | n | first arm correct (acc) | second arm correct (acc) | wins | losses | discordant | exact two-sided p | matches d8_summary.json |
|---|---|---|---|---|---|---|---|---|
| X1u vs A_r (D4-68) | 68 | 52 (76.47%) | 42 (61.76%) | 17 | 7 | 24 | 0.06391465663909912 | yes |
| Xp vs A_r | 200 | 117 (58.50%) | 124 (62.00%) | 26 | 33 | 59 | 0.4349928051221534 | yes |
| GEN vs A_r | 200 | 116 (58.00%) | 124 (62.00%) | 36 | 44 | 80 | 0.4340422553454787 | yes |
| Xp vs A_r on qids with ≥1 page | 124 | 77 (62.10%) | 78 (62.90%) | 23 | 24 | 47 | 1.0 | yes |
| Xp vs X1u (D4-68) | 68 | 45 (66.18%) | 52 (76.47%) | 5 | 12 | 17 | 0.143463134765625 | yes |

Wins, losses, discordant, n, and p all match `d8_summary.json` (p equal to within 1e-12). The 124 page qids taken from `d8_armXp.jsonl` (`n_pages > 0`) are the same set as the 124 qids in `d8_extractions_Xp.jsonl`.

---

## 4. Accuracy table

**Verbatim from the log (lines 490–498):**

```
ACCURACY TABLE   floor 56.50%
====================================================================================
arm        facts                                n     acc%   unpars%  SEC acc%
A_r        none (re-run)                      200    62.00      2.50     62.00
Xp         reachable gold pages, extracted    200    58.50      3.50     58.50
GEN        self-generated English facts       200    58.00      2.50     58.00
X1u        all gold pages, no question_en      68    76.47      1.47     76.47
D1_A_ref   none (D1, reused)                  200    57.50      4.00     57.50
D4_X1_ref  all gold pages + question_en (D4)    68    76.47      0.00     76.47
```

**Per-arm counts** (recomputed from jsonl; D1_A_ref from `d1_armA.jsonl`, D4_X1_ref from `d4_armX1.jsonl` restricted to the 68 X1u qids):

| arm | n | correct | predicted yes (rate) | predicted no | truncated | unparsed |
|---|---|---|---|---|---|---|
| A_r | 200 | 124 | 94 (47.00%) | 101 | 16 | 5 |
| Xp | 200 | 117 | 78 (39.00%) | 115 | 13 | 7 |
| GEN | 200 | 116 | 52 (26.00%) | 143 | 4 | 5 |
| X1u | 68 | 52 | 22 (32.35%) | 45 | 3 | 1 |
| D1_A_ref | 200 | 115 | 77 (38.50%) | 115 | 9 | 8 |
| D4_X1_ref | 68 | 52 | 27 (39.71%) | 41 | 1 | 0 |

All counts agree with the rates in `d8_summary.json` `accuracy_table`. In all four D8 arms, `pred_secondary == pred` on every row, which is why the SEC column equals the primary column.

---

## 5. Xp breakdown

| subset | n | Xp correct (acc) | A_r correct (acc) | Xp wins | Xp losses | p | pred agree | identical generations |
|---|---|---|---|---|---|---|---|---|
| qids with ≥1 page | 124 | 77 (62.10%) | 78 (62.90%) | 23 | 24 | 1.0 | 76/124 | 0/124 |
| qids with no page | 76 | 40 (52.63%) | 46 (60.53%) | 3 | 9 | 0.1460 | 64/76 | 24/76 |

On the 76 no-page rows, Xp had no pass 1 (`n_facts_given = 0`, `had_pass1 = False` on all 76). Its pass-2 prompt comes from the same `answer_prompt(r, [])` call as A_r (`d8_reachable_ceiling.py:677-683`), so on those rows the Xp and A_r prompts are identical. The Xp/A_r differences there therefore cannot come from facts.

- Of R2's net −7 (26 wins / 33 losses), −1 falls on the 124 page rows and −6 on the 76 rows with identical prompts.
- (Inference, not tested.) Identical prompts gave different outputs here (24/76 identical generations), while the same A_r batches in 107156 and 108196 gave byte-identical files (§2). This fits outputs depending on batch composition (`generate_all` groups prompts by length).

**Meta lines.** Regex `not mentioned|no information|does not mention|not provided|not stated|cannot be determined` applied to the kept `facts` of `d8_extractions_Xp.jsonl` (the lines that went into pass 2). The count is the same case-sensitive, case-insensitive, or applied to `raw_extraction`.

- Rows with a meta line: **2 / 124** (one meta line each; 2 lines total; both hit `not mentioned`).
  - `00e13b8d03d35929c049` [Retail]: "Snipt Kip is not mentioned in the passages." (gold no; Xp yes; A_r yes)
  - `d808a0c72dbe605309ab` [Dodgeball]: "Felix Potom is not mentioned in the passage." (gold no; Xp no; A_r no)

| subset | n | Xp correct (acc) | A_r correct (acc) | Xp wins | Xp losses | p |
|---|---|---|---|---|---|---|
| Xp rows with a meta line | 2 | 1 (50.00%) | 1 (50.00%) | 0 | 0 | 1.0 |
| remaining page rows | 122 | 76 (62.30%) | 77 (63.11%) | 23 | 24 | 1.0 |

---

## 6. GEN breakdown

GEN vs A_r on all 200: GEN 116 (58.00%), A_r 124 (62.00%), **wins 36, losses 44**, discordant 80, p = 0.4340. The two arms agree on the prediction for 118/200 rows. Gold is yes on 87/200.

- The 44 losses, by (gold, GEN pred): (yes, no) 33; (no, yes) 6; (no, unparsed) 3; (yes, unparsed) 2.
- The 36 wins, by (gold, A_r pred): (no, yes) 24; (yes, no) 9; (no, unparsed) 2; (yes, unparsed) 1.

**10 random losses (GEN wrong, A_r right).** Selected with `random.Random(11).sample(L, 10)`, where L is the 44 loss qids in `d8_armGEN.jsonl` row order. Facts come from `d8_extractions_GEN.jsonl` and question_en from DEV200.

1. `7f4effbc97ab2b5fd4a7`: gold **yes** (GEN no, A_r yes)
   Q: Could an American confuse breakfast in British cuisine for dinner?
   - The term "breakfast" refers to the first meal of the day.
   - In some cultures, people may eat a late-night meal before sleeping.
   - In the United States, breakfast is typically eaten in the morning.
   - In the United Kingdom, the term "dinner" can refer to the main meal of the day.
   - Some people may confuse the timing of meals in different cultures.
   - The concept of meal times varies across different countries and traditions.

2. `a895b73bdfb2397b2e27`: gold **yes** (GEN no, A_r yes)
   Q: Does 2015 have more unlucky Friday's than usual?
   - Friday the 13th occurred once in 2015.
   - There were 52 Fridays in 2015.
   - The date of Friday the 13th depends on the calendar year.
   - 2015 was not a leap year.
   - The first Friday the 13th in 2015 was on February 13.
   - The last Friday the 13th in 2015 was on December 11.

3. `80aa769f55b14c1e4d8d`: gold **yes** (GEN no, A_r yes)
   Q: Are sables related to wolverines?
   - Sylvie is a character from the Marvel Comics universe.
   - Wolverine is also a character from the Marvel Comics universe.
   - Sylvie and Wolverine first appeared in different comic series.
   - Sylvie is a variant of the character Scarlet Witch.
   - Wolverine is known for his healing factor and adamantium claws.
   - They have not shared a common storyline in the comics.

4. `f8e6087cd52c925fedad`: gold **yes** (GEN no, A_r yes)
   Q: Could ancient Tony Bennett have a baby in 2020?
   - Tony Bennett was born on February 2, 1926.
   - He passed away on July 23, 2023.
   - Bennett married Sandra Williams in 1993.
   - They had a daughter named Nancy Bennett.
   - Nancy Bennett was born in 1994.
   - Tony Bennett was 94 years old when he died.

5. `987260ffef60cb2c5439`: gold **yes** (GEN no, A_r yes)
   Q: Is Pearl Harbor the mythical home of a shark goddess?
   - Perl Harbor is a fictional location in the SpongeBob SquarePants television series.
   - The character SpongeBob SquarePants lives in a pineapple under the sea.
   - Perl Harbor is not an actual place but a part of the show's fictional world.
   - The show is set in the fictional city of Bikini Bottom.
   - SpongeBob's home is often referred to as the "Pineapple Under the Sea."
   - The setting includes various fictional locations like the Krusty Krab and the Chum Bucket.

6. `d8fd81e475df8c2f6972`: gold **no** (GEN unparsed, A_r no)
   Q: Is a Cassowary safer pet than a crane?
   - The German Shepherd is a popular breed known for its intelligence and versatility.
   - It is commonly used in police and military work due to its protective nature.
   - German Shepherds are generally loyal and devoted to their families.
   - They require regular exercise and mental stimulation.
   - They can be protective and may bark to alert their owners.
   - Proper training is essential to ensure they are well-behaved.

7. `315575d5cbdd1d154744`: gold **yes** (GEN no, A_r yes)
   Q: Would Kurt Cobain have benefited from Project Semicolon?
   - Kurt Cobain was the lead singer and guitarist of the band Nirvana.
   - Project Seminal was a fan-made tribute album to Nirvana.
   - Kurt Cobain died by suicide in April 1994.
   - Project Seminal was released in 1995, after Cobain's death.
   - The album included unreleased tracks and demos by Nirvana.
   - Kurt Cobain did not benefit from Project Seminal financially.

8. `2e8e9e51f0f290e32d27`: gold **yes** (GEN no, A_r yes)
   Q: Did any cultures associate celery with death?
   - Wheat is one of the world's oldest cultivated crops.
   - It has been a staple food in many ancient civilizations.
   - In some mythologies, wheat symbolizes life and fertility.
   - Ancient Egyptians associated wheat with the goddess Isis.
   - Wheat was used in religious rituals in various cultures.
   - It was not commonly linked to death in most ancient cultures.

9. `de62d38623fde2c2285a`: gold **yes** (GEN no, A_r yes)
   Q: Would a week be enough time to watch every episode of Ugly Betty?
   - A week is seven days long.
   - Each episode of a series can vary in length.
   - Some episodes may be as short as 20 minutes.
   - Other episodes can last up to an hour or more.
   - Watching one episode per day would take seven days for a seven-episode series.
   - The time required depends on the number of episodes and their length.

10. `8d3ddaee20ad48edc066`: gold **yes** (GEN no, A_r yes)
    Q: Did Brazilian jiu-jitsu Gracie founders have at least a baker's dozen of kids between them?
    - Brazilian Jiu-Jitsu was founded by the Gracie family.
    - The Gracie family originated from Brazil.
    - The Gracie family included multiple brothers.
    - The Gracie brothers were known for their martial arts expertise.
    - The Gracie family had several children.
    - The Gracie family's lineage includes many practitioners of Brazilian Jiu-Jitsu.

---

## 7. V5 sample and V6 matches

**V5** (`d8_summary.json`, seed 8, n = 30, status `PENDING hand check`):

| # | qid | titles_corpus | fact |
|---|---|---|---|
| 1 | `4e1b65e81ec09397b26e` | Giant squid | The giant squid has eight arms and two longer tentacles. |
| 2 | `80aa769f55b14c1e4d8d` | Sable | Sable is primarily found in forest environments of Russia, Siberia, and northern Mongolia. |
| 3 | `86ead3f15417204affc2` | Dustin Hoffman | Dustin Hoffman was born in Los Angeles, California. |
| 4 | `2bc94c6dfc79d7a86fa2` | Christopher Walken | He has guest-hosted Saturday Night Live seven times. |
| 5 | `42c947475244ed02629e` | Oyster | Pearl farmers can culture a pearl by placing a nucleus inside the oyster. |
| 6 | `f7a32d14efd589015fe1` | Herpes simplex virus; Virus | HSV-1 and HSV-2 are very common and contagious. |
| 7 | `068379affae04debca2b` | Nepalese Civil War | The war began on 13 February 1996 with an insurgency by the CPN (Maoist). |
| 8 | `15885efe6f91724c16d9` | Audi R8; Sound barrier | The sound barrier refers to the large increase in aerodynamic drag when an object approaches the speed of sound. |
| 9 | `3295844627a9bd1b9135` | Harry Potter | The total value of the Harry Potter franchise was estimated at $25 billion in 2016. |
| 10 | `5ab66b27a0caee886970` | Miami | Miami has a population of 442,241 as of the 2020 census. |
| 11 | `ae0891799990442fbb0e` | Urban planner | Land use planning involves regulating land use, development, and subdivision to achieve urban planning goals. |
| 12 | `48da75d87c66754ccc2e` | Attack on Pearl Harbor | The attack aimed to prevent the U.S. Pacific Fleet from interfering. |
| 13 | `8d708927b1593d8b9b42` | Plessy v. Ferguson | The decision upheld the "separate but equal" doctrine. |
| 14 | `de62d38623fde2c2285a` | Ugly Betty | Viewership dropped significantly in the show's third season. |
| 15 | `064d8faaa8f164b55270` | Apollo 15; Unicycle | Apollo 15 used the Lunar Roving Vehicle for lunar surface exploration. |
| 16 | `9b491a31c4212e70b18c` | Middle child syndrome | Alfred Adler developed the theory of birth order and its impact on children's personality. |
| 17 | `a2dfceff60d5b18fc70d` | Carolina Reaper | It was developed by American breeder Ed Currie. |
| 18 | `9883d6c816d4855035ea` | Onion; Tree | The bulb of the onion is at the base of the plant. |
| 19 | `8d06b619a7045ed02f51` | Shirley Bassey | She was made a Dame in 1999. |
| 20 | `a54b16376f82a6e14bf9` | Atmosphere; Atmosphere of Earth | The current composition of Earth's atmosphere is the result of billions of years of biochemical modification by living organisms. |
| 21 | `c921bdade889def26cf4` | Isopropyl alcohol; Salt; Solubility | Salt is primarily composed of sodium chloride (NaCl). |
| 22 | `42c947475244ed02629e` | Oyster | Windowpane oysters are harvested for their translucent shells. |
| 23 | `8d708927b1593d8b9b42` | Plessy v. Ferguson | The decision was later weakened by Brown v. Board of Education in 1954. |
| 24 | `16c08e5f354b035c41a5` | The Godfather; USB flash drive | The Godfather stars Marlon Brando, Al Pacino, James Caan, and others. |
| 25 | `a0896de3fd13cd0f3e16` | Atlantic cod | Atlantic cod is labelled vulnerable on the IUCN Red List of Threatened Species. |
| 26 | `557e389d7efe47c5bc4d` | Blue | Blue has been used in art and decoration since ancient times. |
| 27 | `05e9cdc44f1b066badd7` | Moustache | Earliest depictions of moustaches trace back to Ancient Egypt's Old Kingdom era (ca. 2649–2130 B.C.). |
| 28 | `f4a336168f5165de7f0d` | Anesthesia | The risks of complications during or after anesthesia depend on the individual's health. |
| 29 | `690a36d5fed5b17969aa` | Jack Dempsey | Jack Dempsey's career as a professional boxer lasted from 1914 to 1927. |
| 30 | `b3443bf3c312c9d144e4` | Panthéon | The Panthéon was first used as a mausoleum in 1791. |

**V6 matches** (`d8_summary.json`; reported, not a gate):

| arm | count | qids | qid | matching fact |
|---|---|---|---|---|
| Xp | 1 | 1 | `f07a6acf27e25dedbb8d` | Andrew Johnson was the 17th president of the United States. |
| GEN | 3 | 3 | `06215d4b6f1a4b975d9c` | The Space Needle is located in Seattle, Washington. |
| | | | `06ebd3ca1fe98d5796f7` | Douglas Adams was born in 1952. |
| | | | `f07a6acf27e25dedbb8d` | Andrew Johnson was the 17th president of the United States. |
| X1u | 0 | 0 | | |

---

## 8. Log grep

`grep -n -E 'WARNING|Traceback|FATAL|LEAK|VOID' logs/d8_reachable_FULL_108196.log` returns **no matches** (exit 1).

The pattern is case-sensitive, so it does not match the lowercase `[leak-guard]` lines (log lines 72–75, 263, 312, 369, 426). Those lines report every prompt clean, except the pass-2 gold-fact substrings in Xp (1) and GEN (3), which are listed as V6 above.
