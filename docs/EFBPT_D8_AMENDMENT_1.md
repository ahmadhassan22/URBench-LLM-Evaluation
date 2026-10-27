# D8 — Amendment 1 (FROZEN 2026-10-27, pre-run)

Status: amends docs/EFBPT_D8_REACHABLE_CEILING_FREEZE.md before any D8 script
has been submitted and before any D8 number exists. Written after the
read-only mapping audit docs/handoff/D8_MAPPING_AUDIT.md (2026-10-27). The
freeze stays in force except where this file says otherwise. No change after
numbers exist.

## A1. Xp page set: trace EXACT only

Xp uses the gold titles whose lexical trace class is EXACT and which are
present under freeze §3. PARTIAL instances are dropped from Xp.

Reason (audit §2a). Of the 34 PARTIAL-and-present pages, 22 are matched on one
shared common word and are not named in the question ("year", "bee", "milk",
"color", "dynasty", ...); 10 are the wrong subject for the question, and 2 of
those 10 are wrong in the gold annotation itself (Miami River (New York) for
the swim-to-Miami question, USS President (1800) for the US-presidents
question). A question-driven linker cannot reach a page that the question does
not name, so such pages do not belong in a reachability ceiling.

Consequences. Xp is a lower bound on what a linker can reach: alias forms that
the trace rule scores PARTIAL (for example "Home Depot" -> "The Home Depot",
"CW" -> "The CW", "Temple of Doom" -> "Indiana Jones and the Temple of Doom")
are excluded even though a linker could plausibly resolve them. X1u is
unchanged (all gold pages of the D4-68, present under §3). Readings R1-R3,
their thresholds and the validity checks V1-V5 in freeze §6-§7 are unchanged;
V4 now compares against the counts in A2.

## A2. Pre-run counts (replace freeze §2 "traced-and-present")

Computed read-only on 2026-10-27 by applying build_title_map (freeze §3) from
eval/error_analysis_tests/efbpt/d8_reachable_ceiling.py to the 613 distinct
DEV200 gold titles (427 present), restricted to trace class EXACT:

| quantity | value |
|---|---|
| EXACT instances | 256 |
| EXACT-and-present instances | 156 |
| DEV200 qids with >= 1 EXACT-and-present page | 124 |
| pages-per-qid histogram | {0: 76, 1: 96, 2: 24, 3: 4} |

The script asserts all four in every mode (preflight, TEST, FULL). The
previous §2 values (302 traced, 190 present, 136 qids, {0:64, 1:93, 2:34,
3:7, 4:2}) are superseded. The trace file, its MD5 and its class counts
(636; EXACT 256, PARTIAL 46, ABSENT 334) are unchanged.

## A3. GEN system message

The GEN pass-1 system message is replaced by:

    You are a careful research assistant. You write accurate English facts from your own knowledge. You never state a fact you are not confident about.

Reason: the D4 system message ("You extract facts from provided passages. You
never invent information that is not in the passages.") forbids exactly what
GEN is asked to do, and GEN has no passages. The GEN user instruction in
freeze §5 is unchanged. Xp and X1u keep the D4 system message and the D4
EXTRACT_INSTRUCTION unchanged. The MD5 of the GEN system message is recorded
in d8_START.json next to the other pass-1 instruction MD5s.

## A4. FULL wall time

The FULL job requests 08:00:00 (was 04:00:00). At the D4 rates (about 4 s per
512-token extraction row, about 3.3 s per 1024-token answer row, plus model
load and two corpus passes) the 1,072 generations of the FULL run
(pass 1: 124 + 200 + 68; pass 2: 200 + 200 + 200 + 68, with a safety factor
for longer Xp/X1u prompts) take about 5-6 h. The partition q_intel_share_L20
has MaxTime=UNLIMITED. TEST stays at 01:30:00.

## A5. Carried to D9, no D8 change

Recorded here so they are not lost; none of them changes D8:

- Trace rule artefact (audit §3): the head token is taken from a parenthetical
  disambiguator, so "Racket (sports equipment)", "Rock (geology)",
  "Crane (bird)", "The Matrix (franchise)" score ABSENT although the bare
  title is in the question. About 10% of ABSENT instances in the audit sample.
  A D9 linker should strip "( ... )" before matching.
- Corpus loss (audit §4): 112 of the 302 traced instances have no page under
  §3, and almost none has a page under any other name; the missing set is
  dominated by core Wikipedia articles (Beach, Heart, Human, Oxygen, Week,
  Year, England, Television, Eiffel Tower, French Revolution, ...). Only a
  replacement corpus recovers these.
- Raw wikitext (audit §2b): the first chunk of "Darth Vader" is an infobox in
  wikitext, not prose; the extractor receives it as one of its three chunks.
- Page depth (audit §5): the first 3 chunks are about 19% of a page on
  average (median 28 chunks per page); 85% of D4-68 gold facts have at least
  half their keywords in those chunks, 25% have all of them.

## Execution

Script and sbatch updated in the same change as this file. The script dies if
this file is missing and records its MD5 in d8_START.json and d8_SEAL.json
next to the freeze MD5. TEST-5 (--limit 5 --dry-run) before FULL, as in
freeze §9.
