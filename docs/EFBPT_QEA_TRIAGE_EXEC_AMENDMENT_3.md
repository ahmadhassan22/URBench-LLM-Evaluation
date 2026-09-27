# QEA triage: amendment 3 (clarified review prompt and paired review calibration packet)

**Development note, 2026-09-27. Prepared offline; not executed.**
- No model was loaded and no inference or job ran.
- Nothing here edits:
  - the QEA protocol revision 0.2 or amendments 1–2;
  - any seal, including jobs 96151, 96317 and 96350;
  - any historical label;
  - the acceptance rule or reviewer blinding.
- Scope: `POST_OUTCOME_AI_ASSISTED_DEVELOPMENT`. The trigger was the job-96350 audit, where the q1 wording was thinner
  than the Stage-0 definition.

## 1. Review prompt versions

| Version | Constant | SHA-256 | Status |
|---|---|---|---|
| v1 | `REVIEW_SYSTEM` | `5c656e40dee28fcb44b2787662c5f33f7332cc8a625f7aee24d80f6a0f9399cf` | Frozen and unchanged; still used by T4, R1, R2 and smoke-gemma |
| v2 | `REVIEW_SYSTEM_V2` | `59e53e83ee05a1a5fb9378ce343b3cdcd2c5c012a198a240d66076623ee9fd16` | Calibration only; not approved for production |

v2 differs from v1 only in:
- the added definition block;
- the q1 and q2 sentences.

The framing sentences, q3, q4, the blinding and the output format are v1's own text. The exact sentence-level diff is
in `logs/efbpt_qea_amendment3_review_prompt.diff`.

**Definition.** v2 quotes the Stage-0 freeze §6.1 definition verbatim. That quotation contains the frozen word
"reasonably"; nothing was added to it.

**Also from the freeze:**
- all five permitted mappings;
- the `DIRECT_SPECIFIC_CONCEPT` exclusions;
- the §6.4 demonym, broad-page, disambiguation and decomposition-implied boundaries.

**New statements:**
- The question is Urdu and the titles are English.
- q1 is judged from the question and the proposed parent title only; child fields and usefulness cannot establish it.
- q2 uses the same meaning, applied to the child, with its negative polarity kept: Y means the child's direct mapping
  is absent.
- Y, N and U are defined:
  - Y: a permitted mapping supports direct identification.
  - N: the required mapping is absent.
  - U: insufficient understanding, or unresolved page identity.

**Rollout:** if a version is approved, R1 and R2 must both receive it through T4. Switching production is a separate,
later step.

## 2. Calibration packet

The packet is at `outputs/efbpt/qea_review_calibration_v1/`, created with `calib-prepare`.

| Part | Seal SHA-256 | Contents |
|---|---|---|
| `packet/` | `353611eab1016248c20bb9925fc78647d9c7fe7d42f0d87f82fdf00981335d43` | Case payloads, call plan and both prompt texts; no labels or generator claims |
| `labels/` | `1412ee53537a20ba38e1575cbdf2f6bbf9e5ad0d2425019b0a72dbf4addce3a3` | Expected labels; linked to the packet seal; never read by the inference stage |

**Cases (51):**
- **H01–H41:** all 41 human-verified DEV200 candidate pairs, in 30 questions (36 accepted, 5 rejected).
  - Their packets are built by the same `build_packet` path as the smoke. H02 and H04 reproduce the pinned smoke
    payloads.
  - Every historical q1 is Y. The five rejections are all `Y,Y,N,O` (rejected on q3/q4), so none is a q1-negative
    reference.
  - No qid appears in the development or reserve qid lists.
  - Labels are copied verbatim from `human_verified_candidates.jsonl`.
- **C01–C10:** `AI_AUTHORED_DIAGNOSTIC_CONTROLS_NOT_HUMAN_GOLD`, scored on q1 only. There are 4 expected Y, 4 N and
  2 U, in five matched groups. Within a group, only the proposed parent differs, and no step names an N or U parent.
  Section 4 lists them.

**Call plan:** 102 calls, each case v1 then v2, in case order H01…C10. Budget: 1 model load and at most
102 × 256 = 26,112 new tokens. The maximum Gemma prompt width is 483 tokens (v1) and 954 (v2).

## 3. Pre-declared report (`calib-report`; failures stay in every denominator)

For each version separately:
- **Validity:** parser validity, length failures, missing records, invalid outputs by reason, and U counts by question.
- **Historical q1:** q1 = Y on all 41 historical pairs, with N, U and failures shown separately.
- **Accepted 36:** exact acceptance-vector (Y,Y,Y,C) agreement, plus per-question agreement.
- **Rejected 5:** each is classed as acceptance vector, valid rejection, uncertain or invalid.
- **Controls:** q1 correct by expected class, and an expected-by-observed confusion table.

**Paired:** changes by case, and by qid.

**Scoring rules:**
- U and invalid outputs never count as correct negatives.
- There are no p-values or population-accuracy claims. The cases are exposed, clustered and partly artificial, and
  overall agreement alone is not treated as sufficient evidence.

## 4. Controls (Urdu question; proposed parent; expected q1; rule)

| Case | Question (gloss is AI-assisted) | Parent | Expected | Justification | Debatable? |
|---|---|---|---|---|---|
| C01 | کیا البرٹ آئن سٹائن کو نوبل انعام ملا تھا؟ (Did Albert Einstein receive the Nobel Prize?) | Albert Einstein | Y | TRANSLITERATION of the name | no |
| C02 | same | Theory of relativity | N | Related topic, not expressed | no |
| C03 | کیا چاند پر کشش ثقل زمین کے مقابلے میں کم ہوتی ہے؟ (Is gravity on the Moon weaker than on Earth?) | Gravity | Y | DIRECT_SPECIFIC_CONCEPT: کشش ثقل = gravity | low |
| C04 | same | Physics | N | Hypernym / superordinate field | no |
| C05 | کیا عطارد نظام شمسی میں سورج کے سب سے قریب سیارہ ہے؟ (Is Utarid, Mercury, the planet closest to the Sun?) | Mercury (planet) | Y | عطارد is the Urdu planet name; سیارہ fixes the sense | moderate: needs lexical knowledge |
| C06 | same | Mercury (element) | N | The question names the planet; the element (پارہ) is absent | moderate |
| C07 | کیا مرکری کا نام ایک رومی دیوتا کے نام پر رکھا گیا تھا؟ (Was Mercury named after a Roman god?) | Mercury (planet) | U | Shared base token; the page sense is not identified | yes: N also defensible |
| C08 | same | Mercury (element) | U | as C07 | yes: N also defensible |
| C09 | کیا ناسا نے کبھی انسانوں کو چاند پر اتارا ہے؟ (Has NASA ever landed humans on the Moon?) | NASA | Y | Transliterated abbreviation | no |
| C10 | same | Neil Armstrong | N | An entity inferred from another fact | no |

## 5. Proposed later execution (not submitted; needs approval)

**Submit** (from the repository root):

```
sbatch --time=01:30:00 --export=ALL,QEA_EXPECT_RUNNER_SHA256=<runner>,QEA_EXPECT_TEST_SHA256=<tests>,QEA_EXPECT_WRAPPER_SHA256=<gpu wrapper> eval/error_analysis_tests/efbpt/efbpt_qea_triage_gpu_v1.sbatch calib-gemma
```

- The job writes `outputs/efbpt/qea_review_calibration_v1/gemma_run/`.
- `calib-report` (offline) then writes `.../report/`.

**Time limit:** 01:30:00 is proposed instead of the one-hour default.
- Load took about 8–9 minutes in jobs 96317 and 96350.
- The expected run is about 25 minutes.
- The worst case, all calls at 256 tokens at the observed ~7.9 tokens/s, is about 64 minutes.

## 6. Provenance clarification (job 96350; its seal is not edited)

**What the job-96350 field was.** Its seal field `model.generation_config_file` actually holds
`model.generation_config`, read when the seal was built, after both generate calls.
- Transformers 4.57.6 `generate()` unsets a default `cache_implementation: "hybrid"` on that same object in place
  (`generation/utils.py:1749–1750`). That explains the recorded `null` against the file's `"hybrid"`.
- The cache class actually used at runtime was not recorded, and no claim is made about it.

**Going forward.** The Gemma backend now records:
- `generation_config_at_load`: a snapshot taken before any generation;
- `generation_config_live_at_seal`;
- a note saying which is which.

The generation behaviour itself is unchanged.

## 7. Record: calibration run, job 96416 (2026-09-27; the single approved run)

**Wording decision:** "reasonably" stays inside the verbatim freeze quotation. Both prompt versions are unchanged
(v1 `5c656e40…`, v2 `59e53e83…`).

**Correction to §2 and §4:** within each matched control group, two payload fields differ:
- `parent_title`;
- `stated_intermediate_information`, which follows the historical template and so differs only by the parent name.

The v1/v2 comparison of a case is unaffected, because both versions receive the identical payload.

**Preflight:**
- Seals, the approved code hashes, all 51 case identities and all 102 calls verified.
- The historical payloads and labels, rebuilt from the unchanged sources, are identical to the sealed ones.
- Each call is one fresh single-turn conversation.
- The control questions are in logical order: they begin U+06A9 U+06CC U+0627 and end with U+061F, and contain no
  bidirectional control characters.

**Execution:**
- COMPLETED 0:0 on L20003 in 24 min 10 s (1 h 30 min limit).
- Load took 723 s. `peak_cuda_bytes_allocated` was 23,109,114,368, and the maximum sampled GPU use was 22,446 of
  46,068 MiB.
- 7,489 of at most 26,112 new tokens were generated.

**Seals:**
- run `21ea719fb52b011f44a6c64971e80fc492831092ea9750ce34f7fadf19ffc8fd` (the labels were never read);
- report `b988991cebf2f8644a84f4ed8cb9df948c243fb4a8f78bc56acde401643f5dbc`.

**Validity:** 51 of 51 valid for each version. Every call stopped on `<end_of_turn>`, with no length stops and no
non-finite values. U was never answered in any of the 102 calls, and q3 was Y in all 102.

| Measure | v1 | v2 |
|---|---:|---:|
| Historical q1 = Y (of 41) | 3 | 35 |
| Exact Y, Y, Y, C on the 36 accepted pairs | 3 | 23 |
| Historically rejected pairs given Y, Y, Y, C (of 5) | 0 | 2 (H03, H23) |
| Controls, expected Y: q1 = Y (of 4) | 1 | 4 |
| Controls, expected N: q1 = N (of 4) | 4 | 3 (C04 Physics answered Y) |
| Controls, expected U: q1 = U (of 2) | 0 (both N) | 0 (both N) |

**Diagnostic findings:**
- **v1:** q1 is N on 47 of 51 calls. Its "correct" negatives come from near-blanket rejection, not discrimination.
- **v2 q2:** v2 answered q2 = N on 11 calls. In five historical cases (H04, H05, H08, H28, H30), the model's own note
  says the child is not directly identifiable, which contradicts its answer.
- **v2 q1 scope:** it was not always respected. C04's note cites a decomposition-step term. The two Africanized bee
  pairs, which share a question and parent, received different q1 answers.
- **Neither version:** neither rejected any of the five human q3/q4 rejections on q3 or q4.

**Conclusion:** neither version is deployed to R1 or R2. Further diagnosis is needed before triage.

## 8. Correction (2026-09-27, from the read-only audit of job 96416; §7 is left as written)
- **v2's 13 non-accepted historical pairs:** these fall into disjoint failure groups of 8 q2-only, 4 q1-only and
  1 q1-and-q4 (H22). Counted per condition, and overlapping, that is q2 on 8, q1 on 5 and q4 on 1.
- **§7 was wrong about the rejected pairs.** It said neither version rejected any of the five historical negatives on
  q3 or q4. In fact:
  - q3 detected none of the five.
  - q4 returned O for H06 under both versions.
- **C07/C08:** answering N was already listed as defensible in the control specification (§4), so those two N
  answers are not evidence of reviewer error.
