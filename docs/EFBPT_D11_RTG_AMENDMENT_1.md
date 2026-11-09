# EFBPT D11 RTG — AMENDMENT 1 (2026-11-09, written with the scripts, before TEST-5)

Freeze: docs/EFBPT_D11_RTG_FREEZE.md (MD5 ce7d68801c302fa5a8d173bd3ee93eb7). Operational
interpretations needed to make §3, §5 and §8 executable; no arm, prompt, model, decoding or
decision rule changes.

A1.1 §3 leak guard and the reading. The teacher input is the sealed D11-R reading. A reading
     that is byte-identical to the gold question_en is a legitimate exact translation (D11-R M3:
     14.67% of RDEV300 rows for R1), so a literal "gold question_en substring is fatal" rule
     would abort the FULL run on correct rows. Rule applied: (a) the prompt with the reading
     slot blanked (system + user template) must contain no question_en of any row, no gold
     fact, no evidence id and no required title — FATAL; (b) the reading itself must contain
     no OTHER row's question_en, no gold fact and no evidence id ("<Title>-<n>") — FATAL;
     own-row equality (reading == gold question_en) is counted and reported; required titles
     inside a reading are reported (that is reading quality, D11-R M4), not fatal; (c) V4: the
     reading used for every qid equals the sealed file (MD5 pinned in §2 for DEV200; the
     reader's R_SEAL entry for eval458), one record per qid, every record ok and non-empty.
     Student prompts: question_en, evidence ids and every reading string FATAL in every arm;
     gold-fact substrings in teacher-fact arms reported as V6 (D10 A2.2 / D11-O A1.1).
A1.2 §5 V1 "across arms" = A_r2, TR, TR2, TR_L1, TR_wrong. TR_noq (layout G, no question)
     is excluded and its unparsed rate reported, as D10 §5 V1 and D1 arm G. In FULL a spread
     > 5.0 pp marks the readings VOID in the summary and the student exits 3 after sealing;
     in TEST-5 (one row = 20 pp) V1 and V5 are reported only, never gates.
A1.3 Jobs. One teacher job runs R1, R2, R3 in sequence (03:00:00), each sealed under
     outputs/efbpt/d11/teacher/<set>/<R1|R2|R3>/ (t_facts.jsonl, T_START.json, T_SEAL.json).
     The student seals S_SEAL.json after every arm. TEST-5 outputs live under
     outputs/efbpt/d11/_test/ and are never results. Chain: TEST teacher → TEST student →
     FULL teacher → FULL student, each --dependency=afterok (eval/error_analysis_tests/efbpt/
     d11_submit.sh).
A1.4 Derangement. FULL DEV200: the sealed D10 map (outputs/efbpt/d10/teacher/derangement.json,
     MD5 8f00e2f9e54f27138964a0577a8eff52) is loaded, verified, and must equal
     d10_teacher.derangement(qids, 1010) recomputed. TEST-5 uses d10_teacher.derangement over
     the 5 qids, seed 1010 (the D10 map covers 200 qids). eval458 uses the same function over
     the 458 qids, seed 1010, written to the student output directory and sealed.
A1.5 §8 mechanics. --set eval458 dies unless docs/D11_GO_EVAL458.md exists, and the student
     needs --eval-arm TR|TR2|TR_L1 equal to dev200_best in the sealed FULL DEV200
     d11_summary.json (tie → TR, §7). The eval458 reading file is verified against the
     reader's D11-R R_SEAL.json (the freeze pins DEV200 MD5s only). A_r2_eval is generated in
     the job (no sealed eval458 floor exists); V2' does not apply to eval458.
A1.6 API_direct_ur (§4) runs on the login node (d11_api_direct.py): system message
     "صرف ہاں یا نہیں لکھیں", user message = question_ur, qwen3.8-max, enable_thinking=false,
     temperature 0, max_tokens 256, 2 concurrent, D11-R retry policy, resumable, sealed under
     outputs/efbpt/d11/api_direct/dev200/. Scored with the frozen extractor (Rule 2) and the
     secondary scorer. Descriptive only, as frozen.
