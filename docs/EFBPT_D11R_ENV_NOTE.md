# EFBPT D11-R — environment note (recorded changes outside the repo)

Companion to docs/EFBPT_D11R_READER_FREEZE.md (MD5 ed260e0ea2373337e8a3fb850bebe963).

## 1. sacrebleu installed in env urbench_eval (freeze §3 M2)
- Date: 2026-11-07. Command: `/mnt/home/user41/miniconda3/envs/urbench_eval/bin/pip install sacrebleu`
- Installed: **sacrebleu 2.6.0** (pure Python wheel sacrebleu-2.6.0-py3-none-any.whl) into
  /mnt/home/user41/miniconda3/envs/urbench_eval/lib/python3.10/site-packages
- Declared requirements: colorama, lxml, numpy, portalocker, regex, tabulate (all already present or
  pulled as pure-Python wheels; torch 2.9.0, transformers 4.57.6, vllm 0.13.0, scipy 1.15.3 unchanged).
- chrF is used with default settings (char order 6, word order 0, beta 2); the signature string is
  written into every scores JSON by d11r_score.py.

## 2. NLLB-200-3.3B copied into downloaded_models (freeze §2 L4)
- Source: /mnt/home/user49/junhaoliu/models/nllb-200-3.3B (another user's directory, read access only;
  files dated 2026-06-16). Destination: /mnt/home/user41/downloaded_models/nllb-200-3.3B (cp -p, 2026-11-07).
- The 12 repo files; MD5 of the copy == MD5 of the source for every file (md5sum both sides, diff empty).
  d11r_local_readers.py pins these and refuses to load on any mismatch.

| file | bytes | MD5 |
|---|---:|---|
| .gitattributes | 1,224 | 5a36cfd0d6a096c9479e1a0d2b19860c |
| README.md | 7,640 | 67d5eef3cbbf558a813c994713dc5d76 |
| config.json | 808 | 799b8dc12c1c0e3945393f51a3a15b9a |
| generation_config.json | 189 | d02f9e936c18ec442032be6958a1f887 |
| pytorch_model-00001-of-00003.bin | 6,933,375,959 | 00a270667cd56865fa2e513edc8f82ff |
| pytorch_model-00002-of-00003.bin | 8,545,249,147 | bf769074dccfe9cff4f2e435aed70308 |
| pytorch_model-00003-of-00003.bin | 2,098,840,299 | e7d757298a1a909aab58ed521b48a049 |
| pytorch_model.bin.index.json | 90,035 | d1d61694c01b5fc1481eb2fa9c6dea42 |
| sentencepiece.bpe.model | 4,852,054 | 05c551ae7955b3980d5a9d044eb09d70 |
| special_tokens_map.json | 3,548 | 9f9c7f6c1ad369053fdecd51145baafd |
| tokenizer_config.json | 564 | aab01dbc3bc6cdba3d83169124d77923 |
| tokenizer.json | 17,331,176 | 3f7d9492d20a2a391cc3e24851b2a2f3 |

The HF repo facebook/nllb-200-3.3B (commit 1a07f7d195896b2114afcb79b7b57ab512e7b43e) lists exactly these
12 file names; the repo files themselves were not downloaded (HEAD request only, D11-O readout §9b).

## 3. RDEV300 (freeze §1)
- Built 2026-11-07 by eval/error_analysis_tests/efbpt/d11r_build_rdev300.py: pool 1632 = mapped 2290 −
  DEV200 200 − eval458 458; random.Random(2026).sample(pool, 300); overlap DEV200 0, eval458 0;
  gold yes 143 / no 157; every row has evidence ids.
- data/strategyqa_official/efbpt/rdev300_seed2026.jsonl: MD5 **e307ad857b2f646f423bcd09fd3060d7**,
  522,211 bytes, 300 lines; seal data/strategyqa_official/efbpt/rdev300_seed2026.SEAL.json.

## 4. API credentials
- ~/.urbench_api_env (mode 600, outside the repo) exports URB_API_BASE and URB_API_KEY. The scripts read
  them from the environment only; nothing under the repo holds key material; R_START records the host
  name of the base URL only; every stored string passes through a scrub that would redact the key.
