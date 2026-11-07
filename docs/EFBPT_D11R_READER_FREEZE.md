# EFBPT D11-R — URDU→ENGLISH READER BENCHMARK (FREEZE)
Status: FROZEN. Purpose: choose the reading stage of D11 from measured reading
quality, on questions never used by any experiment, before any D11 answer run.
D11-O established (readout 9e06403): with the gold English question the v1
teacher gives +14.0 pp (p = 0.0006) vs +5.0 from Urdu; TO1 vs gold facts
−2.0 (p = 0.65). The reading step is the wall.

## 1. Item sets
- RDEV300: 300 rows sampled with random.Random(2026) from the training pool =
  rows of data/strategyqa_official/strategyqa_official_mapped_urbench_qid.jsonl
  in neither DEV200 nor eval458 (1632 rows), with non-empty question_ur and
  question_en. Written to data/strategyqa_official/efbpt/rdev300_seed2026.jsonl
  and sealed. Reader selection uses RDEV300 ONLY.
- DEV200 (200) and eval458 (458, the is_eval rows of the mapped file): read by
  every reader in the same job; readings sealed. DEV200 metrics may be
  reported after selection; eval458 readings are NOT scored by this experiment.
## 2. Readers (same instruction for every LLM reader; one-line output)
SYSTEM: "You are an expert Urdu-to-English translator. Translate the question
exactly, preserving its meaning. Named entities in Urdu questions are usually
transliterations of English names (people, places, brands, titles, works);
write each in its standard English spelling. Do not answer the question and
do not explain. Output only the English question, on one line."
USER: "{question_ur}"
API (OpenAI-compatible, URB_API_BASE/chat/completions, temperature 0,
max_tokens 256, 4 concurrent requests, retry on 429/5xx with 2/4/8 s backoff,
3 attempts): qwen3.8-max, deepseek-v4-pro, deepseek-v4-flash, qwen3.7-flash.
Thinking: the PROBE (§4) tries, per model, the documented switches
(enable_thinking=false; thinking={"type":"disabled"}; reasoning_effort="none")
and records reasoning_tokens; the FULL run uses, per model, the first variant
that yields reasoning_tokens == 0, else the default; the choice is recorded in
R_START. Every response is stored raw (minus nothing — the key is never in a
response) with usage, latency and finish_reason.
Local: (L1) Llama-3.1-70B-AWQ via vLLM, D10 §2 args, greedy, max_tokens 128;
(L2) Gemma-2-27B-it NF4 (QEA HfBnbBackend load), greedy, system text
prepended to the user turn, max_new 128; (L3) Qwen3-14B nf4 (the student,
D10 load), greedy, enable_thinking=False, max_new 128; (L4) NLLB-200-3.3B from
/mnt/home/user49/junhaoliu/models/nllb-200-3.3B (copy or MD5-pin the 12 files
first), urd_Arab→eng_Latn, beam 4, max_length 128.
Post-processing for all: first non-empty line, strip quotes/markdown.
## 3. Metrics vs gold question_en (RDEV300; DEV200 descriptive)
PRIMARY M1 proper-noun recall: gold proper nouns = capitalised tokens of the
gold question not in sentence-initial position (punctuation stripped; tokens
of ≥2 chars; all-caps kept; stoplist: I), plus digit tokens; recall = share
found in the reading, case-insensitive, token or substring. Macro over rows.
M2 chrF (sacrebleu, default settings; pip install sacrebleu recorded as an
environment change). M3 exact match after norm(). M4 title presence: for rows
with evidence ids, required titles present in the reading / present in the
gold question (ratio to the gold ceiling). Also: latency, output tokens,
reasoning tokens, cost at list price, empty/garbled readings count (reading
with <3 Latin letters), Urdu-script leakage count.
## 4. Protocol
PROBE (today): 3 RDEV300 rows × 4 API models × thinking variants; records
only; no metric. FULL: API readers on the login node (nohup, log under logs/);
local readers as one sbatch job sequence (L1 vLLM job; L2+L3+L4 HF job).
Selection rule (pre-declared): best M1 on RDEV300; if two readers are within
1.0 pp of M1, the higher M2 wins; ties → local over API, cheaper over dearer.
The selected reader and the runner-up are both carried into the D11 freeze
(runner-up = agreement partner). Outputs under outputs/efbpt/d11r/<reader>/,
R_START/R_SEAL per reader with the freeze MD5; readings for the three item
sets in separate files.
## 5. Leak and scope rules
Reader prompts contain only question_ur. No reader output is used to answer
anything in this experiment. eval458 readings are sealed unread. Gold
question_en is used only to score RDEV300/DEV200 readings.
