# D11 prep: resources, prior numbers, student generation, network, fact-check dump, gold-facts layout

Read-only, 2026-11-04. No code change, no job. Files written: this one and `docs/handoff/FACT_CHECK_DUMP.md` (§5). Sources: `docs/handoff/TEACHER_INVENTORY.md` (2026-10-30), `experiments.md`, `eval/error_analysis_tests/efbpt/{d1_eval_arms,d4_extract_facts,d10v2_student}.py`, `outputs/efbpt/d1/arms/d1_summary.json`, `outputs/efbpt/d8/d8_summary.json`, the two `tf_facts_sample30.jsonl` files, two `curl -I` calls.

## 1. Teacher inventory (from `docs/handoff/TEACHER_INVENTORY.md`)

### 1a. API access (§4 of the inventory)

**No API model or key is usable from the cluster as things stand.**

- No `.env`, secrets or credentials files in the repo (apart from `.git`).
- `ANTHROPIC_API_KEY`, `DASHSCOPE_API_KEY`: no references anywhere (main tree, code under `rag/` and `outputs/`, `urbench_repo.zip`, full git history).
- `OPENAI_API_KEY`: one reference, a comment in `scripts/translation.py:47` (`# Requires OPENAI_API_KEY to be set in env`, then `client = OpenAI()`), from the 2026-02-28 initial commit. Seven `scripts/translate_*.py` / `repair_*.py` files import `openai.OpenAI`; they were the dataset-translation tools and have no key configured.
- No hard-coded keys, no `api_key=` / `base_url=` anywhere in `scripts/` or `eval/`.
- Installed in `urbench_eval`: `anthropic 0.71.0` (never imported), `openai 2.15.0`; `dashscope` not installed.
- None of the three variable names is set in the shell (names checked, never values).
- Status: an API teacher would need a key provisioned and an outbound route; §4 below shows HTTPS egress works through the local proxy, so the network side is not the blocker.

### 1b. Local models ≥ 27B (§1 of the inventory; paths under `/mnt/home/user41/downloaded_models/`)

| model | path | du | quantisation on disk | how it has been loaded |
|---|---|---|---|---|
| Llama-3.1-70B-Instruct | `LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4` (symlink `LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`) | 38G | **AWQ 4-bit**, group 128, GEMM, zero_point, fp16; 9/9 shards, 39.77 GB | vLLM 0.13.0, TP 1, `quantization="awq"`, `max_model_len 4096`, `enforce_eager`; fits one 46 GB L20 with ≈3 GiB KV cache (D10 teacher, QEA T1/T2) |
| Gemma-2-27B-it | `AI-ModelScope/gemma-2-27b-it` | 51G | **none** (bf16; 12/12 shards, 54.45 GB) | HF transformers + bitsandbytes **NF4** (double-quant, bf16 compute), `HfBnbBackend` in `efbpt_qea_triage_v1.py:995-1036`; peak 22,446 MiB in job 96350. Never served with vLLM; bf16 does not fit one L20 |

Only these two models exceed 14B. Next largest: `Qwen/Qwen3-14B` (bf16, 28G; the student), `Qwen/Qwen2.5-14B-Instruct-AWQ` (9.4G).

### 1c. Gemma-2-27B and Qwen2.5-72B on disk?

- **Gemma-2-27B: yes**, `AI-ModelScope/gemma-2-27b-it`, bf16, complete (12/12 shards).
- **Qwen2.5-72B: no.** The inventory lists every weights directory; the only Qwen2.5 models are 14B-AWQ, 7B and 1.5B. The inventory also notes **no Llama-3.3-70B weights**: the five `*Llama-3___3-70B*` / `*Llama-3___1-70B-Instruct-AWQ` directories are 8.0K `.mdl` stubs.

## 2. Prior numbers from `experiments.md`

### 2a. EFBPT L0 constrained title-linking pre-test (2026-08-18; `experiments.md` §"EFBPT L0", lines 4748–4836)

**Setup.** Pre-test before building L1: can constrained title selection with `paraphrase-multilingual-MiniLM-L12-v2` beat free canonical-title generation? Frozen pair set: 289 pair instances (126 TRANSLIT, 161 SEMANTIC, 2 COREF hand-confirmed under Amendment 1); the LINKABLE gate set = 287 (TRANSLIT + SEMANTIC). Part B = free-generation baseline with the frozen D6 stage-0 prompt (289/289 measurable). Part C = encode Urdu spans and the deduplicated English-title universe (6,402,346 unique titles) with MiniLM; gold title absent from the corpus for 86/289 pair instances. Frozen gate: delivered = R@10 × PICK_ACC (0.85) must reach B + 6 pp.

**Part B, free generation:**

| set | correct / n | baseline B |
|---|---|---|
| ALL | 131/289 | 45.33% |
| LINKABLE | 131/287 | 45.64% |
| SEMANTIC | 58/161 | 36.02% |
| TRANSLIT | 73/126 | 57.94% |

**Part C, constrained retrieval over 6.4M titles:**

| set | n | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---|---|---|---|---|---|---|
| ALL | 289 | 5.88% | 9.34% | 11.76% | 13.49% | 15.57% | 17.99% |
| LINKABLE | 287 | 5.92% | 9.41% | 11.85% | 13.59% | 15.68% | 18.12% |
| SEMANTIC | 161 | 8.07% | 11.80% | 15.53% | 18.01% | 21.12% | 25.47% |
| TRANSLIT | 126 | 3.17% | 6.35% | 7.14% | 7.94% | 8.73% | 8.73% |

**Verdict.** LINKABLE B = 45.64%, R@10 = 11.85%, delivered = 10.07% vs pass threshold 51.64% → **GATE FAIL, DO NOT BUILD L1**. Corrected diagnostic (job 65737, 2026-08-19, 5,194-title universe, 203 usable pairs): English gold-title self-retrieval 99.5% @1; Urdu spans ALL R@1/5/10 = 28.1 / 36.0 / 36.5%, TRANSLIT 15.7 / 21.3 / 22.5%, SEMANTIC 38.4 / 48.2 / 48.2%. Diagnostic only, no gate change. Narrow conclusion recorded: this MiniLM encoder aligns raw Urdu entity spans poorly with canonical English titles.

### 2b. D1 arms B and E (job 58510; `experiments.md` §"D1 — knowledge vs language", lines 4043–4071; cross-checked against `outputs/efbpt/d1/arms/d1_summary.json`)

DEV200, base Qwen3-14B nf4, no adapters, floor 56.50%. Script `d1_eval_arms.py`, scorer `d1_score_dual.py`.

| arm | question | facts | acc % | correct | unparsed % | truncated % | pred-yes % | SEC acc % |
|---|---|---|---|---|---|---|---|---|
| A | Urdu | none | 57.50 | 115 | 4.00 | 4.5 | 38.5 | 57.50 |
| **B** | **Urdu** | **gold English (`urbench_facts`)** | **78.00** | **156** | **0.50** | 2.0 | 37.5 | **78.00** |
| C | Urdu | English, wrong row (shift 97) | 57.00 | 114 | 6.00 | 3.5 | 25.5 | 57.00 |
| **E** | **English** | **gold English** | **76.50** | **153** | **10.00** | 1.5 | 31.5 | **82.50** |
| F | English | none | 55.50 | 111 | 16.50 | 4.0 | 22.5 | 59.00 |
| G | none | gold English | 42.00 | 84 | 33.50 | 0.5 | 37.0 | 42.00 |

Accuracy, unparsed and SEC columns are as printed in `experiments.md`; correct, truncated and pred-yes come from `d1_summary.json` (the SEC column is from the Devanagari-secondary scorer and is not in the summary file). Recorded readings: B − A = +20.50 pp (the knowledge ceiling with gold facts); E − B = −1.5 pp primary / +4.5 pp secondary (English vs Urdu question with the same facts); C at the floor; G below the floor (no answer leakage); the D1 comparison is formally VOID on the unparsed spread (33 pp) but B − A survives any bounding (B leads by ≥ 16.5 pp even if every unparsed A row were correct). The D10 A_r2 re-run of arm A later gave 60.00% (v1 readout §1), with only 27/200 generations identical to D1 arm A at batch size 8 vs 1.

## 3. Student generation: the HF generate call and forced-answer logits

`generate_all` in `eval/error_analysis_tests/efbpt/d4_extract_facts.py` lines 122–148 (imported unchanged by `d10_student.py` and `d10v2_student.py`; `d1_eval_arms.py` lines 144–170 carries an identical local copy with `MAX_NEW_TOKENS` hard-wired):

```python
def generate_all(model, tok, prompts, batch_size, eos_ids, max_new):
    n = len(prompts)
    order = sorted(range(n), key=lambda i: len(prompts[i]))
    out = [None] * n
    t0 = time.time()
    done = 0
    for s in range(0, n, batch_size):
        idxs = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idxs], return_tensors="pt",
                  padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new,
                               do_sample=False, pad_token_id=tok.pad_token_id)
        for j, i in enumerate(idxs):
            new_ids = g[j][in_len:]
            keep = [t for t in new_ids.tolist() if t != tok.pad_token_id]
            trunc = (len(keep) >= max_new) and (
                len(new_ids) > 0 and new_ids[-1].item() not in eos_ids)
            out[i] = (tok.decode(new_ids, skip_special_tokens=True),
                      len(keep), trunc)
        done += len(idxs)
        el = time.time() - t0
        print("    %d/%d  (%.1fs, %.1fs/row)" % (done, n, el, el / done),
              flush=True)
    return out
```

The model it receives is loaded in every D-script the same way (`d1_eval_arms.py:215-221`, `d4_extract_facts.py:229-237`, `d10v2_student.py:673-679`):

```python
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb, dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map={"": 0}, trust_remote_code=True)
model.config.use_cache = True
model.eval()
```

Tokenizer: left padding, `pad_token = eos_token` if unset; `eos_ids` = `{eos_token_id, <|im_end|>}`; the prompt is `tok.apply_chat_template([system, user], add_generation_prompt=True, enable_thinking=False)`, so every generation starts after an empty `<think>\n\n</think>\n\n` block. Decoding: greedy (`do_sample=False`), `max_new_tokens` = `MAX_NEW_TOKENS` = 1024, batch size 1 in D10/D10v2 (8 in D1). The call returns only token ids: no `output_scores`, no `return_dict_in_generate`, so **the sealed runs contain no logits**.

**Are logits at a forced-answer position obtainable with the loaded nf4 model? Yes, with a new script; nothing in the quantisation prevents it.**

- A bitsandbytes 4-bit `AutoModelForCausalLM` is an ordinary `PreTrainedModel`: `model(**enc).logits` (one forward pass, no generation) returns `[batch, seq, vocab]` bf16 logits (the `lm_head` is not quantised by bitsandbytes, and the compute dtype is bf16). `model.generate(..., output_scores=True, return_dict_in_generate=True)` likewise returns per-step scores; with `do_sample=False` and no logits processors, `scores[t]` equals the raw next-token logits at step t.
- "Forced-answer position" in this pipeline means: append the answer prefix the extractor looks for (`extract_answer` in `efbpt_eval_dev200.py` reads the marker `حتمی جواب:` followed by `ہاں` / `نہیں`, or a bare leading yes/no) to the prompt and read the logits of the next token. The two Urdu answer words are multi-token in the Qwen3 tokenizer, so a clean two-way score needs either (i) the log-probability of each full answer string by teacher forcing (sum of token log-probs under `model(...)`), or (ii) a comparison of the first-token logits only if the two words' first tokens differ. Which of the two holds must be checked once with `tok.encode("ہاں")` and `tok.encode("نہیں")` on the actual tokenizer before any freeze; this file does not run the tokenizer.
- Cost: one forward pass per prompt (no 1024-token generation), so roughly 50–100× cheaper per row than the current arms; memory is the same model plus one prompt.
- Caveats to write into any D11 freeze: a forced-answer score is a different measurement from the free-generation accuracy used in D1–D10 (no reasoning tokens before the answer, so it measures the prior at the answer position, not the answer after Urdu chain-of-thought); greedy generation with thinking off is what every prior arm scored, so forced-answer numbers must not be compared to the 60.00% / 65.00% / 78.00% figures without a bridging arm; bf16 logits on an nf4 model are deterministic at batch size 1 only (the D10 A_r2 file reproduced byte for byte across nodes at batch size 1, D10V2 pack §5a).

## 4. Network (HEAD only, `curl -I -s -S -m 10`, no download)

| target | reachable | HTTP | time |
|---|---|---|---|
| `https://dumps.wikimedia.org/enwiki/latest/` | **yes** | 200 | 0.99 s |
| `https://huggingface.co` | **yes** | 200 | 0.85 s |

Both go through the login node's configured proxy (`HTTPS_PROXY=http://localhost:20000`, `HTTP_PROXY` same, `NO_PROXY=localhost,127.0.0.1,*.hitsz.edu.cn,*.edu.cn`). Reachability was tested from this interactive shell on the login node only; compute nodes (L20001–L20005) were not tested and may not share the proxy. Nothing was downloaded.

## 5. Fact-check dump

Written: **`docs/handoff/FACT_CHECK_DUMP.md`** (1083 lines, 90 facts). AUDIT FILE: nothing in it enters any prompt.

The three samples (each `random.Random(8).sample(…, 30)`):

| sample | file | drawn from | n | distinct qids | hand-check fields |
|---|---|---|---|---|---|
| D8-Xp | `outputs/efbpt/d8/d8_summary.json` → `V5_Xp_hand_check_sample.facts` (job 108196). **There is no `d8_xp_sample30.jsonl`**: the D8 script (`d8_reachable_ceiling.py:963-971`) lists the sample inside the summary only; each entry has `qid`, `titles_corpus`, `fact` | Xp arm injected facts | 30 | 28 | `status: PENDING hand check` on the summary block |
| D10-TF | `outputs/efbpt/d10/student/tf_facts_sample30.jsonl` (job 109582) | TF arm, v1 teacher facts | 30 | 29 | `truth: null`, `relevance: null` per row |
| D10v2-TF | `outputs/efbpt/d10v2/student/tf_facts_sample30.jsonl` (job 110655) | TF arm, v2 teacher facts | 30 | 29 | `truth: null`, `relevance: null` per row |

Per row the dump gives: sample name, index, qid, the fact to check, the sample row's context (`titles_corpus` or teacher `entities`), gold `question_en` (audit), gold answer, the gold `urbench_facts` (audit), and blank `truth` / `relevance` slots. All 90 qids are DEV200 qids. Verdicts should be recorded in a separate file; the dump is generated.

## 6. The gold-facts layout: is `build_user(…, "B")` with `row["facts"] = urbench_facts` exactly D1 arm B?

**Yes, the user message is identical; the chat wrapping and decoding are identical; the only differences are batch size and the resulting generations.**

- `ARM_SPEC["B"] = ("question_ur", "gold")` in `d1_eval_arms.py:74`. `build_user(instruction, row, donor_facts, "B")` (lines 114–123) returns `instruction + "\n\n" + facts_block(row["facts"]) + "\n\n" + row["question_ur"]`, where `facts_block` (line 110) is `FACTS_HEADER + ":\n" + "\n".join("- " + f.strip() for f in facts)` and `FACTS_HEADER` is `حقائق` built from codepoints. D1 fills `row["facts"]` from the loader (`"facts": list(r["urbench_facts"])`, line 102). So D1 arm B = `instruction`, blank line, `حقائق:` + one `- ` line per gold `urbench_facts` string (stripped), blank line, `question_ur`.
- The D10/D10v2 student calls the same function for its layout B: `build_user(instruction, dict(row, facts=list(facts)), None, "B")` (`d10v2_student.py:425`), with `facts` = the teacher's post-T2 facts. Passing `facts = DEV200 urbench_facts` instead reproduces the D1 arm B user message byte for byte, because D1 also used the stripped list and the same header.
- Wrapping: D1 (`d1_eval_arms.py:237-241`) and D10v2 (`chat()`, lines 408–413) both call `tok.apply_chat_template([{"role":"system","content":SYSTEM_MESSAGE},{"role":"user","content":user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)`; same `SYSTEM_MESSAGE`, same instruction file (`plan_a_instruction_ur.txt`, MD5 `f3b58d76…`), same model load, same `MAX_NEW_TOKENS` 1024, greedy. Difference: D1 ran at `--batch-size 8` (default); D10/D10v2 at batch size 1 (freeze §2). The A_r2 re-run showed that this alone changes 173/200 generations and +2.5 pp (57.50% → 60.00%), so a batch-size-1 gold-facts arm should be expected near, not at, 78.00%.
- The leak guard is the reason this layout was never built into D10: `d8_reachable_ceiling.load_leak_needles` makes every `urbench_facts` string a prompt needle, and `leak_guard(..., fatal_gold_facts=True)` dies on any match (D10 relaxed it to "reported as V6" only for teacher facts under A2.2). A gold-facts arm must bypass or re-scope that guard explicitly in its freeze, and must be labelled as the ceiling, never as a method arm.

**Which loader gives rows WITH facts.**

| loader | file | row keys | facts |
|---|---|---|---|
| `d1_eval_arms.load_rows(limit=None)` | `d1_eval_arms.py:82-107` | `qid`, `question_ur`, `question_en`, **`facts`** (= `list(r["urbench_facts"])`), `gold` | **yes**; dies if `urbench_facts` is empty or `answer` is not a bool |
| `d4_extract_facts.load_rows()` | `d4_extract_facts.py:76-93` | `qid`, `question_ur`, `question_en`, `gold`, `required` (titles from `evidence_paragraph_ids`) | **no**; the key is never copied, and the D10/D10v2 teacher and student assert `"facts" not in row` on every loaded row (`d10v2_teacher.py:420-423`, `d10v2_student.py:323-326`) |

Both read the same `DEV_PATH` (`data/strategyqa_official/dev200_seed4242.jsonl`) and both hard-check 200 rows. A D11 gold-facts ceiling arm therefore imports `load_rows` from `d1_eval_arms` (or reads `urbench_facts` itself), not from `d4_extract_facts`, and cannot reuse the D10 student's preflight unchanged because that preflight dies on a `facts` key.
