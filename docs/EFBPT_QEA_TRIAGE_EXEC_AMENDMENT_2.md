# QEA triage: operational amendment 2 (Gemma reviewer backend: load dtype and stop tokens)

**Development note, 2026-09-27. Operational only.**

This note does not edit:
- the QEA protocol revision 0.2 (`776b26f0…`);
- amendment 1 (`docs/EFBPT_QEA_TRIAGE_EXEC_AMENDMENT_1.md`);
- any seal, manifest or prior output;
- the Llama execution;
- the review prompt, the inputs, or the acceptance and selection rules.

## 1. Evidence from the installed code and local files (read-only; transformers 4.57.6, bitsandbytes 0.49.1)

**Load dtype without a dtype argument**
- `from_pretrained` calls `get_hf_quantizer` (`modeling_utils.py:4881`) before `_get_dtype` (`:4962`).
- `get_hf_quantizer` runs `dtype = hf_quantizer.update_dtype(dtype)` (`quantizers/auto.py:326`).
- The 4-bit quantizer's `update_dtype` turns `None` into `torch.float16` (`quantizer_bnb_4bit.py:244–255`).
- `_get_dtype` then sets float16 as the construction dtype (`modeling_utils.py:1263–1287`).
- The previous `HfBnbBackend` passed no dtype, so its non-quantized parameters loaded in float16. This affects the
  tied embedding/`lm_head` (not converted, `integrations/bitsandbytes.py:291–330`) and the RMSNorm weights.

**Quantized layers**
- `Linear4bit.forward` casts its input to `compute_dtype` (bf16, from `bnb_4bit_compute_dtype`), and returns
  `.to(inp_dtype)` (`bitsandbytes/nn/modules.py:550–557`).
- With a float16 model, activations between layers were therefore float16, while the checkpoint is bfloat16
  (`config.json`).

**Tokens (local tokenizer)**
- `<pad>` 0, `<eos>` 1, `<bos>` 2, `<unk>` 3, `<start_of_turn>` 106, `<end_of_turn>` 107.

**Stopping**
- The local `generation_config.json` has `eos_token_id: 1`, so the stop set was {1 `<eos>`} only.
- Generate builds the stop set from `generation_config.eos_token_id` (`generation/utils.py:2080`, `:1337`), and
  `EosTokenCriteria` checks the last token only (`stopping_criteria.py:472`).
- `<end_of_turn>` (107) was not a stop token.
- The old finish classification treated {1, 107} as stops, which is a different set from the one generation used.

**Chat template (sha256 `ecd6ae51…`)**
- It renders one `<bos>` and a single user turn (Gemma has no system role; system text and user payload are
  joined by a blank line).
- It ends with the assistant prefix `<start_of_turn>model\n`.
- Re-tokenizing the rendered text reproduces the IDs.
- No defect.

**Slicing and cleanup**
- Decoder-only generate returns the prompt followed by the new tokens (`generation/utils.py:2840`).
- `out[0, n:]` is the continuation. Decoding uses `clean_up_tokenization_spaces=False`.
- The only later cleanup is the parser's `strip_one_fence`.
- No defect.

**Compilation:** the bitsandbytes quantizer does not override `is_compileable` (base returns False), so generate
does not auto-compile. Forward hooks run in eager mode.

## 2. Corrections (only these)

1. **Explicit bfloat16 load.** `from_pretrained(..., dtype=torch.bfloat16)`.
   - NF4, double quantization, bf16 compute, eager attention and `device_map={"": 0}` are unchanged.
   - After loading, the backend fails closed unless `embed_tokens`, `lm_head`, the final norm and the 4-bit compute
     dtype are all bfloat16.
2. **Stop set.** The generation config's existing EOS (1 `<eos>`) plus `<end_of_turn>` (107), both verified
   against the local tokenizer, are passed to `generate(eos_token_id=[1, 107])`.
3. **Finish classification.** Finish is classified against exactly that set:
   - `stop` when the last generated token is a stop token, including on the final allowed token;
   - `length` when the budget is spent without one;
   - anything else is `unexplained_stop`, which the parser rejects.

**Consequential changes**
- Inputs go to `self.model.device`, which is `cuda:0` under the unchanged device map. This allows CPU tests.
- A prefix check confirms the returned sequence begins with the exact prompt.

**Unchanged:** the two historical review inputs, the review prompt (`5c656e40…`), greedy decoding, eager
attention and the 256-token cap. There is no output repair, forced label, token suppression, numerical
sanitization, structured decoding or longer budget.

## 3. Evidence recorded per request (`diagnostics` in smoke and R1 records)

**Input and output**
- Input payload hash, input token hash and width.
- The complete continuation token IDs and their count.
- Decodes with special tokens kept and removed, and the text after the parser cleanup.

**Token and stop summary**
- Token frequency (top 10 with names, distinct count, special-token count).
- Last token, finish reason, stop set, and whether the cap was reached.

**Numerics.** Two passive observers keep only scalar counts:
- **Raw logits** (a forward hook on the causal-LM output, last position): any NaN or ±Inf is a numerical failure.
- **Processed scores** (a logits processor appended after the defaults; it returns the same tensor): NaN, +Inf or
  an all −Inf step is a failure, and other −Inf entries are counted as masking.
- For this greedy configuration no default processor is active.

**Seal identity** records the requested load dtype and a census of effective parameter dtypes. It also records:
- the quantization configuration;
- the effective attention implementation;
- the generation-config file values and the generate-call overrides;
- the tokenizer's special tokens and chat-template hash.

**Output location:** the smoke writes to `outputs/efbpt/qea_triage_v1_smoke_gemma_v2/gemma/`. The root must be
absent before the model loads and is created with an exclusive `mkdir` at write time. Job 96317's output stays in
`outputs/efbpt/qea_triage_v1_smoke/gemma/`, and its missing token IDs are not reconstructed.

## 4. Record: Gemma smoke, job 96350 (2026-09-27; the single authorized retry)

**Execution:**
- COMPLETED 0:0 on L20005 in 9 min 57 s.
- The guard passed (runner `8bf2227f…`, tests `89dc5cb5…`, executed wrapper `7f61db36…`), and all 80 tests
  passed on the node.
- Model load took 533 s. `peak_cuda_bytes_allocated` was 23,109,114,368, and the maximum sampled GPU use was
  22,446 of 46,068 MiB.

**Seal:** `3b8f61c5c940dfbb505238c0a1e06738d2d21f77f1e9b778bc70955163a40810`, records `3bbd829c…` (6,537 bytes)
match. The inputs are the two pinned review payloads.

**Effective settings:**
- All non-quantized parameters are bfloat16 (186 tensors), and there are 322 Linear4bit layers (uint8 storage,
  bf16 compute and quant state).
- The stop set is [1 `<eos>`, 107 `<end_of_turn>`], with eager attention.

**Results:**

| Request | Generated tokens | Finish | Last token | Non-finite in raw logits or processed scores | Parser | Decision (q1–q4) |
|---|---:|---|---|---|---|---|
| review[0] | 70 of 256 | stop | 107 `<end_of_turn>` | none | valid | N, Y, Y, C |
| review[1] | 70 of 256 | stop | 107 `<end_of_turn>` | none | valid | N, Y, Y, C |

**Verdict: PASS under the stated criteria.**
- Two non-empty, parser-valid reviews.
- No length exhaustion, numerical failure, OOM or integrity failure.

The corrected backend works on these two requests. Which correction produced the recovery is not identified.
Agreement with the historical labels (Y, Y, Y, C) is not a criterion.
