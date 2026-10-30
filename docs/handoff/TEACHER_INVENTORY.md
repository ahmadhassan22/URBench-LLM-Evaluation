# Teacher inventory: models, GPU, API access, vLLM, D1 arm G

Read-only inventory, 2026-10-30. Sources: the filesystem (`du`, `config.json`, safetensors index files), the repo, the logs, `sacct`, `scontrol`, `sinfo`, and `sacctmgr`. I wrote no file other than this one, made no edits or commits, and ran no sbatch or downloads. API-key searches printed variable names only, never values.

---

## 1. Models on disk

- **`~/.cache/modelscope/hub/models/`** does not exist.
- **`~/.cache/huggingface/hub/`** holds two directories, `models--enstazao--Qalb-1.0-8B-Instruct` (12K) and `models--large-traversaal--Alif-1.0-8B-Instruct` (12K). Each contains only `refs/main`, with no blobs or snapshots, so no weights.
- **`downloaded_models/`**: every directory with weights is listed below, from `config.json`. Every sharded model has all of its shards present, checked against `model.safetensors.index.json`.

| path (`downloaded_models/…`) | du -sh | architecture | hidden | layers | quantization | >14B |
|---|---|---|---|---|---|---|
| `AI-ModelScope/gemma-2-27b-it` | 51G | Gemma2ForCausalLM | 4608 | 46 | none (bf16; 12/12 shards, 54.45 GB) | **YES (27B)** |
| `LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4` | 38G | LlamaForCausalLM | 8192 | 80 | **AWQ 4-bit**, group 128, GEMM, zero_point (fp16; 9/9 shards, 39.77 GB) | **YES (70B)** |
| `Qwen/Qwen3-14B` | 28G | Qwen3ForCausalLM | 5120 | 40 | none (bf16; 8/8) | no (14B) |
| `Qwen/Qwen2.5-14B-Instruct-AWQ` | 9.4G | Qwen2ForCausalLM | 5120 | 48 | **AWQ 4-bit**, group 128, GEMM (fp16; 3/3) | no (14B) |
| `AI-ModelScope/gemma-2-9b-it` | 18G | Gemma2ForCausalLM | 3584 | 42 | none (bf16; 4/4) | no |
| `LLM-Research/Meta-Llama-3___1-8B-Instruct` | 30G | LlamaForCausalLM | 4096 | 32 | none (bf16; 4/4, 16.06 GB, plus 15G `original/` Meta checkpoint) | no |
| `Qwen/Qwen3-8B` | 16G | Qwen3ForCausalLM | 4096 | 36 | none (bf16; 5/5) | no |
| `Qalb/Qalb-1.0-8B-Instruct` | 15G | LlamaForCausalLM | 4096 | 32 | none (4/4) | no |
| `Alif/Alif-1.0-8B-Merged` | 15G | LlamaForCausalLM | 4096 | 32 | none (4/4) | no |
| `Alif/Alif-1.0-8B-Instruct` | 6.2G | LoRA adapter only. `config.json` = `{"model_type": "llama"}`; adapter r=128, alpha 32, base `unsloth/Meta-Llama-3.1-8B`, modules_to_save embed_tokens + lm_head | – | – | – | no |
| `Qwen/Qwen2___5-7B-Instruct` | 15G | Qwen2ForCausalLM | 3584 | 28 | none (bf16; 4/4) | no |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 15G | Qwen2ForCausalLM | 3584 | 28 | none (bf16; 2/2) | no |
| `Qwen/Qwen2___5-1___5B-Instruct` | 2.9G | Qwen2ForCausalLM | 1536 | 28 | none (bf16) | no |
| `BAAI/bge-reranker-v2-m3` | 2.2G | XLMRobertaForSequenceClassification | 1024 | 24 | none (fp32) | no (reranker) |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 4.4G | BertModel | 384 | 12 | none | no (embedder) |

**Flagged above 14B:**
- **gemma-2-27b-it**: the bf16 weights are 54.45 GB and do not fit on one 46 GB L20. The QEA jobs loaded it in 4-bit NF4 (§2).
- **Llama-3.1-70B-Instruct-AWQ-INT4**: the weights are 39.77 GB and fit on one L20, leaving about 3 GiB for KV cache (§2).

**Symlinks** (no extra weights):
- `LLM-Research/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` → `…/Meta-Llama-3___1-70B-Instruct-AWQ-INT4`
- `LLM-Research/Meta-Llama-3.1-8B-Instruct` → `…/Meta-Llama-3___1-8B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct` → `…/Qwen2___5-1___5B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct` → `…/Qwen2___5-7B-Instruct`

**Empty or placeholder directories (no weights):**
- These hold only a `.mdl` stub, 8.0K each:
  - `AI-ModelScope/Llama-3___1-70B-Instruct-AWQ`
  - `AI-ModelScope/Llama-3___3-70B-Instruct-AWQ`
  - `AI-ModelScope/Meta-Llama-3___1-70B-Instruct-AWQ`
  - `LLM-Research/Meta-Llama-3___3-70B-Instruct-AWQ`
  - `modelscope/Meta-Llama-3___3-70B-Instruct-AWQ`
  - `large-traversaal/Alif-1___0-8B-Instruct`
- `LLM-Research/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (8.0K) holds only an empty `._____temp/`.
- `models--large-traversaal--Alif-1.0-8B-Instruct` (4.0K) holds only `refs/`.
- `._____temp/` (124K) holds empty subdirectories and no files.
- `.lock/` holds empty lock files.

**No Llama-3.3-70B weights are on disk.** The only weights above 14B are the two flagged models.

---

## 2. Models loaded by the QEA Llama and Gemma smoke jobs

**Runner** `eval/error_analysis_tests/efbpt/efbpt_qea_triage_v1.py`:

```
78:MODELS = {
79:    "llama": {"path": "/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4",
80:              "family": "Llama-3.1-70B-Instruct AWQ-INT4", "backend": "vllm", "role": "T1/T2 annotator",
81:              "max_model_len": 4096},
82:    "gemma": {"path": "/mnt/home/user41/downloaded_models/AI-ModelScope/gemma-2-27b-it",
83:              "family": "Gemma-2-27B-it (4-bit NF4 load)", "backend": "hf_bnb4", "role": "T4 reviewer R1",
84:              "max_model_len": 8192},
448:    "model_path": MODELS["llama"]["path"], "quantization": "awq", "dtype": "float16", "enforce_eager": True,
834:        self.llm = LLM(model=spec["path"], tensor_parallel_size=1, max_model_len=spec["max_model_len"],
836:                       enforce_eager=True, disable_log_stats=True, quantization="awq", dtype="float16", seed=0, **extra)
1009:        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
1010:                                   bnb_4bit_compute_dtype=torch.bfloat16)
```

**sbatch** `eval/error_analysis_tests/efbpt/efbpt_qea_triage_gpu_v1.sbatch`. It contains no model path; it selects the backend, and the runner's `MODELS` table supplies the path:

```
3:#SBATCH --partition=q_intel_share_L20
6:#SBATCH --gres=gpu:1
80:  smoke-llama) exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" smoke --backend llama ;;
81:  smoke-llama-so) exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" smoke --backend llama --constrained ;;
82:  smoke-gemma) exec "${PY}" -B -s "${DIR}/efbpt_qea_triage_v1.py" smoke --backend gemma ;;
```

(`efbpt_qea_triage_cpu_v1.sbatch` loads no model.)

**Job 96151** (`stage=smoke-llama-so job=96151 host=L20002 utc=2026-09-26T15:23:26Z`). Model lines from `logs/efbpt_qea_triage_gpu_96151.log`:

```
INFO 09-26 23:24:06 [utils.py:253] non-default args: {'dtype': 'float16', 'max_model_len': 4096, 'gpu_memory_utilization': 0.95, 'max_num_seqs': 4, 'disable_log_stats': True, 'quantization': 'awq', 'enforce_eager': True, 'structured_outputs_config': StructuredOutputsConfig(backend='xgrammar', …), 'model': '/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4'}
INFO 09-26 23:24:06 [model.py:514] Resolved architecture: LlamaForCausalLM
(EngineCore_DP0 pid=3352548) INFO 09-26 23:24:09 [core.py:93] Initializing a V1 LLM engine (v0.13.0) with config: model='/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4', … tensor_parallel_size=1, … quantization=awq, enforce_eager=True, …
(EngineCore_DP0 pid=3352548) INFO 09-26 23:24:11 [gpu_model_runner.py:3562] Starting to load model /mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4...
(EngineCore_DP0 pid=3352548) INFO 09-26 23:28:10 [gpu_model_runner.py:3659] Model loading took 37.0789 GiB memory and 237.534893 seconds
(EngineCore_DP0 pid=3352548) INFO 09-26 23:28:23 [gpu_worker.py:375] Available KV cache memory: 3.05 GiB
(EngineCore_DP0 pid=3352548) INFO 09-26 23:28:23 [kv_cache_utils.py:1291] GPU KV cache size: 10,000 tokens
(EngineCore_DP0 pid=3352548) INFO 09-26 23:28:23 [kv_cache_utils.py:1296] Maximum concurrency for 4,096 tokens per request: 2.44x
```

(The `…` marks long config strings I shortened.) The SEAL `outputs/efbpt/qea_triage_v1_smoke_v2/llama/SEAL.json` records `model.path` = the same AWQ-INT4 directory, `backend vllm`, `quantization awq`, `vllm 0.13.0`, and `load_seconds 257.594`.

**Job 96350** (`stage=smoke-gemma job=96350 host=L20005 utc=2026-09-27T08:45:58Z`). Neither the `.log` nor the `.err` has a `[load]` line or a model-id line. The `.log` has 9 lines: identity, nvidia-smi, sampler, seal. The only load trace in `.err` is the shard progress bar:

```
Loading checkpoint shards: 100%|██████████| 12/12 [08:48<00:00, 44.06s/it]
```

The model identity is in the SEAL `outputs/efbpt/qea_triage_v1_smoke_gemma_v2/gemma/SEAL.json`:
- `model.path` = `/mnt/home/user41/downloaded_models/AI-ModelScope/gemma-2-27b-it`
- `model.family` = `Gemma-2-27B-it (4-bit NF4 load)`
- `model.backend` = `transformers+bitsandbytes`
- `model.quantization` = `nf4 double-quant, bf16 compute`
- `model.load_dtype` = `torch.bfloat16`
- `model.load_seconds` = 533.053
- `model.transformers` = 4.57.6, `model.bitsandbytes` = 0.49.1

**Peak GPU memory.** The nvidia-smi output in both logs is the pre-load snapshot only: `NVIDIA L20, 46068 MiB total, 0 MiB used, 45595 MiB free, driver 535.183.01`. The peak comes from each job's 2-second sampler CSV:

| job | stage | node | peak memory.used | at | samples | note |
|---|---|---|---|---|---|---|
| 96151 | smoke-llama-so | L20002 | **42,794 MiB** / 46,068 | 2026/09/26 23:28:45.667 | 150 | last sample 0 MiB (clean exit) |
| 96350 | smoke-gemma | L20005 | **22,446 MiB** / 46,068 | 2026/09/27 16:55:50.557 | 288 | the peak is the final sample; the sampler stopped with the job, so a higher value just before exit was not captured |

For reference, the other QEA GPU jobs were: 96120 smoke-llama (L20001), 96136 smoke-llama (L20005), 96317 smoke-gemma (L20002), 96416 calib-gemma (L20003), and 96485 ident-gemma (L20001).

---

## 3. GPU and partition

**GPU type and memory.** The D8 log (`logs/d8_reachable_FULL_108196.log`, nvidia-smi) shows `NVIDIA L20 … 0MiB / 46068MiB`, driver 535.183.01, CUDA 12.2, on node L20001. `sacct -j 108196 -X` records Partition `q_intel_share_L20` and AllocTRES `billing=8,cpu=8,gres/gpu:tesla=1,mem=32G,node=1`. SLURM labels the GRES type `tesla`; the actual model (L20) comes from nvidia-smi. 46,068 MiB = 45.0 GiB (about 48.3 GB decimal). This matches "L20, 46 GB".

**Partition** (`scontrol show partition q_intel_share_L20`). The output has no separate `Gres=` line; the GPU count appears in the TRES line:

```
   AllowGroups=L20 AllowAccounts=ALL AllowQos=ALL
   Nodes=L[20001-20007]
   MaxNodes=UNLIMITED MaxTime=UNLIMITED
   State=UP TotalCPUs=336 TotalNodes=7
   TRES=cpu=336,mem=6479200M,node=7,billing=336,gres/gpu:tesla=56
```

`sinfo -p q_intel_share_L20 -o "%N %G"`:

```
NODELIST GRES
L[20001-20007] gpu:tesla:8
```

Each node has 8 GPUs, 48 CPUs, and 925,600 MB RAM.

**Can two GPUs be requested? Not under your current limits.**
- **Hardware:** a node has 8 GPUs and the partition sets no per-job GPU cap, so the partition itself would allow it.
- **Your association** (`sacctmgr show assoc user=user41`): `hpc01|hpc01|user41||normal|gres/gpu:tesla=1,gres/gpu=1|…`. The QOS is `normal` and **GrpTRES caps the association at 1 GPU in total**. QOS `normal` has no limits of its own. Separate QOS `2gpu` (GrpTRES gres/gpu=2) and `4gpu` exist, but your association is not granted them.
- **Enforcement:** `AccountingStorageEnforce = associations,limits,qos`, so the cap is enforced. A `--gres=gpu:2` job could not run: it would be rejected at submit or left pending on the association GRES limit (I didn't test which, since sbatch was not allowed).
- **History:** 249 of your jobs since 2026-01-01 used `gres/gpu:tesla=1`, and none used 2 or more. No sbatch file in the repo requests more than 1 GPU.
- Getting 2 GPUs needs an admin to raise GrpTRES or attach QOS `2gpu`.

---

## 4. API access

- **No `.env`, secrets, or credentials files** exist in the repo, apart from `.git`.
- **`ANTHROPIC_API_KEY` and `DASHSCOPE_API_KEY`: no references anywhere.** I searched:
  - the main tree, excluding `rag/` (121G corpus) and `outputs/` (3.2G)
  - code and config files under `rag/` and `outputs/`
  - the `.py` members of `urbench_repo.zip`
  - git history across all branches (`git log --all -S`)
- **`OPENAI_API_KEY`: one reference**, `scripts/translation.py:47`, in the comment `# Requires OPENAI_API_KEY to be set in env`, followed by `client = OpenAI()`. The same file is inside `urbench_repo.zip`. It dates from commit `424cd82` (2026-02-28, "Initial URBench evaluation pipeline").
- **OpenAI client imports** (`from openai import OpenAI`), all in `scripts/`:
  - `translation.py`
  - `translate_csqa_1500.py`
  - `translate_boolq_1550.py`
  - `translate_piqa_750_baber.py`
  - `translate_gsm8k_main_700.py`
  - `repair_boolq_errors.py`
  - `repair_piqa_errors.py`
- **No hard-coded keys:** there are no literal `api_key="…"` assignments in `scripts/` or `eval/`, and no `api_key` or `base_url` mentions at all.
- **`import anthropic` / `from anthropic`: never used**, either in the current tree or in git history.
- **Installed packages** in `urbench_eval`: `anthropic 0.71.0` is installed but unused; `openai 2.15.0` is installed; `dashscope` is not installed.
- **This shell:** none of the three variable names is set (I checked names only).

---

## 5. vLLM

- **Version:** `vllm 0.13.0` in `urbench_eval`, from `importlib.metadata`. The same version appears in the 96151 engine banner (`V1 LLM engine (v0.13.0)`) and its SEAL. Related versions: torch 2.9.0, transformers 4.57.6, xgrammar 0.1.27. autoawq and flash_attn are not installed.
- **137 files import vLLM.**
  - `tensor_parallel_size=1` appears 81 times. The only other value is `tensor_parallel_size=args.tp` in `efbpt/efbpt_plan_a_draft.py:430` (default `--tp 1`, model Qwen3-14B).
  - No TP above 1 anywhere, and no `tensor-parallel-size` flag.
  - **No `gptq` anywhere** in code.
  - `quantization="awq"` appears 17 times.
  - `max_model_len` values: 4096 (95), 8192 (33), 16384 (4), 6144 (2), 3072 (1), 2048 (1).
- **Serving a model above 14B with vLLM: yes, but only Llama-3.1-70B-Instruct-AWQ-INT4, on a single GPU.** Two groups of scripts do it:
  - 15 legacy scripts, `eval/eval_{boolq,csqa,gsm8k,piqa,strategyqa}_{cot,three_shot,zero_shot}_vllm.py`.
    - They use `MODEL_NAME = ".../Meta-Llama-3.1-70B-Instruct-AWQ-INT4"` (the symlink), with `tensor_parallel_size=1`, `quantization="awq"`, and `MAX_MODEL_LEN = 4096`.
    - For example, `eval_strategyqa_cot_vllm.py` sets `gpu_memory_utilization=0.90` and `max_num_seqs=1`.
    - Their outputs exist in `outputs/{boolq,csqa,gsm8k,piqa,strategyqa}/llama3.1_70b_awq/` (3 files each).
  - `efbpt/efbpt_qea_triage_v1.py` `VllmBackend`: TP 1, awq, fp16, enforce_eager, max_model_len 4096, gpu_memory_utilization 0.95 (smoke and T1/T2), max_num_seqs 4. In job 96151 this left 3.05 GiB of KV cache (10,000 tokens, 2.44× concurrency at 4,096 tokens).
- **Gemma-2-27B is never served with vLLM.** It loads only through HF transformers + bitsandbytes NF4 (`HfBnbBackend`, `efbpt_qea_triage_v1.py:995-1036`).

---

## 6. D1 arm G (leakage-probe reference)

- **File:** `outputs/efbpt/d1/arms/d1_armG.jsonl` exists, 200 lines, 200 unique qids, MD5 `cf5e5d57a39e361954073812fb8d15c0`.
- **Arm spec** (`d1_eval_arms.py` `ARM_SPEC`): `"G": (None, "gold")`, which means no question and gold facts only.
- **Recomputed from jsonl** (matches `d1_summary.json` arm G):

| metric | value |
|---|---|
| accuracy | **42.00%** (84/200) |
| unparsed | 33.5% (67) |
| truncated | 0.5% (1) |
| predicted yes | 37.0% (74) |
| floor (always no) | 56.5% |
| G − floor | −14.5 pp |
| accuracy on the 133 parsed rows | 63.16% (84/133) |
| G vs D1 arm A (paired, exact two-sided) | 40 wins, 71 losses, p = 0.0042 |

Note: in D1 the leak check is G − floor. G is 14.5 points below the floor, mainly because one third of its outputs are unparsed. On the parsed rows alone, G is 6.7 points above the floor.
