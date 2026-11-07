#!/usr/bin/env python3
"""
d11r_local_readers.py — D11-R local readers L1–L4 (docs/EFBPT_D11R_READER_FREEZE.md §2).

  llama    L1  Llama-3.1-70B-Instruct-AWQ-INT4 via vLLM, D10 §2 engine args
               (d10_teacher.VLLM_ARGS / SAMPLING, asserted equal to the pinned
               copy below), greedy, max_tokens 128
  gemma    L2  Gemma-2-27B-it, NF4 double-quant bf16 compute, the QEA
               HfBnbBackend load (efbpt_qea_triage_v1.py:1008-1016, replicated),
               greedy, SYSTEM text prepended to the user turn (no system role),
               max_new 128, stop on <eos> and <end_of_turn>
  qwen14b  L3  Qwen3-14B nf4, the D10 student load (d1_eval_arms.py:215-221),
               greedy, enable_thinking=False, max_new 128
  nllb     L4  NLLB-200-3.3B from downloaded_models/nllb-200-3.3B (12 files
               MD5-pinned below, copied 2026-11-07 from
               /mnt/home/user49/junhaoliu/models/nllb-200-3.3B), fp16,
               urd_Arab -> eng_Latn, beam 4, max_length 128

Every reader reads the three item sets (rdev300, dev200, eval458) and writes
the common record schema (d11r_common.make_record) to
outputs/efbpt/d11r/<reader>/readings_<set>.jsonl, plus R_START.json and
R_SEAL.json. Reader prompts contain only question_ur (§5): the loader is
called with with_gold=False. HF readers run at batch size 1 (greedy,
deterministic); vLLM runs with the D10 max_num_seqs.

TEST: --limit N reads the first N rows of each set and writes only under
outputs/efbpt/d11r/_test/<reader>/.

Usage (from ~/URBench, env urbench_eval, on a GPU node):
  python eval/error_analysis_tests/efbpt/d11r_local_readers.py --readers llama
  python eval/error_analysis_tests/efbpt/d11r_local_readers.py --readers gemma qwen14b nllb
  python eval/error_analysis_tests/efbpt/d11r_local_readers.py --readers nllb --limit 3
"""
import argparse
import gc
import json
import os
import socket
import sys
import time
from collections import OrderedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from d11r_common import (  # noqa: E402
    ROOT, LOCAL_READERS, LOCAL_MAX_NEW, SYSTEM_PROMPT, USER_TEMPLATE, ITEM_SETS,
    die, verify_freeze, load_items, md5_file, make_record, reader_dir, readings_path, append_jsonl,
    write_start, update_seal, done_qids, compact_jsonl, git_state,
)

os.chdir(ROOT)

LLAMA_PATH = "/mnt/home/user41/downloaded_models/LLM-Research/Meta-Llama-3___1-70B-Instruct-AWQ-INT4"
# D10 §2 (d10_teacher.py:89-99), asserted against the module at load
LLAMA_VLLM_ARGS = OrderedDict([
    ("tensor_parallel_size", 1), ("max_model_len", 4096), ("quantization", "awq"),
    ("gpu_memory_utilization", 0.90), ("max_num_seqs", 1), ("enforce_eager", True),
    ("trust_remote_code", True), ("disable_log_stats", True),
])
LLAMA_SAMPLING = OrderedDict([("temperature", 0.0), ("top_p", 1.0), ("seed", 0)])

GEMMA_PATH = "/mnt/home/user41/downloaded_models/AI-ModelScope/gemma-2-27b-it"
QWEN_PATH = "/mnt/home/user41/downloaded_models/Qwen/Qwen3-14B"
NLLB_PATH = "/mnt/home/user41/downloaded_models/nllb-200-3.3B"
NLLB_SRC_LANG, NLLB_TGT_LANG = "urd_Arab", "eng_Latn"
NLLB_BEAMS, NLLB_MAX_LENGTH = 4, 128
# the 12 repo files, MD5 of the copy == MD5 of the source (verified 2026-11-07)
NLLB_MD5 = OrderedDict([
    (".gitattributes", "5a36cfd0d6a096c9479e1a0d2b19860c"),
    ("README.md", "67d5eef3cbbf558a813c994713dc5d76"),
    ("config.json", "799b8dc12c1c0e3945393f51a3a15b9a"),
    ("generation_config.json", "d02f9e936c18ec442032be6958a1f887"),
    ("pytorch_model-00001-of-00003.bin", "00a270667cd56865fa2e513edc8f82ff"),
    ("pytorch_model-00002-of-00003.bin", "bf769074dccfe9cff4f2e435aed70308"),
    ("pytorch_model-00003-of-00003.bin", "e7d757298a1a909aab58ed521b48a049"),
    ("pytorch_model.bin.index.json", "d1d61694c01b5fc1481eb2fa9c6dea42"),
    ("sentencepiece.bpe.model", "05c551ae7955b3980d5a9d044eb09d70"),
    ("special_tokens_map.json", "9f9c7f6c1ad369053fdecd51145baafd"),
    ("tokenizer_config.json", "aab01dbc3bc6cdba3d83169124d77923"),
    ("tokenizer.json", "3f7d9492d20a2a391cc3e24851b2a2f3"),
])


def versions():
    import importlib.metadata as im
    v = OrderedDict([("python", sys.version.split()[0])])
    for name in ("torch", "transformers", "bitsandbytes", "accelerate", "vllm", "sentencepiece", "tokenizers"):
        try:
            v[name] = im.version(name)
        except Exception:
            v[name] = None
    return v


# ----------------------------------------------------------------------------
# Readers: load() / read_all(questions) -> list of (raw_text, prompt_tokens, output_tokens, finish_reason, latency_s) / unload()
# ----------------------------------------------------------------------------
class LlamaVLLM:
    key = "llama"

    def load(self):
        import d10_teacher as T
        if OrderedDict(T.VLLM_ARGS) != LLAMA_VLLM_ARGS or OrderedDict(T.SAMPLING) != LLAMA_SAMPLING or T.TEACHER_PATH != LLAMA_PATH:
            die("D10 §2 vLLM args / sampling / path differ from the pinned copy")
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(LLAMA_PATH, trust_remote_code=True)
        t0 = time.time()
        self.llm = LLM(model=LLAMA_PATH, **LLAMA_VLLM_ARGS)
        self.load_s = time.time() - t0
        self.params = SamplingParams(temperature=LLAMA_SAMPLING["temperature"], top_p=LLAMA_SAMPLING["top_p"],
                                     seed=LLAMA_SAMPLING["seed"], max_tokens=LOCAL_MAX_NEW)

    def identity(self):
        return OrderedDict([("model_path", LLAMA_PATH), ("backend", "vllm"), ("vllm_args", LLAMA_VLLM_ARGS),
                            ("sampling", LLAMA_SAMPLING), ("max_tokens", LOCAL_MAX_NEW), ("load_seconds", round(self.load_s, 1))])

    def prompt_ids(self, q):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_TEMPLATE.format(question_ur=q)}]
        return list(self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))

    def read_all(self, questions):
        ids = [self.prompt_ids(q) for q in questions]
        t0 = time.time()
        outs = self.llm.generate([{"prompt_token_ids": i} for i in ids], self.params, use_tqdm=False)
        el = time.time() - t0
        if len(outs) != len(ids):
            die("vLLM returned %d outputs for %d prompts" % (len(outs), len(ids)))
        res = []
        for i, o in zip(ids, outs):
            c = o.outputs[0]
            res.append((c.text, len(i), len(c.token_ids), c.finish_reason, el / len(ids)))
        return res

    def unload(self):
        del self.llm
        gc.collect()


class HfCausal:
    """Shared HF greedy loop for Gemma (L2) and Qwen3-14B (L3), batch size 1."""

    def _generate(self, input_ids, stop_ids):
        import torch
        input_ids = input_ids.to(self.model.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                      do_sample=False, max_new_tokens=LOCAL_MAX_NEW,
                                      eos_token_id=sorted(stop_ids), pad_token_id=self.pad_id)
        el = time.time() - t0
        new = out[0, input_ids.shape[1]:].tolist()
        stopped = bool(new) and new[-1] in stop_ids
        n_new = len(new) - 1 if stopped else len(new)
        text = self.tok.decode([t for t in new if t not in stop_ids], skip_special_tokens=True)
        finish = "stop" if stopped else ("length" if len(new) >= LOCAL_MAX_NEW else "stop")
        return text, input_ids.shape[1], n_new, finish, el

    def unload(self):
        import torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class Gemma(HfCausal):
    key = "gemma"

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        t0 = time.time()
        # efbpt_qea_triage_v1.py:1008-1016 (QEA Amendment 2 load), replicated
        self.tok = AutoTokenizer.from_pretrained(GEMMA_PATH, local_files_only=True, trust_remote_code=False)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, local_files_only=True, trust_remote_code=False,
                                                          quantization_config=quant, device_map={"": 0},
                                                          attn_implementation="eager", use_safetensors=True,
                                                          dtype=torch.bfloat16)
        self.model.eval()
        self.load_s = time.time() - t0
        eos = self.tok.eos_token_id
        eot = self.tok.convert_tokens_to_ids("<end_of_turn>")
        if not isinstance(eot, int) or eot == self.tok.unk_token_id:
            die("Gemma tokenizer has no <end_of_turn>")
        self.stop = {eos, eot}
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else eos
        self.quant = quant.to_dict()

    def identity(self):
        return OrderedDict([("model_path", GEMMA_PATH), ("backend", "transformers+bitsandbytes"),
                            ("quantization", "nf4 double-quant, bf16 compute, bf16 load, eager attention"),
                            ("stop_ids", sorted(self.stop)), ("system_handling", "prepended to the user turn (no system role)"),
                            ("max_new_tokens", LOCAL_MAX_NEW), ("load_seconds", round(self.load_s, 1))])

    def read_all(self, questions):
        res = []
        for q in questions:
            msgs = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(question_ur=q)}]
            ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
            res.append(self._generate(ids, self.stop))
        return res


class Qwen14B(HfCausal):
    key = "qwen14b"

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from efbpt_eval_dev200 import MODEL_PATH
        if MODEL_PATH != QWEN_PATH:
            die("student MODEL_PATH differs from the pinned Qwen3-14B path")
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # d1_eval_arms.py:215-221 (D10 student load), replicated
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(QWEN_PATH, quantization_config=bnb, dtype=torch.bfloat16,
                                                          attn_implementation="sdpa", device_map={"": 0},
                                                          trust_remote_code=True)
        self.model.config.use_cache = True
        self.model.eval()
        self.load_s = time.time() - t0
        self.stop = {self.tok.eos_token_id}
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0:
            self.stop.add(im_end)
        self.pad_id = self.tok.pad_token_id

    def identity(self):
        return OrderedDict([("model_path", QWEN_PATH), ("backend", "transformers+bitsandbytes"),
                            ("quantization", "nf4 double-quant, bf16 compute, sdpa"), ("enable_thinking", False),
                            ("stop_ids", sorted(self.stop)), ("max_new_tokens", LOCAL_MAX_NEW), ("load_seconds", round(self.load_s, 1))])

    def read_all(self, questions):
        res = []
        for q in questions:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": USER_TEMPLATE.format(question_ur=q)}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            res.append(self._generate(ids, self.stop))
        return res


class NLLB:
    key = "nllb"

    def load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        for name, want in NLLB_MD5.items():
            p = os.path.join(NLLB_PATH, name)
            if not os.path.exists(p):
                die("NLLB file missing: " + p)
            got = md5_file(p)
            if got != want:
                die("NLLB %s MD5 %s != pinned %s" % (name, got, want))
        print("  [md5] 12 NLLB files verified", flush=True)
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(NLLB_PATH, src_lang=NLLB_SRC_LANG, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_PATH, dtype=torch.float16, device_map={"": 0},
                                                           local_files_only=True, use_safetensors=False)
        self.model.eval()
        self.load_s = time.time() - t0
        self.bos = self.tok.convert_tokens_to_ids(NLLB_TGT_LANG)
        if not isinstance(self.bos, int) or self.bos == self.tok.unk_token_id:
            die("NLLB tokenizer has no %s token" % NLLB_TGT_LANG)

    def identity(self):
        return OrderedDict([("model_path", NLLB_PATH), ("files_md5", NLLB_MD5), ("backend", "transformers seq2seq fp16"),
                            ("src_lang", NLLB_SRC_LANG), ("tgt_lang", NLLB_TGT_LANG), ("num_beams", NLLB_BEAMS),
                            ("max_length", NLLB_MAX_LENGTH), ("load_seconds", round(self.load_s, 1))])

    def read_all(self, questions):
        import torch
        res = []
        for q in questions:
            enc = self.tok(q, return_tensors="pt").to(self.model.device)
            t0 = time.time()
            with torch.no_grad():
                out = self.model.generate(**enc, forced_bos_token_id=self.bos, num_beams=NLLB_BEAMS,
                                          max_length=NLLB_MAX_LENGTH)
            el = time.time() - t0
            seq = out[0].tolist()
            text = self.tok.batch_decode(out, skip_special_tokens=True)[0]
            n_out = len([t for t in seq if t not in (self.tok.pad_token_id, self.tok.eos_token_id)])
            finish = "length" if len(seq) >= NLLB_MAX_LENGTH and seq[-1] != self.tok.eos_token_id else "stop"
            res.append((text, enc["input_ids"].shape[1], n_out, finish, el))
        return res

    def unload(self):
        import torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


READER_CLASSES = OrderedDict([("llama", LlamaVLLM), ("gemma", Gemma), ("qwen14b", Qwen14B), ("nllb", NLLB)])


# ----------------------------------------------------------------------------
def run_reader(key, sets, limit):
    reader = LOCAL_READERS[key]
    test = bool(limit)
    rdir = reader_dir(reader, test)
    os.makedirs(rdir, exist_ok=True)
    print("\n" + "=" * 78)
    print("D11-R local reader %s (%s)%s" % (reader, key, "  TEST limit %d" % limit if test else ""))
    print("=" * 78, flush=True)
    r = READER_CLASSES[key]()
    r.load()
    ident = r.identity()
    print("  [load] %s in %.0fs" % (ident["model_path"], ident["load_seconds"]), flush=True)
    write_start(rdir, OrderedDict([
        ("job", "R-local"), ("reader", reader), ("key", key), ("mode", "TEST" if test else "FULL"),
        ("model", ident), ("system_prompt", SYSTEM_PROMPT), ("user_template", USER_TEMPLATE),
        ("start_time", time.strftime("%Y-%m-%d %H:%M:%S")), ("hostname", socket.gethostname()),
        ("slurm_job_id", os.environ.get("SLURM_JOB_ID")), ("sets", list(sets)),
        ("script_md5", md5_file(os.path.abspath(__file__))), ("versions", versions()), ("git", git_state()),
    ]))
    for set_name in sets:
        items = load_items(set_name, with_gold=False, limit=limit)
        order = [it["qid"] for it in items]
        path = readings_path(reader, set_name, test)
        done, n_existing = done_qids(path)
        todo = [it for it in items if it["qid"] not in done]
        print("\n  set %s: %d items, %d already done, %d to read" % (set_name, len(items), len(done), len(todo)), flush=True)
        t0 = time.time()
        CH = 25
        for s in range(0, len(todo), CH):
            chunk = todo[s:s + CH]
            outs = r.read_all([it["question_ur"] for it in chunk])
            for it, (text, n_prompt, n_out, finish, lat) in zip(chunk, outs):
                rec = make_record(it["qid"], set_name, reader, it["question_ur"], text, finish, n_prompt, n_out,
                                  0, "local_no_reasoning", lat, True, None,
                                  extra=OrderedDict([("mode", "test" if test else "full"), ("key", key)]))
                append_jsonl(path, rec)
            el = time.time() - t0
            dn = min(s + CH, len(todo))
            print("    %d/%d  (%.0fs, %.2fs/row)  e.g. %r" % (dn, len(todo), el, el / dn, outs[0][0][:80]), flush=True)
        n_final, missing = compact_jsonl(path, order)
        if missing:
            die("%s %s: %d qids missing after the run" % (reader, set_name, len(missing)))
        update_seal(rdir, reader, set_name, path, len(order), extra=OrderedDict([("mode", "TEST" if test else "FULL")]))
    r.unload()
    print("reader %s done (%s)" % (reader, ", ".join(sets)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readers", nargs="+", required=True, choices=list(READER_CLASSES), help="run in this order, one process")
    ap.add_argument("--sets", nargs="+", default=list(ITEM_SETS), choices=list(ITEM_SETS))
    ap.add_argument("--limit", type=int, default=None, help="TEST: first N rows per set, written under _test/")
    args = ap.parse_args()
    verify_freeze()
    print("[freeze] verified; readers %s; sets %s" % (args.readers, args.sets), flush=True)
    for key in args.readers:
        run_reader(key, args.sets, args.limit)


if __name__ == "__main__":
    main()
